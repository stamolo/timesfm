import os
import torch
import numpy as np
import matplotlib
import glob

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchinfo import summary
import logging
from typing import Tuple, Any, Optional, List

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Импорт модульных функций из ранее вынесённых файлов:
from utils.model import TransformerTimeSeriesModel
from utils.save_model import save_checkpoint, save_traced_model
from utils.pre_save import save_script_copy, save_model_script, save_scaler
from utils.create_dataset import create_datasets
from utils.plot_results import plot_regression_results, plot_initial_dataset_check
from utils.messaging import send_message, send_photo

# Конфигурация
CONFIG: dict[str, Any] = {
    "INPUT_LENGTH": 120,
    "OUTPUT_LENGTH": 120,
    "WINDOW_STEP": 30,
    "CASCADE": False,
    "BATCH_SIZE": 64,
    "TRAIN_CSV": "dataset\\train_ns_kl_pred.csv",
    "VAL_TEST_CSV": "dataset\\test_ns_kl_pred.csv",
    "CSV_SETTINGS": {"sep": ";", "decimal": ","},
    "TARGET_COLUMN": 4,  # Индекс столбца, который нужно предсказывать (-1 = последний)
    # === ИЗМЕНЕНИЕ: Добавлен список столбцов для использования в качестве признаков ===
    # Если список пуст или None, используются все столбцы, кроме TARGET_COLUMN
    "FEATURE_COLUMNS": [0, 1],
    # =========================================================================
    "EMBED_SIZE": 64,
    "NHEAD": 4,
    "NUM_LAYERS": 6,
    "DIM_FEEDFORWARD": 256,
    "DROPOUT": 0.1,
    "NUM_EPOCHS": 5000,
    "LR": 0.0005,

    "KEEP_TOP_N_EPOCHS": 40,
    "COLORS": {"train": "#1f77b4", "validation": "#2ca02c", "test": "#d62728"},
    "FORCE_FEATURE_SCALING": True,
    "FORCED_FEATURE_BOUNDS": {
        0: {"min": -25, "max": 300},
        1: {"min": -25, "max": 40},
        # 2: {"min": -25, "max": 40},
        # 3: {"min": -25, "max": 40},
    },
    "USE_FP16": True,
}

# Директория для сохранения чекпоинтов (используем raw string для надежности)
BASE_CHECKPOINT_DIR: str = r"D:\models\checkpoints_k_regression"
os.makedirs(BASE_CHECKPOINT_DIR, exist_ok=True)


def get_central_slice(y: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
    """
    Вычисляет центральный срез по оси временных шагов.
    """
    center_start = (config["INPUT_LENGTH"] - config["OUTPUT_LENGTH"]) // 2
    center_end = center_start + config["OUTPUT_LENGTH"]
    return y[:, center_start:center_end]


def train_one_epoch(
        model: torch.nn.Module,
        train_loader: Any,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        device: torch.device,
        config: dict[str, Any],
        use_fp16: bool = False,
        scaler_amp: Optional[torch.cuda.amp.GradScaler] = None
) -> float:
    """
    Обучает модель за одну эпоху.
    Возвращает среднюю потерю за эпоху.
    """
    model.train()
    total_loss = 0.0

    batch_bar = tqdm(train_loader, desc="Training Batches", leave=True, dynamic_ncols=True)
    for i, (X, y) in enumerate(batch_bar):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        y_central = get_central_slice(y, config)

        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_fp16):
            output = model(X)
            loss = criterion(output.squeeze(), y_central)

        if use_fp16 and scaler_amp is not None:
            scaler_amp.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * X.size(0)
        batch_bar.set_postfix(loss=f"{loss.item():.5f}")

    avg_loss = total_loss / len(train_loader.dataset) if len(train_loader.dataset) > 0 else 0.0
    return avg_loss


def validate(
        model: torch.nn.Module,
        val_loader: Any,
        criterion: torch.nn.Module,
        device: torch.device,
        config: dict[str, Any]
) -> float:
    """
    Выполняет валидацию модели.
    Возвращает среднюю потерю.
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            y_central = get_central_slice(y, config)
            output = model(X)
            loss = criterion(output.squeeze(), y_central)
            total_loss += loss.item() * X.size(0)

    avg_loss = total_loss / len(val_loader.dataset) if len(val_loader.dataset) > 0 else float("inf")
    return avg_loss


def log_epoch(epoch: int, num_epochs: int, train_loss: float, val_loss: float) -> str:
    """
    Формирует и выводит сообщение с метриками текущей эпохи, а также отправляет его.
    """
    train_loss_str = f"{train_loss:.5f}" if not np.isnan(train_loss) else "nan"
    val_loss_str = f"{val_loss:.5f}" if not np.isnan(val_loss) else "nan"

    message = (
        f"Epoch {epoch:03}/{num_epochs} | "
        f"Train Loss: {train_loss_str} | "
        f"Val Loss: {val_loss_str}"
    )
    logger.info(message)
    try:
        send_message(message)
    except Exception as e:
        logger.error("Ошибка отправки текстового сообщения: %s", e)
    return message


def save_epoch_checkpoint(model: torch.nn.Module, run_folder: str, epoch: int, val_loss: float) -> None:
    """
    Сохраняет чекпоинт модели и выполняет трассировку модели.
    """
    checkpoint_path = os.path.join(run_folder, f"epoch_{epoch:04d}_valloss_{val_loss:.5f}.pth")
    try:
        save_checkpoint(model, checkpoint_path)
    except Exception as e:
        logger.error("Ошибка сохранения чекпоинта: %s", e)
    try:
        scripted_model_path = os.path.join(run_folder, f"epoch_{epoch:04d}_scripted.pt")
        save_traced_model(model, scripted_model_path)
    except Exception as e:
        logger.error("Ошибка при сохранении скриптованной модели: %s", e)


def plot_and_send_results(model: torch.nn.Module, train_loader: Any, val_loader: Any, scaler_obj: Any,
                          device: torch.device, config: dict[str, Any], run_folder: str, epoch: int) -> None:
    """
    Строит график результатов, сохраняет его и отправляет.
    """
    try:
        fig = plot_regression_results(
            model,
            train_loader.dataset,
            val_loader.dataset,
            scaler_obj,
            device,
            config=config
        )
        plot_path = os.path.join(run_folder, f"epoch_{epoch:04d}_results.png")
        fig.savefig(plot_path)
        plt.close(fig)
        send_photo(plot_path, caption=f"Epoch {epoch} results")
    except Exception as e:
        logger.error("Ошибка при построении или отправке графиков: %s", e)


def manage_checkpoints(
        run_folder: str,
        best_epochs: List[Tuple[float, int]],
        current_epoch: int,
        current_metric: float,
        n_to_keep: int,
) -> List[Tuple[float, int]]:
    """
    Управляет чекпоинтами, сохраняя только N лучших эпох по метрике.
    Предполагается, что чем НИЖЕ метрика (loss), тем лучше.
    """
    best_epochs.append((current_metric, current_epoch))
    best_epochs.sort(key=lambda x: x[0], reverse=False)  # Сортировка по возрастанию

    if len(best_epochs) > n_to_keep:
        worst_metric, epoch_to_delete = best_epochs.pop()  # Удаляем худший (самый большой)
        file_pattern = os.path.join(run_folder, f"epoch_{epoch_to_delete:04d}_*")
        files_to_delete = glob.glob(file_pattern)

        if not files_to_delete:
            logger.warning(
                f"Пытался удалить файлы для эпохи {epoch_to_delete}, но они не найдены (шаблон: {file_pattern}).")
        else:
            logger.info(
                f"Удаление файлов для худшей эпохи: {epoch_to_delete} (Val Loss: {worst_metric:.5f}). "
                f"Сохранено {len(best_epochs)} лучших эпох."
            )
            for file_path in files_to_delete:
                try:
                    os.remove(file_path)
                except OSError as e:
                    logger.error(f"Ошибка при удалении файла {file_path}: {e}")

    return best_epochs


def process_epoch(
        epoch: int,
        num_epochs: int,
        model: torch.nn.Module,
        train_loader: Any,
        val_loader: Any,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        device: torch.device,
        config: dict[str, Any],
        use_fp16: bool,
        scaler_amp: Optional[torch.cuda.amp.GradScaler],
        run_folder: str,
        scaler_obj: Any,
        best_epochs: List[Tuple[float, int]]
) -> Tuple[float, float, List[Tuple[float, int]]]:
    """
    Выполняет обучение, валидацию, логирование, построение графиков и сохранение модели для одной эпохи.
    """
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, config, use_fp16,
                                 scaler_amp)

    if np.isnan(train_loss):
        logger.error(f"Обучение на эпохе {epoch} прервано из-за NaN loss.")
        return train_loss, float('nan'), best_epochs

    val_loss = validate(model, val_loader, criterion, device, config)

    log_epoch(epoch, num_epochs, train_loss, val_loss)
    plot_and_send_results(model, train_loader, val_loader, scaler_obj, device, config, run_folder, epoch)
    save_epoch_checkpoint(model, run_folder, epoch, val_loss)

    n_to_keep = config.get("KEEP_TOP_N_EPOCHS", 0)
    if n_to_keep > 0:
        best_epochs = manage_checkpoints(
            run_folder=run_folder,
            best_epochs=best_epochs,
            current_epoch=epoch,
            current_metric=val_loss,
            n_to_keep=n_to_keep,
        )

    return train_loss, val_loss, best_epochs


def train_model_loop(
        model: torch.nn.Module,
        train_loader: Any,
        val_loader: Any,
        device: torch.device,
        scaler_obj: Any,
        run_folder: str,
        config: dict[str, Any]
) -> torch.nn.Module:
    """
    Основной цикл обучения, использующий process_epoch для обработки каждой эпохи.
    """
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["LR"])

    use_fp16 = config.get("USE_FP16", False) and device.type == 'cuda'
    scaler_amp = torch.cuda.amp.GradScaler(enabled=use_fp16)

    best_epochs: List[Tuple[float, int]] = []
    num_epochs = config["NUM_EPOCHS"]
    for epoch in range(1, num_epochs + 1):
        logger.info("Начало эпохи %d/%d", epoch, num_epochs)

        train_loss, _, best_epochs = process_epoch(
            epoch, num_epochs, model, train_loader, val_loader, optimizer,
            criterion, device, config, use_fp16, scaler_amp, run_folder, scaler_obj,
            best_epochs
        )

        if np.isnan(train_loss):
            logger.error("Обнаружен NaN loss. Обучение остановлено.")
            break

    return model


# Основной блок
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Используемое устройство: %s", device)
    run_folder = os.path.join(BASE_CHECKPOINT_DIR, f"{len(os.listdir(BASE_CHECKPOINT_DIR)):04d}")
    os.makedirs(run_folder, exist_ok=True)
    logger.info("Чекпоинты будут сохранены в: %s", run_folder)

    try:
        current_script = os.path.abspath(__file__)
        save_script_copy(current_script, run_folder)
        model_script_path = os.path.join(os.path.dirname(current_script), "utils", "model.py")
        save_model_script(model_script_path, run_folder)
    except Exception as e:
        logger.error("Ошибка при сохранении копий скриптов: %s", e)

    # === НАЧАЛО ИЗМЕНЕНИЯ: Построение графика для проверки данных ===
    try:
        logger.info("--- НАЧАЛО ПРОВЕРКИ ДАННЫХ ---")
        plot_initial_dataset_check(
            csv_path=CONFIG["TRAIN_CSV"],
            config=CONFIG,
            output_folder=run_folder
        )
        logger.info("--- ПРОВЕРКА ДАННЫХ ЗАВЕРШЕНА ---")
    except Exception as e:
        logger.error("Критическая ошибка на этапе построения графика для проверки данных: %s", e)
    # === КОНЕЦ ИЗМЕНЕНИЯ ===

    try:
        train_loader, val_loader, test_loader, scaler, num_features = create_datasets(CONFIG)
        if scaler is not None:
            save_scaler(scaler, run_folder)

        model = TransformerTimeSeriesModel(
            input_channels=num_features,
            output_dim=1,  # Для регрессии выход один
            config=CONFIG
        ).to(device)
        logger.info("Параметры модели:")
        logger.info("Input shape: (batch, %d, %d)", CONFIG["INPUT_LENGTH"], num_features)
        summary(model, input_size=(1, CONFIG["INPUT_LENGTH"], num_features))

        best_model_path = os.path.join(BASE_CHECKPOINT_DIR, "best_model.pth")
        if os.path.exists(best_model_path):
            logger.info("Найден файл best_model.pth. Попытка загрузки весов...")
            try:
                incompatible_keys = model.load_state_dict(
                    torch.load(best_model_path, map_location=device),
                    strict=False
                )
                if incompatible_keys.missing_keys:
                    logger.warning(f"Отсутствующие ключи в state_dict: {incompatible_keys.missing_keys}")
                if incompatible_keys.unexpected_keys:
                    logger.warning(f"Неожиданные ключи в state_dict: {incompatible_keys.unexpected_keys}")

                if not incompatible_keys.missing_keys and not incompatible_keys.unexpected_keys:
                    logger.info("Веса из best_model.pth успешно загружены. Архитектура совпадает.")
                else:
                    logger.warning("Веса загружены, но архитектура модели отличается от сохраненной.")

            except Exception as e:
                logger.error("Критическая ошибка при загрузке весов. Обучение начнется с нуля.", exc_info=True)
        else:
            logger.info("Файл best_model.pth не найден. Обучение начнется с нуля.")

        model = train_model_loop(model, train_loader, val_loader, device, scaler, run_folder, CONFIG)

        try:
            scripted_model_path = os.path.join(run_folder, "best_model_scripted.pt")
            save_traced_model(model, scripted_model_path)
            logger.info("Сохранена финальная скриптованная (traced) модель: %s", scripted_model_path)
        except Exception as e:
            logger.error("Ошибка при трассировке модели: %s", e)

    except Exception as e:
        error_msg = f"Ошибка выполнения: {str(e)}"
        logger.error(error_msg, exc_info=True)
        try:
            send_message(error_msg)
        except Exception as inner_e:
            logger.error("Ошибка отправки сообщения об ошибке: %s", inner_e)

