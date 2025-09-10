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
from utils.label_stats import print_label_statistics
from utils.calc_weights import calculate_class_weights
from utils.plot_results import plot_classification_results_with_labels
from utils.messaging import send_message, send_photo

# Конфигурация
CONFIG: dict[str, Any] = {
    "INPUT_LENGTH": 1200,
    "OUTPUT_LENGTH": 1100,
    "WINDOW_STEP": 200,
    "CASCADE": False,
    "BATCH_SIZE": 256,
    "TRAIN_CSV": "dataset\\train_kl.csv",
    "VAL_TEST_CSV": "dataset\\test_kl.csv",
    "CSV_SETTINGS": {"sep": ";", "decimal": ","},
    "EMBED_SIZE": 256,
    "NHEAD": 4,
    "NUM_LAYERS": 2,
    "DIM_FEEDFORWARD": 1024,
    "DROPOUT": 0.25,
    "NUM_EPOCHS": 5000,
    "LR": 0.000020,
    "NUM_CLASSES": 4,
    "USE_CLASS_WEIGHTS": True,
    "KEEP_TOP_N_EPOCHS": 4,
    "COLORS": {"train": "#1f77b4", "validation": "#2ca02c", "test": "#d62728"},
    "FORCE_FEATURE_SCALING": True,
    "FORCED_FEATURE_BOUNDS": {
        0: {"min": -25, "max": 300},
        1: {"min": -25, "max": 40},
        2: {"min": -25, "max": 300},
        3: {"min": -25, "max": 280},
    },
    "USE_FP16": True,
}

# Директория для сохранения чекпоинтов (используем raw string для надежности)
BASE_CHECKPOINT_DIR: str = r"D:\models\checkpoints_k"
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
) -> Tuple[float, float]:
    """
    Обучает модель за одну эпоху.
    Возвращает среднюю потерю и точность за эпоху.
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    batch_bar = tqdm(train_loader, desc="Training Batches", leave=True, dynamic_ncols=True)
    for X, y in batch_bar:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        y_central = get_central_slice(y, config)

        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_fp16):
            logits = model(X)
            loss = criterion(logits.reshape(-1, config["NUM_CLASSES"]), y_central.reshape(-1))

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
        preds = torch.argmax(logits, dim=2)
        total_correct += (preds == y_central).sum().item()
        total_samples += y_central.numel()

        batch_bar.set_postfix(loss=f"{loss.item():.5f}")

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    return avg_loss, accuracy


def validate(
        model: torch.nn.Module,
        val_loader: Any,
        criterion: torch.nn.Module,
        device: torch.device,
        config: dict[str, Any]
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Выполняет валидацию модели.
    Возвращает среднюю потерю, точность, объединённые предсказания и истинные метки.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            y_central = get_central_slice(y, config)
            logits = model(X)
            loss = criterion(logits.reshape(-1, config["NUM_CLASSES"]), y_central.reshape(-1))
            total_loss += loss.item() * X.size(0)
            preds = torch.argmax(logits, dim=2)
            total_correct += (preds == y_central).sum().item()
            total_samples += y_central.numel()
            all_preds.append(preds.cpu().numpy().flatten())
            all_targets.append(y_central.cpu().numpy().flatten())

    avg_loss = total_loss / total_samples if total_samples > 0 else float("inf")
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    all_preds = np.concatenate(all_preds) if total_samples > 0 else np.array([])
    all_targets = np.concatenate(all_targets) if total_samples > 0 else np.array([])
    return avg_loss, accuracy, all_preds, all_targets


def log_epoch(epoch: int, num_epochs: int, train_loss: float, train_acc: float,
              val_loss: float, val_acc: float, val_f1: float) -> str:
    """
    Формирует и выводит сообщение с метриками текущей эпохи, а также отправляет его.
    """
    message = (
        f"Epoch {epoch:03}/{num_epochs} | "
        f"Train Loss: {train_loss:.5f} | Train Acc: {train_acc * 100:.2f}% | "
        f"Val Loss: {val_loss:.5f} | Val Acc: {val_acc * 100:.2f}% | "
        f"Val F1 (macro): {val_f1:.5f}"
    )
    logger.info(message)
    try:
        send_message(message)
    except Exception as e:
        logger.error("Ошибка отправки текстового сообщения: %s", e)
    return message


def save_epoch_checkpoint(model: torch.nn.Module, run_folder: str, epoch: int, val_acc: float) -> None:
    """
    Сохраняет чекпоинт модели и выполняет трассировку модели.
    """
    checkpoint_path = os.path.join(run_folder, f"epoch_{epoch:04d}_valacc_{val_acc * 100:.2f}.pth")
    try:
        save_checkpoint(model, checkpoint_path)
    except Exception as e:
        logger.error("Ошибка сохранения чекпоинта: %s", e)
    try:
        scripted_model_path = os.path.join(run_folder, f"epoch_{epoch:04d}_scripted.pt")
        save_traced_model(model, scripted_model_path)
    except Exception as e:
        logger.error("Ошибка при сохранении скриптованной модели: %s", e)


def plot_and_send_results(model: torch.nn.Module, val_loader: Any, scaler_obj: Any,
                          device: torch.device, config: dict[str, Any], run_folder: str, epoch: int) -> None:
    """
    Строит график результатов, сохраняет его и отправляет.
    """
    try:
        fig = plot_classification_results_with_labels(
            model, val_loader.dataset, scaler_obj, device, num_examples=1, config=config
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
    Предполагается, что чем выше метрика, тем лучше.
    """
    best_epochs.append((current_metric, current_epoch))
    best_epochs.sort(key=lambda x: x[0], reverse=True)

    if len(best_epochs) > n_to_keep:
        worst_metric, epoch_to_delete = best_epochs.pop()
        file_pattern = os.path.join(run_folder, f"epoch_{epoch_to_delete:04d}_*")
        files_to_delete = glob.glob(file_pattern)

        if not files_to_delete:
            logger.warning(
                f"Пытался удалить файлы для эпохи {epoch_to_delete}, но они не найдены (шаблон: {file_pattern}).")
        else:
            logger.info(
                f"Удаление файлов для худшей эпохи: {epoch_to_delete} (Val Acc: {worst_metric:.4f}). "
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
) -> Tuple[float, float, float, float, float, List[Tuple[float, int]]]:
    """
    Выполняет обучение, валидацию, логирование, построение графиков и сохранение модели для одной эпохи.
    """
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, config, use_fp16,
                                            scaler_amp)

    val_loss, val_acc, val_preds, val_targets = validate(model, val_loader, criterion, device, config)
    try:
        from sklearn.metrics import f1_score
        val_f1 = float(f1_score(val_targets, val_preds, average="macro", zero_division=0))
    except Exception as e:
        logger.error("Ошибка при вычислении F1: %s", e)
        val_f1 = 0.0

    log_epoch(epoch, num_epochs, train_loss, train_acc, val_loss, val_acc, val_f1)
    plot_and_send_results(model, val_loader, scaler_obj, device, config, run_folder, epoch)
    save_epoch_checkpoint(model, run_folder, epoch, val_acc)

    n_to_keep = config.get("KEEP_TOP_N_EPOCHS", 0)
    if n_to_keep > 0:
        best_epochs = manage_checkpoints(
            run_folder=run_folder,
            best_epochs=best_epochs,
            current_epoch=epoch,
            current_metric=val_acc,
            n_to_keep=n_to_keep,
        )

    return train_loss, train_acc, val_loss, val_acc, val_f1, best_epochs


def train_model_loop(
        model: torch.nn.Module,
        train_loader: Any,
        val_loader: Any,
        device: torch.device,
        scaler_obj: Any,
        run_folder: str,
        train_dataset: Any,
        config: dict[str, Any]
) -> torch.nn.Module:
    """
    Основной цикл обучения, использующий process_epoch для обработки каждой эпохи.
    """
    if config.get("USE_CLASS_WEIGHTS", True):
        weights_tensor = calculate_class_weights(train_dataset.tensors[1].numpy(), config["NUM_CLASSES"], device)
    else:
        weights_tensor = torch.ones(config["NUM_CLASSES"], dtype=torch.float32).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["LR"])

    use_fp16 = config.get("USE_FP16", False) and device.type == 'cuda'
    scaler_amp = torch.cuda.amp.GradScaler(enabled=use_fp16)

    best_epochs: List[Tuple[float, int]] = []
    num_epochs = config["NUM_EPOCHS"]
    for epoch in range(1, num_epochs + 1):
        logger.info("Начало эпохи %d/%d", epoch, num_epochs)

        # Получаем метрики из process_epoch, но val_acc больше не используется для сохранения лучшей модели
        _, _, _, _, _, best_epochs = process_epoch(
            epoch, num_epochs, model, train_loader, val_loader, optimizer,
            criterion, device, config, use_fp16, scaler_amp, run_folder, scaler_obj,
            best_epochs
        )

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

    try:
        train_loader, val_loader, test_loader, scaler, num_features, train_dataset = create_datasets(CONFIG)
        if scaler is not None:
            save_scaler(scaler, run_folder)
        print_label_statistics(train_dataset.tensors[1].numpy(), CONFIG["NUM_CLASSES"])
        weights_tensor = calculate_class_weights(train_dataset.tensors[1].numpy(), CONFIG["NUM_CLASSES"], device)
        logger.info("Рассчитанные веса классов: %s", weights_tensor)

        model = TransformerTimeSeriesModel(
            input_channels=num_features,
            num_classes=CONFIG["NUM_CLASSES"],
            config=CONFIG
        ).to(device)
        logger.info("Параметры модели:")
        logger.info("Input shape: (batch, %d, %d)", CONFIG["INPUT_LENGTH"], num_features)
        logger.info("Number of classes: %d", CONFIG["NUM_CLASSES"])
        summary(model, input_size=(1, CONFIG["INPUT_LENGTH"], num_features))

        # --- НАЧАЛО ИЗМЕНЕНИЙ: УЛУЧШЕННАЯ ЗАГРУЗКА ВЕСОВ ---
        best_model_path = os.path.join(BASE_CHECKPOINT_DIR, "best_model.pth")
        if os.path.exists(best_model_path):
            logger.info("Найден файл best_model.pth. Попытка загрузки весов...")
            try:
                # Загружаем веса с strict=False, чтобы избежать падения при несовпадении ключей
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
        # --- КОНЕЦ ИЗМЕНЕНИЙ ---

        model = train_model_loop(model, train_loader, val_loader, device, scaler, run_folder, train_dataset, CONFIG)

        # Этот блок больше не нужен, так как лучшая модель уже сохраняется в цикле
        # best_model_run_path = os.path.join(run_folder, "best_model.pth")
        # if os.path.exists(best_model_run_path):
        # ...

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

