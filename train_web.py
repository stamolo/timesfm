import streamlit as st
import os
import time
import torch
import plotly.graph_objects as go
from torchinfo import summary

# Импортируем необходимые функции и конфигурацию из main.py
from main import CONFIG, create_datasets, TransformerTimeSeriesModel, process_epoch

st.title("Обучения модели Transformer, предобученной на данных ГТИ, для распознавания произвольных событий")

# ────────────────────────────────
# Настройки обучения в боковой панели
# ────────────────────────────────
st.sidebar.header("Гиперпараметры обучения")
num_epochs = st.sidebar.number_input("Количество итераций", value=10, min_value=1, step=1)
learning_rate = st.sidebar.number_input("Скорость", value=CONFIG["LR"], format="%.6f")
batch_size = st.sidebar.number_input("Объем", value=CONFIG["BATCH_SIZE"], step=1)
use_fp16 = st.sidebar.checkbox("Использовать FP16", value=CONFIG["USE_FP16"])

st.sidebar.header("Данные для обучения")
uploaded_train_csv = st.sidebar.file_uploader("Загрузите CSV для обучения", type=["csv"])
uploaded_val_csv = st.sidebar.file_uploader("Загрузите CSV для валидации", type=["csv"])

# Добавляем кнопку для прерывания обучения
if "stop_training" not in st.session_state:
    st.session_state.stop_training = False


def stop_training_callback():
    st.session_state.stop_training = True


st.sidebar.button("Прервать обучение", on_click=stop_training_callback)

st.sidebar.markdown("Остальные настройки берутся из конфигурации (CONFIG)")

# Кнопка для старта обучения
if st.sidebar.button("Начать обучение"):
    st.info("Подготовка к обучению...")

    # Создаём папку для чекпоинтов и логов (для демонстрации используем временную папку)
    base_checkpoint_dir = CONFIG.get("BASE_CHECKPOINT_DIR", "checkpoints")
    os.makedirs(base_checkpoint_dir, exist_ok=True)
    run_folder = os.path.join(base_checkpoint_dir, f"run_{int(time.time())}")
    os.makedirs(run_folder, exist_ok=True)

    # Если CSV файлы загружены, сохраняем их во временные файлы и обновляем CONFIG
    if uploaded_train_csv is not None:
        train_csv_path = os.path.join(run_folder, "train.csv")
        with open(train_csv_path, "wb") as f:
            f.write(uploaded_train_csv.getbuffer())
        CONFIG["TRAIN_CSV"] = train_csv_path
    if uploaded_val_csv is not None:
        val_csv_path = os.path.join(run_folder, "val.csv")
        with open(val_csv_path, "wb") as f:
            f.write(uploaded_val_csv.getbuffer())
        CONFIG["VAL_TEST_CSV"] = val_csv_path

    # Обновляем CONFIG согласно выбранным гиперпараметрам
    CONFIG["NUM_EPOCHS"] = num_epochs
    CONFIG["LR"] = learning_rate
    CONFIG["BATCH_SIZE"] = batch_size
    CONFIG["USE_FP16"] = use_fp16

    st.info("Загрузка датасетов...")
    try:
        # Функция create_datasets должна вернуть train_loader, val_loader, test_loader, scaler, число фич и train_dataset
        train_loader, val_loader, test_loader, scaler, num_features, train_dataset = create_datasets(CONFIG)
    except Exception as e:
        st.error(f"Ошибка при загрузке данных: {e}")
        st.stop()

    # Создание и вывод структуры модели
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerTimeSeriesModel(
        input_channels=num_features,
        num_classes=CONFIG["NUM_CLASSES"],
        config=CONFIG
    ).to(device)
    st.subheader("Структура модели")
    st.text(summary(model, input_size=(1, CONFIG["INPUT_LENGTH"], num_features)))

    # Подготовка для обучения: определяем критерий и оптимизатор.
    if CONFIG.get("USE_CLASS_WEIGHTS", True):
        # Для простоты здесь задаём единичные веса (либо можно импортировать функцию расчёта весов)
        weights_tensor = torch.ones(CONFIG["NUM_CLASSES"], dtype=torch.float32).to(device)
    else:
        weights_tensor = torch.ones(CONFIG["NUM_CLASSES"], dtype=torch.float32).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["LR"])
    use_fp16_flag = CONFIG.get("USE_FP16", False)
    scaler_amp = torch.cuda.amp.GradScaler() if use_fp16_flag else None

    st.info("Начало обучения...")

    # Контейнер для отображения логов и прогресса
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Списки для сохранения метрик по эпохам
    epochs_list = []
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    val_f1_list = []

    # Основной цикл обучения (по эпохам)
    for epoch in range(1, num_epochs + 1):
        # Проверяем, если пользователь нажал кнопку "Прервать обучение"
        if st.session_state.stop_training:
            st.info("Обучение прервано пользователем.")
            break

        status_text.text(f"Эпоха {epoch}/{num_epochs}...")
        try:
            # Функция process_epoch из main.py выполняет обучение и валидацию за одну эпоху,
            # строит и сохраняет графики, а также сохраняет чекпоинты.
            train_loss, train_acc, val_loss, val_acc, val_f1 = process_epoch(
                epoch, num_epochs, model, train_loader, val_loader, optimizer,
                criterion, device, CONFIG, use_fp16_flag, scaler_amp, run_folder, scaler
            )
        except Exception as e:
            st.error(f"Ошибка в эпохе {epoch}: {e}")
            break

        # Сохраняем метрики для визуализации
        epochs_list.append(epoch)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        val_f1_list.append(val_f1)

        progress_bar.progress(epoch / num_epochs)

    status_text.text("Обучение завершено")

    # Построение интерактивных графиков с Plotly для потерь и точности
    st.subheader("Кривые обучения")
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(x=epochs_list, y=train_loss_list, mode='lines+markers', name='Train Loss'))
    fig_loss.add_trace(go.Scatter(x=epochs_list, y=val_loss_list, mode='lines+markers', name='Validation Loss'))
    fig_loss.update_layout(title="Потери (Loss)", xaxis_title="Эпоха", yaxis_title="Loss")
    st.plotly_chart(fig_loss, use_container_width=True)

    fig_acc = go.Figure()
    fig_acc.add_trace(go.Scatter(x=epochs_list, y=train_acc_list, mode='lines+markers', name='Train Accuracy'))
    fig_acc.add_trace(go.Scatter(x=epochs_list, y=val_acc_list, mode='lines+markers', name='Validation Accuracy'))
    fig_acc.update_layout(title="Точность (Accuracy)", xaxis_title="Эпоха", yaxis_title="Accuracy")
    st.plotly_chart(fig_acc, use_container_width=True)

    st.success("Обучение завершено!")
