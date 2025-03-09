import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

st.set_page_config(page_title="Фильтрация временного ряда", layout="wide")

st.title("Фильтрация временного ряда с синхронизированным зумом и экспортом")

st.markdown(
    """
    Загрузите CSV-файл с временным рядом. Если в файле один столбец – он используется как данные;
    если столбцов больше, выбирается второй столбец.

    **Настройки:**

    - **Процент наименее амплитудных гармоник**, которые будут обнулены.
    - **Десятичный разделитель** (например, `.` или `,`).
    - **Округление** – количество знаков после запятой для округления значений (0 – округление до целых).
    """
)

# Загрузка файла
uploaded_file = st.file_uploader("Выберите CSV-файл с временным рядом", type="csv")

# Виджеты для ввода параметров
percentage = st.number_input("Процент наименее амплитудных гармоник для фильтрации (%)",
                             min_value=0.0, max_value=100.0, value=0.0, step=0.1)
decimal_sep = st.text_input("Десятичный разделитель", value=".")
round_digits = st.number_input("Количество знаков после запятой для округления",
                               min_value=0, value=2, step=1)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, decimal=decimal_sep)
    except Exception as e:
        st.error(f"Ошибка при чтении файла: {e}")
    else:
        st.write("Загруженные данные (первые 5 строк):", df.head())

        # Выбор временного ряда: если столбец один – берем его, иначе второй столбец.
        if df.shape[1] == 1:
            series_raw = df.iloc[:, 0]
        else:
            series_raw = df.iloc[:, 1]

        # Приведение данных к числовому типу
        series_numeric = pd.to_numeric(series_raw, errors='coerce')
        if series_numeric.isna().all():
            st.error("Данные не содержат числовых значений после преобразования.")
        else:
            # Отбрасываем нечисловые значения и сбрасываем индексы
            series_numeric = series_numeric.dropna().reset_index(drop=True)
            n_points = len(series_numeric)
            x_values = np.arange(n_points)

            # Фильтрация через преобразование Фурье
            fft_coeffs = np.fft.fft(series_numeric)
            amplitudes = np.abs(fft_coeffs)

            # Обнуляем заданный процент коэффициентов с наименьшей амплитудой
            sorted_indices = np.argsort(amplitudes)
            n_remove = int(len(fft_coeffs) * percentage / 100)
            filtered_fft = fft_coeffs.copy()
            filtered_fft[sorted_indices[:n_remove]] = 0

            filtered_series = np.real(np.fft.ifft(filtered_fft))

            # Вычисление ошибки
            error = np.abs(series_numeric - filtered_series)

            # Создание единого графика с тремя подграфиками (общая ось X)
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=("Исходный временной ряд", "Отфильтрованный ряд", "Абсолютная ошибка")
            )

            # Добавление исходного ряда
            fig.add_trace(
                go.Scatter(x=x_values, y=series_numeric, mode="lines", name="Исходный ряд"),
                row=1, col=1
            )

            # Добавление отфильтрованного ряда
            fig.add_trace(
                go.Scatter(x=x_values, y=filtered_series, mode="lines", name="Отфильтрованный ряд",
                           line=dict(color="orange")),
                row=2, col=1
            )

            # Добавление графика ошибки
            fig.add_trace(
                go.Scatter(x=x_values, y=error, mode="lines", name="Ошибка",
                           line=dict(color="green")),
                row=3, col=1
            )

            # Параметр uirevision гарантирует синхронизацию зума по оси X
            fig.update_layout(uirevision="constant", height=800)
            fig.update_xaxes(title_text="Индекс", row=3, col=1)
            fig.update_yaxes(title_text="Значение", row=1, col=1)
            fig.update_yaxes(title_text="Значение", row=2, col=1)
            fig.update_yaxes(title_text="Ошибка", row=3, col=1)

            st.plotly_chart(fig, use_container_width=True)

            # Округление отфильтрованного ряда согласно настройке пользователя
            filtered_series_rounded = np.round(filtered_series, round_digits)

            # Подготовка данных для экспорта: только один столбец, без заголовка
            export_df = pd.DataFrame(filtered_series_rounded, columns=[""])

            # Экспорт в CSV через кнопку download_button.
            # Параметр header=False гарантирует отсутствие имени столбца.
            csv_buffer = io.StringIO()
            export_df.to_csv(csv_buffer, index=False, header=False)
            csv_data = csv_buffer.getvalue().encode('utf-8')

            st.download_button(
                label="Скачать отфильтрованный ряд в CSV",
                data=csv_data,
                file_name="filtered_series.csv",
                mime="text/csv"
            )
