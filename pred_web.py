import streamlit as st
import tempfile
import os
import pandas as pd
from pred256_3 import process_csv, CONFIG

st.title("Интерфейс для разметки данных с помощью предобученной модели")

# Панель настроек
st.sidebar.header("Настройки параметров")
segment_length = st.sidebar.number_input(
    "Длина сегмента (segment_length)", value=CONFIG["INPUT_LENGTH"], step=100
)
output_length = st.sidebar.number_input(
    "Длина центрального участка (output_length)", value=CONFIG["OUTPUT_LENGTH"], step=100
)
csv_sep = st.sidebar.text_input("Разделитель CSV (sep)", value=";")
csv_decimal = st.sidebar.text_input("Десятичный разделитель (decimal)", value=",")

st.sidebar.header("Загрузка файлов")
uploaded_csv = st.sidebar.file_uploader("Выберите входной CSV файл", type=["csv"])
uploaded_model = st.sidebar.file_uploader("Выберите модель TorchScript (.pt)", type=["pt"])
uploaded_scaler = st.sidebar.file_uploader("Выберите scaler (pickle, опционально)", type=["pkl"])

if st.sidebar.button("Запустить предсказание"):
    if not uploaded_csv or not uploaded_model:
        st.error("Загрузите, пожалуйста, CSV файл и модель.")
    else:
        # Сохраняем загруженные файлы во временные файлы
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_csv:
            tmp_csv.write(uploaded_csv.getbuffer())
            input_csv_path = tmp_csv.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_model:
            tmp_model.write(uploaded_model.getbuffer())
            model_path = tmp_model.name

        scaler_path = None
        if uploaded_scaler:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp_scaler:
                tmp_scaler.write(uploaded_scaler.getbuffer())
                scaler_path = tmp_scaler.name

        # Создаём временный файл для сохранения результата
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_out:
            output_csv_path = tmp_out.name

        try:
            # Вызов функции обработки
            process_csv(
                model_path=model_path,
                csv_path=input_csv_path,
                output_csv=output_csv_path,
                segment_length=segment_length,
                output_length=output_length,
                sep=csv_sep,
                decimal=csv_decimal,
                scaler_path=scaler_path,
            )
            st.success("Предсказание выполнено успешно.")

            # Чтение и отображение результатов
            df_out = pd.read_csv(output_csv_path, sep=csv_sep, decimal=csv_decimal)
            st.dataframe(df_out)

            # Кнопка для скачивания результата
            csv_data = df_out.to_csv(index=False, sep=csv_sep, decimal=csv_decimal)
            st.download_button(
                label="Скачать результаты CSV",
                data=csv_data,
                file_name="output.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Ошибка при выполнении предсказания: {e}")
        finally:
            # Очистка временных файлов
            os.remove(input_csv_path)
            os.remove(model_path)
            if scaler_path:
                os.remove(scaler_path)
            os.remove(output_csv_path)
