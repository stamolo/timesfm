import streamlit as st
import tempfile
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
from pred256_3 import process_csv, CONFIG

st.title("Распознавание событий и визуализация временных рядов")

# Панель настроек параметров
st.sidebar.header("Настройки параметров")
segment_length = st.sidebar.number_input(
    "Длина сегмента (segment_length)",
    value=CONFIG["INPUT_LENGTH"],
    step=100
)
output_length = st.sidebar.number_input(
    "Длина центрального участка (output_length)",
    value=CONFIG["OUTPUT_LENGTH"],
    step=100
)
csv_sep = st.sidebar.text_input("Разделитель CSV (sep)", value=";")
csv_decimal = st.sidebar.text_input("Десятичный разделитель (decimal)", value=",")

# Загрузка файлов
st.sidebar.header("Загрузка файлов")
uploaded_csv = st.sidebar.file_uploader("Выберите входной CSV файл", type=["csv"])
uploaded_model = st.sidebar.file_uploader("Выберите модель", type=["pt"])
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

        # Временный файл для результата
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

            # Получаем список фич (исключая столбец с метками)
            feature_cols = [col for col in df_out.columns if col != "predicted_label"]

            # Пользователь выбирает фичи, которые будут отображаться на дополнительной оси
            secondary_features = st.multiselect(
                "Выберите параметры для дополнительной оси Y (остальные отобразятся по основной оси)",
                options=feature_cols,
                default=[]
            )
            # Фичи для основной оси – те, что не выбраны для дополнительной оси
            primary_features = [f for f in feature_cols if f not in secondary_features]

            # Создание интерактивного графика с двумя строками:
            # - Верхняя: все фичи с разделением по оси (primary и secondary)
            # - Нижняя: предсказанные метки
            fig = sp.make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.07,
                specs=[[{"secondary_y": True}],
                       [{}]],
                subplot_titles=("Временные ряды (фичи)", "Предсказанные события")
            )

            # Верхний график: добавляем фичи
            for feature in primary_features:
                fig.add_trace(
                    go.Scatter(
                        x=df_out.index, y=df_out[feature],
                        mode="lines", name=f"{feature} (основная)"
                    ),
                    row=1, col=1, secondary_y=False
                )
            for feature in secondary_features:
                fig.add_trace(
                    go.Scatter(
                        x=df_out.index, y=df_out[feature],
                        mode="lines", name=f"{feature} (доп.)"
                    ),
                    row=1, col=1, secondary_y=True
                )

            fig.update_yaxes(title_text="Основная ось Y", row=1, col=1, secondary_y=False)
            fig.update_yaxes(title_text="Дополнительная ось Y", row=1, col=1, secondary_y=True)

            # Нижний график: предсказанные метки (используем step-график для дискретных значений)
            if "predicted_label" in df_out.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df_out.index, y=df_out["predicted_label"],
                        mode="lines", name="Предсказанная метка",
                        line_shape="hv"
                    ),
                    row=2, col=1
                )
                fig.update_yaxes(title_text="Метка класса", row=2, col=1)
                fig.update_xaxes(title_text="Время", row=2, col=1)

            fig.update_layout(
                height=700,
                title_text="Интерактивная визуализация результатов",
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)

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
