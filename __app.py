import os
import re
import streamlit as st
import pandas as pd
# from ydata_profiling import ProfileReport  # Используем ydata-profiling
import xml.etree.ElementTree as ET
import requests
from io import StringIO
import joblib  # для загрузки модели
from my_preprocessing_module import PreprocessText

# Конфигурация страницы Streamlit
st.set_page_config(layout="wide", page_title="Force Line", page_icon="⚡")

# Загрузка CSS для кастомного стиля
def load_css(file_path):
    """Загружает CSS для настройки внешнего вида приложения.""" 
    with open(file_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

css_file = 'style.css'
load_css(css_file)

# Боковая панель навигации
with st.sidebar:
    st.image("logo.png", use_column_width=True)  # Добавление логотипа
    st.title("⚡ Force Line")
    choice = st.radio("Навигация", ["Загрузка",  "Заявка", "Классификация", "Экспорт"])
    st.info("🤖 Программа для автоматической диспетчеризации заявок на основе сообщений AP, загруженного датасета и электронной почты.")

# Приветствие пользователя
st.markdown("<h1 style='color: #d51d29;'>Добро пожаловать в Force Line! ⚡</h1>", unsafe_allow_html=True)
st.markdown("👋 Здесь вы можете загружать и анализировать данные заявок, классифицировать их по типам и выделять ключевые параметры.")

# Проверка наличия загруженного набора данных
if os.path.exists('./dataset.csv'):
    df = pd.read_csv('dataset.csv', index_col=None)
else:
    df = pd.DataFrame()  # Пустой DataFrame для случаев, если данных еще нет

# Блок загрузки данных
if choice == "Загрузка":
    st.title("📥 Загрузка данных")
    files = st.file_uploader("Загрузите файлы с заявками (можно загрузить несколько файлов)", type=["csv", "xlsx", "xls"], accept_multiple_files=True)

    # Функция для чтения файлов
    def read_file(uploaded_file):
        try:
            if uploaded_file.name.endswith('.csv'):
                return pd.read_csv(uploaded_file, sep=',', index_col=None)
            else:
                return pd.read_excel(uploaded_file, index_col=None)
        except pd.errors.EmptyDataError:
            st.error("🚨 Файл пустой. Пожалуйста, загрузите другой файл.")
        except Exception as e:
            st.error(f"🚨 Произошла ошибка при загрузке {uploaded_file.name}: {e}")
        return None

    # Список для хранения загруженных датафреймов
    dataframes = []

    # Загрузка всех файлов
    if files:
        for file in files:
            df = read_file(file)
            if df is not None:
                dataframes.append(df)
                st.success(f"Файл {file.name} успешно загружен! 🎉")

        # Объединение всех загруженных датафреймов
        if dataframes:
            combined_df = pd.concat(dataframes, ignore_index=True)
            combined_file_name = 'combined_dataset.csv'
            combined_df.to_csv(combined_file_name, index=None)
            st.success("Все данные успешно объединены и сохранены в файл: combined_dataset.csv")

            # Кнопка для скачивания файла
            st.download_button(
                label="📥 Скачать объединённый файл",
                data=combined_df.to_csv(index=False).encode('utf-8'),
                file_name='combined_dataset.csv',
                mime='text/csv'
            )

            st.session_state.combined_df = combined_df
            

# Анализ данных и отбор признаков
# if choice == "Анализ":
#     st.title("🔍 Анализ данных")
#     if 'combined_df' in st.session_state:
#         df = st.session_state.combined_df
#         st.markdown("### Ключевые параметры заявок")
#         st.write("Анализируем содержание заявок и выделяем ключевые атрибуты, такие как серийный номер и тип оборудования.")

#         # Генерация отчета с помощью ydata-profiling
#         profile = ProfileReport(df, title="Pandas Profiling Report", minimal=True)
#         st.components.v1.html(profile.to_html(), height=1000, scrolling=True)
#     else:
#         st.warning("⚠️ Сначала загрузите данные.")

# Страница для создания и классификации заявки
if choice == "Заявка":
    st.title("📑 Создание новой заявки")
    st.subheader("Введите информацию о заявке")

    # Ввод темы заявки (обязательное поле)
    тема = st.text_input("Тема заявки", placeholder="Введите тему заявки", max_chars=100)

    # Ввод описания заявки (обязательное поле)
    описание = st.text_area("Описание заявки", placeholder="Опишите проблему подробно", height=200)

    # Выпадающий список для выбора типа оборудования (обязательное поле)
    тип_оборудования = st.selectbox(
        "Выберите тип оборудования", 
        ["Ноутбук", "Сервер", "Рабочая станция", "Принтер", "Другой"]
    )

    # Выпадающий список для выбора точки отказа (обязательное поле)
    точка_отказа = st.selectbox(
        "Выберите точку отказа", 
        ["Блок питания", "Материнская плата", "Матрица", "Процессор", "Другой"]
    )

    # Функция для извлечения серийного номера из текста
    def extract_serial_number(text):
        match = re.search(r'\b[A-Z0-9]{8,}\b', text)
        return match.group(0) if match else "Не указан"

    # Извлечение серийного номера из описания (обязательное поле)
    серийный_номер = extract_serial_number(описание)
    if серийный_номер == "Не указан":
        серийный_номер = st.text_input("Серийный номер", placeholder="Введите серийный номер (если не найден)")

    # Классификация заявки по типу (обязательное поле)
    if "ошибка" in описание.lower():
        тип_заявки = "Техническая"
    else:
        тип_заявки = "Общая"

    # Вывод результата
    if st.button("Классифицировать заявку"):
        st.write(f"**Результаты классификации**:")
        st.write(f"Тема заявки: {тема}")
        st.write(f"Описание заявки: {описание}")
        st.write(f"Тип оборудования: {тип_оборудования}")
        st.write(f"Точка отказа: {точка_отказа}")
        st.write(f"Серийный номер: {серийный_номер}")
        st.write(f"Тип заявки: {тип_заявки}")

        # Добавление результата в DataFrame
        new_row = {
            "Тема": тема,
            "Описание": описание,
            "Тип оборудования": тип_оборудования,
            "Точка отказа": точка_отказа,
            "Серийный номер": серийный_номер
        }
        df = df.append(new_row, ignore_index=True)

        # Сохранение обновленного DataFrame в session_state
        st.session_state.combined_df = df

        st.success("Заявка классифицирована успешно!")

        # Вывод таблицы с результатами
        st.write("Текущие данные после добавления заявки:")
        st.dataframe(df)

# Страница для классификации заявок
if choice == "Классификация":
    st.title("📌 Классификация заявок")
    
    # Загружаем модель только при необходимости
    if 'model' not in st.session_state:
        st.session_state.model = joblib.load("D:\\ForceLine3000\\catboost_pipeline.pkl", mmap_mode=None)
    
    model = st.session_state.model
    
    if 'combined_df' in st.session_state:
        df = st.session_state.combined_df

        # Преобразование данных для подачи в модель, если требуется
        # (например, если модель обучена на особом наборе признаков)
        input_data = df[["Описание", "Тип оборудования", "Точка отказа"]]  # Замените на нужные столбцы
        
        # Использование модели для предсказания классов
        df['Класс'] = model.predict(input_data)  # предположим, что модель ожидает DataFrame с нужными признаками
        
        # Отображение предсказанных классов
        st.write("Результаты классификации заявок:")
        st.write(df[['Описание', 'Класс']])
    else:
        st.warning("⚠️ Сначала загрузите данные.")


# Экспорт данных в CSV и XML
if choice == "Экспорт":
    st.title("⬇️ Экспорт данных")
    
    # Экспорт в CSV
    if 'combined_df' in st.session_state:
        df = st.session_state.combined_df
        csv_data = df.to_csv(index=False)
        st.download_button(label="📂 Скачать данные в формате CSV", data=csv_data, file_name="export_data.csv", mime="text/csv")
    
        # Экспорт в XML
        def to_xml(df):
            """Форматирование данных в XML для экспорта в 1С.""" 
            root = ET.Element("Data")
            for _, row in df.iterrows():
                entry = ET.SubElement(root, "Entry")
                for col_name, col_value in row.items():
                    col_element = ET.SubElement(entry, col_name)
                    col_element.text = str(col_value)
            return ET.tostring(root, encoding="utf-8")

        xml_data = to_xml(df)
        st.download_button(label="📂 Скачать данные в формате XML", data=xml_data, file_name="export_data.xml", mime="application/xml")

        st.info("📦 Данные готовы для импорта в 1С в формате CSV и XML.")
    else:
        st.warning("⚠️ Сначала загрузите данные.")
