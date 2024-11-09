import os
import re
import streamlit as st
import pandas as pd
from pandas_profiling import ProfileReport
import xml.etree.ElementTree as ET

# Конфигурация страницы Streamlit
st.set_page_config(layout="wide", page_title="Force Line", page_icon="📧")

# Загрузка CSS для кастомного стиля
def load_css(file_path):
    """Загружает CSS для настройки внешнего вида приложения.""" 
    with open(file_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

css_file = 'style.css'
load_css(css_file)

# Боковая панель навигации
with st.sidebar:
    st.title("📧 Force Line")
    choice = st.radio("Навигация", ["Загрузка", "Анализ", "Классификация", "Экспорт"])
    st.info("🤖 Программа для автоматической диспетчеризации заявок на основе сообщений электронной почты.")

# Приветствие пользователя
st.markdown("<h1 style='color: #d51d29;'>Добро пожаловать в Force Line! 📧</h1>", unsafe_allow_html=True)
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
if choice == "Анализ":
    st.title("🔍 Анализ данных")
    if 'combined_df' in st.session_state:
        df = st.session_state.combined_df
        st.markdown("### Ключевые параметры заявок")
        st.write("Анализируем содержание заявок и выделяем ключевые атрибуты, такие как серийный номер и тип оборудования.")
        profile = ProfileReport(df, minimal=True)
        st.components.v1.html(profile.to_html(), height=1000, scrolling=True)
    else:
        st.warning("⚠️ Сначала загрузите данные.")

# Классификация заявок
if choice == "Классификация":
    st.title("📌 Классификация заявок")
    if 'combined_df' in st.session_state:
        df = st.session_state.combined_df

        # Выделение ключевых параметров из текста заявки
        def extract_serial_number(text):
            match = re.search(r'\b[A-Z0-9]{8,}\b', text)
            return match.group(0) if match else "Не указан"

        # Используем столбец 'Описание' вместо 'Сообщение'
        if 'Описание' in df.columns:
            df['Серийный номер'] = df['Описание'].apply(extract_serial_number)
            df['Тип заявки'] = df['Описание'].apply(lambda x: "Техническая" if "ошибка" in x.lower() else "Общая")
            df['Тип оборудования'] = df['Описание'].apply(lambda x: "Сервер" if "сервер" in x.lower() else "Рабочая станция")

            st.write("Первые 5 заявок после обработки:")
            st.write(df.head())
        else:
            st.warning("⚠️ Столбец 'Описание' не найден в данных. Пожалуйста, убедитесь, что файл содержит нужные столбцы.")

    else:
        st.warning("⚠️ Сначала загрузите данные.")

# Экспорт данных в CSV и XML
if choice == "Экспорт":
    st.title("⬇️ Экспорт данных")
    
    # Экспорт в CSV
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
