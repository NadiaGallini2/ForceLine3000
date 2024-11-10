import os
import streamlit as st
import pandas as pd
import xml.etree.ElementTree as ET
import joblib
from imblearn.over_sampling import ADASYN
import emoji
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
import unicodedata
import contractions
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from imblearn.over_sampling import ADASYN
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier




# Загрузка необходимых ресурсов NLTK
nltk.download('stopwords')
nltk.download('punkt')

# Класс для выполнения сбалансировки данных с использованием ADASYN
class ADASYNTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.adasyn = ADASYN()

    def fit(self, X, y=None):
        # Ничего не делаем в fit, так как трансформация происходит в transform
        return self

    def transform(self, X, y=None):
        # Проверяем, что y присутствует для выполнения балансировки
        if y is None:
            raise ValueError("y должен быть передан для ADASYN трансформации")
        
        # Выполняем балансировку и возвращаем X и y
        X_res, y_res = self.adasyn.fit_resample(X, y)
        return X_res, y_res

# Класс для предварительной обработки текста
class PreprocessText(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stemmer = SnowballStemmer('russian')
        self.stop_words = set()
        try:
            # Загрузка стоп-слов
            self.stop_words = set(stopwords.words('russian'))
            print(f"Stopwords loaded: {len(self.stop_words)} stop words.")
        except Exception as e:
            print(f"Error loading stopwords: {e}")

    def emojis_words(self, text):
        clean_text = emoji.demojize(text, delimiters=(" ", " "))
        clean_text = clean_text.replace(":", "").replace("_", " ")
        return clean_text

    def preprocess_text(self, text):
        if not isinstance(text, str):
            return ''
        text = re.sub('<[^<]+?>', '', text)
        text = re.sub(r'http\S+', '', text)
        text = self.emojis_words(text)
        text = text.lower()
        text = re.sub('\s+', ' ', text)
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        text = contractions.fix(text)
        text = re.sub('[^a-zA-Z0-9\s]', '', text)

        tokens = nltk.word_tokenize(text)
        if hasattr(self, 'stop_words') and self.stop_words:  # Проверка наличия stop_words
            tokens = [token for token in tokens if token not in self.stop_words]
        else:
            print("Warning: Stop words are not initialized correctly.")
        
        stem_tokens = [self.stemmer.stem(word) for word in tokens]
        return ' '.join(stem_tokens)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.Series):
            print(f"Transforming {len(X)} items...")
            return X.apply(self.preprocess_text)
        else:
            raise ValueError("Input is not a pandas Series")

 #Функция для очистки текста
def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'(_x000D_|\r|\n)', ' ', text)
    return text.strip()

# Конфигурация страницы Streamlit
st.set_page_config(layout="wide", page_title="Force Line", page_icon="⚡")

# Загрузка CSS для кастомного стиля
def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

css_file = 'style.css'
load_css(css_file)

# Навигация и приветствие
with st.sidebar:
    st.image("logo.png", use_column_width=True)
    st.title("⚡ Force Line")
    choice = st.radio("Навигация", ["Загрузка",  "Заявка", "Классификация", "Экспорт"])

# Загрузка и объединение данных
if choice == "Загрузка":
    st.title("📥 Загрузка данных")
    files = st.file_uploader("Загрузите файлы с заявками", type=["csv", "xlsx", "xls"], accept_multiple_files=True)

    def read_file(uploaded_file):
        try:
            if uploaded_file.name.endswith('.csv'):
                return pd.read_csv(uploaded_file)
            else:
                return pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"🚨 Произошла ошибка: {e}")
        return None

    dataframes = [read_file(file) for file in files if file]
    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)
        st.session_state.combined_df = combined_df
        st.download_button("📥 Скачать объединённый файл", data=combined_df.to_csv(index=False).encode('utf-8'), file_name='combined_dataset.csv', mime='text/csv')

# Создание и классификация заявки
if choice == "Заявка":
    st.title("📑 Создание новой заявки")
    st.subheader("Введите информацию о заявке")

    # Инициализация или создание пустого DataFrame, если он еще не существует
    if "combined_df" not in st.session_state:
        st.session_state.combined_df = pd.DataFrame(columns=["Тема", "Описание", "Тип оборудования", "Точка отказа", "Серийный номер"])

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
        
        # Добавляем новую строку в существующий DataFrame
        st.session_state.combined_df = st.session_state.combined_df.append(new_row, ignore_index=True)

        st.success("Заявка классифицирована успешно!")

        # Вывод таблицы с результатами
        st.write("Текущие данные после добавления заявки:")
        st.dataframe(st.session_state.combined_df)

# Классификация заявок
if choice == "Классификация":
    st.title("🔍 Классификация заявок")
    
    # Загрузка модели, если она ещё не загружена
    if 'model' not in st.session_state:
        try:
            st.session_state.model = joblib.load("D:\\ForceLine3000\\catboost_pipeline.pkl")
        except Exception as e:
            st.error(f"Ошибка при загрузке модели: {e}")
    
    # Проверка наличия данных
    if 'combined_df' in st.session_state:
        df = st.session_state.combined_df
        
        # Проверка пропущенных значений в 'Описание'
        if df['Описание'].isnull().sum() > 0:
            st.error("⚠️ В столбце 'Описание' есть пропущенные значения!")
            st.stop()

        # Очистка данных перед передачей в модель
        df['Описание'] = df['Описание'].apply(lambda x: str(x) if isinstance(x, str) else '')

        
        # Предсказание категорий для заявок
        df['predicted_category'] = st.session_state.model.predict(df['Описание'])
        st.write("Результаты классификации заявок:")
        st.dataframe(df[['Тема', 'Описание', 'predicted_category']])

    else:
        st.warning("Нет загруженных данных для классификации.")

# Экспорт данных
if choice == "Экспорт":
    if 'combined_df' in st.session_state:
        df = st.session_state.combined_df
        csv_data = df.to_csv(index=False)
        st.download_button("📂 Скачать данные в формате CSV", data=csv_data, file_name="export_data.csv", mime="text/csv")

        def to_xml(df):
            root = ET.Element("Data")
            for _, row in df.iterrows():
                entry = ET.SubElement(root, "Entry")
                for col_name, col_value in row.items():
                    col_element = ET.SubElement(entry, col_name)
                    col_element.text = str(col_value)
            return ET.tostring(root, encoding="utf-8")

        xml_data = to_xml(df)
        st.download_button("📂 Скачать данные в формате XML", data=xml_data, file_name="export_data.xml", mime="application/xml")
    else:
        st.warning("⚠️ Сначала загрузите данные.")
