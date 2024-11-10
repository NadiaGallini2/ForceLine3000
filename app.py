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
# class ADASYNTransformer(BaseEstimator, TransformerMixin):
#     def __init__(self):
#         self.adasyn = ADASYN()

#     def fit(self, X, y=None):
#         # Ничего не делаем в fit, так как трансформация происходит в transform
#         return self

#     def transform(self, X):
#         if isinstance(X, pd.Series):
#             print(f"Transforming {len(X)} items...")
#             return X.apply(self.preprocess_text)
#         else:
#             raise ValueError("Input is not a pandas Series")

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
        st.write(f"**Результаты классификации**: ")
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
        st.write("Текущие данные после добавления заявки: ")
        st.dataframe(df)

def get_predict(text):
    return model.predict(pd.Series([text]))[0] 

# Страница для классификации заявок
if choice == "Классификация":
    st.title("📌 Классификация заявок")

    # Загружаем модель только при необходимости
    if 'model' not in st.session_state:
        model_path = "D:\\ForceLine3000\\catboost_pipeline3.pkl"
        if os.path.exists(model_path):
            st.session_state.model = joblib.load(model_path)
            st.success("Модель успешно загружена!")
        else:
            st.warning("⚠️ Модель не найдена! Проверьте путь к файлу модели.")
    
    # Проверяем, если модель загружена
    if 'model' in st.session_state:
        model = st.session_state.model

        if 'combined_df' in st.session_state and not st.session_state.combined_df.empty:
            df = st.session_state.combined_df
            df['Текст'] = df['Тема']+' '+df['Описание']
            df['Класс'] = df['Текст'].apply(get_predict)
            
            # Преобразуем 'Класс' в строковый тип
            # df['Класс'] = df['Класс'].astype(str)
      
            # Отображение предсказанных классов
            st.write("Результаты классификации заявок: ")
            
            st.dataframe(df[['Текст', 'Описание', 'Класс']])
        else:
            st.warning("⚠️ Сначала загрузите данные для классификации.")
    else:
        st.warning("⚠️ Модель не загружена! Пожалуйста, загрузите модель.")

# Экспорт данных в CSV и XML
if choice == "Экспорт":
    st.title("💾 Экспорт данных")
    
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
                for col in df.columns:
                    ET.SubElement(entry, col).text = str(row[col])
            return ET.tostring(root, encoding="unicode")

        xml_data = to_xml(df)
        st.download_button(label="📂 Скачать данные в формате XML", data=xml_data, file_name="export_data.xml", mime="application/xml")
