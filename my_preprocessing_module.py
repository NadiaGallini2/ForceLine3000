import emoji 
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
import unicodedata
import contractions
from sklearn.base import BaseEstimator, TransformerMixin

class PreprocessText(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stemmer = SnowballStemmer('russian')

    def emojis_words(self, text):
        clean_text = emoji.demojize(text, delimiters=(" ", " "))
        clean_text = clean_text.replace(":", "").replace("_", " ")
        return clean_text

    def clean_text(self, input_text):
        clean_text = re.sub('<[^<]+?>', '', input_text)
        clean_text = re.sub(r'http\S+', '', clean_text)
        clean_text = self.emojis_words(clean_text)
        clean_text = clean_text.lower()
        clean_text = re.sub('\s+', ' ', clean_text)
        clean_text = unicodedata.normalize('NFKD', clean_text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        clean_text = contractions.fix(clean_text)
        clean_text = re.sub('[^a-zA-Z0-9\s]', '', clean_text)
        stop_words = set(stopwords.words('russian'))
        tokens = word_tokenize(clean_text)
        tokens = [token for token in tokens if token not in stop_words]
        clean_text = ' '.join(tokens)
        return clean_text

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(text)
        stop_words = set(stopwords.words('russian'))
        tokens = [word for word in tokens if word not in stop_words]
        stem_tokens = [self.stemmer.stem(word) for word in tokens]
        return ' '.join(stem_tokens)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(self.preprocess_text)
