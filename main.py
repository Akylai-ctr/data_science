import re
from datetime import datetime
from transformers import pipeline
from razdel import tokenize
from nltk.corpus import stopwords
from pymorphy2 import MorphAnalyzer
import pandas as pd
import nltk

# Загрузка необходимых ресурсов
nltk.download('stopwords')

# 1. Функции для предобработки текста
def clean_text(text):
    """
    Очищает текст от ссылок, упоминаний и символов.
    """
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # Удаление ссылок
    text = re.sub(r"@\w+", "", text)  # Удаление упоминаний
    text = re.sub(r"[^A-Za-z0-9а-яА-ЯёЁ\s]", "", text)  # Удаление лишних символов
    return text.lower()  # Приведение текста к нижнему регистру

def preprocess_text(text):
    """
    Токенизирует, удаляет стоп-слова и приводит слова к нормальной форме.
    """
    text = clean_text(text)
    tokens = [token.text for token in tokenize(text)]  # Токенизация через razdel
    stop_words = set(stopwords.words("russian"))  # Загрузка стоп-слов
    tokens = [word for word in tokens if word not in stop_words]  # Удаление стоп-слов
    morph = MorphAnalyzer()
    return [morph.parse(word)[0].normal_form for word in tokens]  # Лемматизация

def truncate_text(text, max_length=512):
    """
    Обрезает текст до максимального количества токенов.
    """
    if not isinstance(text, str):
        text = str(text)  # Преобразование в строку, если это не строка
    tokens = text.split()
    if len(tokens) > max_length:
        return " ".join(tokens[:max_length])
    return text

# Функция для преобразования временной метки
def convert_timestamp(x):
    try:
        return datetime.fromtimestamp(float(x)).strftime('%Y-%m-%d %H:%M:%S')
    except (ValueError, TypeError):
        return None

# 2. Инициализация модели тонального анализа (BERT)
def bert_sentiment_analysis(text):
    """
    Анализирует тональность текста с использованием модели BERT.
    """
    try:
        if not text.strip():
            return "Neutral", 0.0  # Если текст пустой
        text = truncate_text(text, max_length=512)
        classifier = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')
        result = classifier(text)[0]
        sentiment_label = result['label']
        confidence = result['score']
        return sentiment_label, round(confidence, 3)
    except Exception as e:
        print(f"Ошибка анализа текста: {e}")
        return "Error", 0.0

# 3. Чтение данных из CSV
def fetch_vk_posts_from_csv(filename):
    """
    Читает данные из CSV файла.
    """
    try:
        df = pd.read_csv(filename, encoding='utf-8')
        return df
    except FileNotFoundError:
        raise Exception(f"Файл {filename} не найден.")
    except pd.errors.EmptyDataError:
        raise Exception(f"Файл {filename} пуст или поврежден.")

# 4. Основной скрипт
def analyze_sentiment_from_csv(input_filename, output_filename):
    """
    Анализирует тональность постов из CSV файла и сохраняет результат в новый CSV файл.
    """
    try:
        df = fetch_vk_posts_from_csv(input_filename)

        # Преобразование типов данных столбцов с временными метками
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
            df['Post_Time'] = df['timestamp'].apply(
                lambda x: convert_timestamp(x) if pd.notnull(x) else None
            )
        elif 'date' in df.columns:
            df['date'] = pd.to_numeric(df['date'], errors='coerce')
            df['Post_Time'] = df['date'].apply(
                lambda x: convert_timestamp(x) if pd.notnull(x) else None
            )
        else:
            raise Exception("Столбец с временной меткой (timestamp или date) не найден.")

        # Анализ текста
        df['Processed_Text'] = df['original_text'].apply(lambda x: " ".join(preprocess_text(str(x))))
        df[['Sentiment', 'Sentiment_Confidence']] = df['Processed_Text'].apply(
            lambda x: pd.Series(bert_sentiment_analysis(x))
        )

        # Удаление ошибок и дублированных записей
        df = df.drop_duplicates()
        df = df[df['Sentiment'] != 'Error']

        # Вывод средней точности
        average_confidence = df['Sentiment_Confidence'].mean()
        print(f"Средняя точность анализа: {average_confidence:.3f}")

        # Сохранение результатов
        df.to_csv(output_filename, index=False, encoding='utf-8-sig')
        print(f"Результаты анализа сохранены в {output_filename}")
    except Exception as e:
        print(f"Ошибка: {e}")

# 6. Запуск
if __name__ == "__main__":
    input_filename = "mentions_analysis.csv"  # Исходный CSV файл
    output_filename = "vk_posts_with_sentiment.csv"  # Файл для сохранения результата

    # Анализируем посты
    analyze_sentiment_from_csv(input_filename, output_filename)
