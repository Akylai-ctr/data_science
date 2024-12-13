import requests
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from pymorphy2 import MorphAnalyzer

# Загружаем необходимые ресурсы
nltk.download('punkt')
nltk.download('stopwords')

# Ваш токен доступа и ID пользователя
access_token = "vk1.a.bPUtJXQc-SZWZB34zKv21jmcNM4t3Syx94pKdtLvczBBiYszurdN2Q9bluvZ9U2HqyzzkkmpBJj2dZgqV9ohUPilPpKZXnOdzeXQcT3V8b8WGSOoAbC5CDuwBV1vmoOvvB6loK-NJfonArccD-9w8S62P8pb9FmBJ0GLmyT0RaWPAGGASxGecvQDLRFO9SYI"
user_id = "717406435"  # ID пользователя или группы, для которой хотите собрать данные
brand_name = "Apple"  # Название бренда, который вас интересует

# API для поиска постов и комментариев с упоминаниями бренда
url = "https://api.vk.com/method/newsfeed.search"
params = {
    "q": brand_name,  # Поиск по упоминаниям бренда
    "access_token": access_token,
    "v": "5.131",  # Версия API
    "count": 100  # Количество записей для выборки
}


# Функция для очистки текста
def clean_text(text):
    # Удаляем ссылки, упоминания, спецсимволы
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^A-Za-z0-9а-яА-ЯёЁ\s]", "", text)  # Убираем все кроме букв и цифр

    # Приводим все к нижнему регистру
    text = text.lower()

    return text


# Функция для токенизации, удаления стоп-слов и лемматизации
def preprocess_text(text):
    text = clean_text(text)
    tokens = word_tokenize(text, language='russian')

    # Убираем стоп-слова
    stop_words = set(stopwords.words('russian'))
    tokens = [word for word in tokens if word not in stop_words]

    # Лемматизация с помощью pymorphy2
    morph = MorphAnalyzer()
    tokens = [morph.parse(word)[0].normal_form for word in tokens]

    return tokens


# Отправляем запрос
response = requests.get(url, params=params)

# Проверяем статус ответа
if response.status_code == 200:
    data = response.json()

    # Проверяем, есть ли данные в ответе
    if 'response' in data and 'items' in data['response']:
        # Обрабатываем и выводим полученные данные
        for post in data['response']['items']:
            text = post['text']  # Получаем текст поста
            print("Оригинальный текст поста:")
            print(text)  # Печать текста без предобработки
            processed_text = preprocess_text(text)  # Применяем предобработку
            print("\nОбработанный текст поста:")
            print(processed_text)  # Печать обработанного текста
            print("\n" + "-" * 50)  # Разделитель между постами
    else:
        print("Нет данных по запросу.")
else:
    print(f"Ошибка запроса: {response.status_code}")





