from cryptography.fernet import Fernet
import csv
import dash
from dash import dcc, html
import plotly.graph_objects as go
import requests
import re
from datetime import datetime
from pandas._libs.tslibs import timestamps
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from pymorphy2 import MorphAnalyzer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
import webbrowser
from textblob import sentiments

# Загружаем необходимые ресурсы
nltk.download('punkt')
nltk.download('stopwords')

# Создание ключа шифрования (выполняется один раз)
key = Fernet.generate_key()
cipher_suite = Fernet(key)

# Зашифровываем данные перед использованием
access_token = cipher_suite.encrypt(b"vk1.a.MDy6Cw9Xgv6OovF9pNskT-rNN7ungtXIz4P76p4UTN4O1AoSKk0S3lxxc-ow7Aq6XumoaqfMKZKP1lf8KDJTyutyapuPhGCNs30xWtGpKIlsteqKruB274E85kWFW8xhSVeMoWWviNctr1tobnHIcpcD46H7eN_KyLG7HOPXf0Slb89Mo6t2czNE9GQUgtj8")
user_id = cipher_suite.encrypt(b"717406435")
brand_name = cipher_suite.encrypt(b"Apple")

# Дешифруем данные перед использованием
try:
    access_token = cipher_suite.decrypt(access_token).decode('utf-8')
    user_id = cipher_suite.decrypt(user_id).decode('utf-8')
    brand_name = cipher_suite.decrypt(brand_name).decode('utf-8')
except Exception as e:
    print(f"Ошибка при дешифровке ключей: {e}")

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
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^A-Za-z0-9а-яА-ЯёЁ\s]", "", text)  # Убираем все кроме букв и цифр
    text = text.lower()  # Приводим все к нижнему регистру
    return text

# Функция для токенизации, удаления стоп-слов и лемматизации
def preprocess_text(text):
    text = clean_text(text)
    tokens = word_tokenize(text, language='russian')
    stop_words = set(stopwords.words('russian'))
    tokens = [word for word in tokens if word not in stop_words]
    morph = MorphAnalyzer()
    tokens = [morph.parse(word)[0].normal_form for word in tokens]
    return tokens

# Naive Bayes: обучаем на текстовом векторизаторе с тремя классами
def train_naive_bayes(texts, labels):
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(texts)
    model = MultinomialNB()
    model.fit(X, labels)
    return model, vectorizer

# Функция для записи данных в CSV
def write_to_csv(data, filename='mentions_analysis.csv'):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['timestamp', 'original_text', 'processed_text', 'sentiment'])
        for row in data:
            writer.writerow(row)

# Загрузка данных и отправка запроса
response = requests.get(url, params=params)

if response.status_code == 200:
    data = response.json()

    if 'response' in data and 'items' in data['response']:
        posts = data['response']['items']
        posts = [post for post in posts if post['text'].strip()]
        labels = [0 if i % 3 == 0 else (1 if i % 3 == 1 else 2) for i in range(len(posts))]
        nb_model, nb_vectorizer = train_naive_bayes([post['text'] for post in posts], labels)

        # Создание окна приложения
        root = tk.Tk()
        root.title("Анализ упоминаний бренда VK")
        root.geometry("900x700")

        # Стили
        style = ttk.Style()
        style.configure("TButton", font=("Arial", 14), padding=10)
        style.configure("TLabel", font=("Arial", 14))

        # Прокручиваемое текстовое поле
        text_box = ScrolledText(root, wrap=tk.WORD, font=("Arial", 12), bg="#f7f7f7", fg="#333")
        text_box.tag_configure("header", font=("Arial", 12, "bold"), foreground="blue")
        text_box.tag_configure("subheader", font=("Arial", 12, "italic"), foreground="green")
        text_box.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Кнопка для обновления данных
        fetch_button = ttk.Button(root, text="Загрузить данные", command=lambda: display_data(nb_model, nb_vectorizer))
        fetch_button.pack(pady=10)

        # Функция для отображения данных в окне
        def display_data(nb_model, nb_vectorizer):
            text_box.delete(1.0, tk.END)  # Очистка поля перед выводом новых данных
            posts = data['response']['items']
            if not posts:
                text_box.insert(tk.END, "Нет данных по запросу.\n")
                return

            true_labels = [0 if i % 3 == 0 else (1 if i % 3 == 1 else 2) for i in range(len(posts))]
            predicted_labels = []
            output_data = []
            timestamps = []  # Временные метки для построения графика
            sentiments = []  # Массив с метками настроений

            for post in posts:
                text = post['text']
                timestamp = post['date']
                post_time = datetime.utcfromtimestamp(timestamp)

                timestamps.append(post_time)  # Используем объект datetime для графика

                # Обрабатываем пост
                text_box.insert(tk.END, f"\nВремя публикации: {post_time.strftime('%Y-%m-%d %H:%M:%S')}\n", "header")
                text_box.insert(tk.END, "Оригинальный текст поста:\n", "subheader")
                text_box.insert(tk.END, text + "\n\n")

                processed_text = preprocess_text(text)
                text_box.insert(tk.END, "Обработанный текст поста:\n", "subheader")
                text_box.insert(tk.END, " ".join(processed_text) + "\n\n")

                # Анализ тональности с Naive Bayes
                nb_input = nb_vectorizer.transform([text])
                nb_pred = nb_model.predict(nb_input)[0]
                predicted_labels.append(nb_pred)

                if nb_pred == 0:
                    sentiment = "Отрицательная"
                elif nb_pred == 1:
                    sentiment = "Положительная"
                else:
                    sentiment = "Нейтральная"

                sentiments.append(sentiment)

                text_box.insert(tk.END, f"Тональность (Naive Bayes): {sentiment}\n", "subheader")

                output_data.append([post_time.strftime('%Y-%m-%d %H:%M:%S'), text, " ".join(processed_text), sentiment])

                text_box.insert(tk.END, "-" * 50 + "\n\n")

            # Вычисление метрик качества
            precision = precision_score(true_labels, predicted_labels, average='weighted')
            recall = recall_score(true_labels, predicted_labels, average='weighted')
            f1 = f1_score(true_labels, predicted_labels, average='weighted')
            accuracy = accuracy_score(true_labels, predicted_labels)

            text_box.insert(tk.END, "-" * 50 + "\n")
            text_box.insert(tk.END, f"Точность (Precision): {precision:.2f}\n")
            text_box.insert(tk.END, f"Полнота (Recall): {recall:.2f}\n")
            text_box.insert(tk.END, f"F1-Мера: {f1:.2f}\n")
            text_box.insert(tk.END, f"Точность (Accuracy): {accuracy:.2f}\n")

            # Функция для открытия Dash графика в браузере
            def open_graph():
                # Запуск Dash в браузере
                webbrowser.open('http://127.0.0.1:8050')

            # Кнопка для открытия графика
            graph_button = ttk.Button(root, text="Открыть график", command=open_graph)
            graph_button.pack(pady=10)

            # Запись данных в CSV
            write_to_csv(output_data)

        # Запуск приложения
        root.mainloop()

        # Запуск Dash приложения
        app = dash.Dash(__name__)

        app.layout = html.Div([
            html.H1("Анализ настроений бренда", style={'textAlign': 'center'}),
            dcc.Graph(
                id='sentiment-graph',
                figure={
                    'data': [
                        go.Scatter(
                            x=timestamps,  # Используем datetime объекты для оси X
                            y=[{'Отрицательная': 0, 'Нейтральная': 1, 'Положительная': 2}[s] for s in sentiments],
                            mode='markers',
                            marker=dict(
                                color=[{'Отрицательная': 0, 'Нейтральная': 1, 'Положительная': 2}[s] for s in sentiments],
                                colorscale='Viridis',
                                size=10,
                                line=dict(width=2, color='black')
                            ),
                            text=sentiments,
                            hoverinfo='text'
                        )
                    ],
                    'layout': go.Layout(
                        title='Изменения настроений по времени',
                        xaxis={'title': 'Время', 'tickangle': 45, 'type': 'date'},  # Ожидаем, что ось X будет иметь тип 'date'
                        yaxis={'title': 'Настроение (0=Отрицательное, 1=Нейтральное, 2=Положительное)'}
                    )
                }
            )
        ])

        app.run_server(debug=True)
    else:
        print("Нет данных по запросу.")
else:
    print(f"Ошибка запроса: {response.status_code}")
