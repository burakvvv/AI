"""ML-частина: класифікація емоцій тексту моделлю DistilBERT (GoEmotions, 28 класів)."""
import json
import os
import random

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")

# Переклад 28 міток GoEmotions українською + емодзі
LABELS_UA = {
    "admiration": ("захоплення", "🤩"), "amusement": ("веселощі", "😄"),
    "anger": ("гнів", "😠"), "annoyance": ("роздратування", "😒"),
    "approval": ("схвалення", "👍"), "caring": ("турбота", "🤗"),
    "confusion": ("розгубленість", "😕"), "curiosity": ("цікавість", "🧐"),
    "desire": ("бажання", "😍"), "disappointment": ("розчарування", "😞"),
    "disapproval": ("несхвалення", "👎"), "disgust": ("огида", "🤢"),
    "embarrassment": ("збентеження", "😳"), "excitement": ("піднесення", "🤩"),
    "fear": ("страх", "😨"), "gratitude": ("вдячність", "🙏"),
    "grief": ("горе", "😭"), "joy": ("радість", "😊"),
    "love": ("любов", "❤️"), "nervousness": ("тривога", "😰"),
    "optimism": ("оптимізм", "🌤️"), "pride": ("гордість", "😌"),
    "realization": ("усвідомлення", "💡"), "relief": ("полегшення", "😮‍💨"),
    "remorse": ("каяття", "😔"), "sadness": ("сум", "😢"),
    "surprise": ("здивування", "😲"), "neutral": ("нейтрально", "😐"),
}

# Групування 28 міток у базові емоції (схема Екмана з роботи GoEmotions)
GROUPS = {
    "joy": ["admiration", "amusement", "approval", "caring", "desire", "excitement",
            "gratitude", "joy", "love", "optimism", "pride", "relief"],
    "sadness": ["disappointment", "embarrassment", "grief", "remorse", "sadness"],
    "anger": ["anger", "annoyance", "disapproval"],
    "fear": ["fear", "nervousness"],
    "surprise": ["confusion", "curiosity", "realization", "surprise"],
    "disgust": ["disgust"],
    "neutral": ["neutral"],
}

# Емодзі, назва, порада і цитати для кожної базової емоції
ADVICE = {
    "joy": {
        "name": "Радість", "emoji": "😊", "color": "#f5b301",
        "advice": "Схоже, у вас гарний настрій! Поділіться радістю з кимось близьким — так її стає більше.",
        "quotes": [("Життя без свята — довга дорога без заїжджих дворів.", "Демокріт")],
    },
    "sadness": {
        "name": "Сум", "emoji": "😢", "color": "#4a7bd0",
        "advice": "Виглядає, що ви сумуєте. Дайте собі час, поговоріть з кимось, кому довіряєте, — ось цитата для натхнення:",
        "quotes": [("Без надії сподіваюсь!", "Леся Українка"),
                   ("Ні! Я жива! Я буду вічно жити!", "Леся Українка")],
    },
    "anger": {
        "name": "Гнів", "emoji": "😠", "color": "#d64545",
        "advice": "Здається, вас щось розлютило. Зробіть паузу, кілька глибоких вдихів — і лише потім вирішуйте, що робити.",
        "quotes": [("Найкращі ліки від гніву — зволікання.", "Сенека")],
    },
    "fear": {
        "name": "Страх / тривога", "emoji": "😨", "color": "#8a5cc7",
        "advice": "Відчувається тривога. Розбийте проблему на маленькі кроки і почніть з першого — так страх зменшується.",
        "quotes": [("Ми частіше страждаємо в уяві, ніж насправді.", "Сенека"),
                   ("Борітеся — поборете!", "Тарас Шевченко")],
    },
    "surprise": {
        "name": "Здивування", "emoji": "😲", "color": "#1aa39a",
        "advice": "Ого, щось вас здивувало! Це чудовий момент, щоб дізнатися щось нове.",
        "quotes": [("Подив — початок філософії.", "Платон")],
    },
    "disgust": {
        "name": "Огида", "emoji": "🤢", "color": "#5f8f2f",
        "advice": "Щось викликало у вас неприйняття. Варто відійти від цього і подбати про свої межі.",
        "quotes": [("Світ ловив мене, та не спіймав.", "Григорій Сковорода")],
    },
    "neutral": {
        "name": "Нейтрально", "emoji": "😐", "color": "#7a8088",
        "advice": "Текст звучить спокійно і нейтрально. Гарний стан, щоб зосередитися на справах.",
        "quotes": [("Життя іде і все без коректур.", "Ліна Костенко")],
    },
}

# Якщо найімовірніша мітка має ймовірність нижче порогу — вважаємо текст нейтральним
NEUTRAL_THRESHOLD = 0.15


class EmotionClassifier:
    def __init__(self, model_dir: str = MODEL_DIR):
        self.tokenizer = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=256)
        self.session = ort.InferenceSession(
            os.path.join(model_dir, "model.onnx"), providers=["CPUExecutionProvider"]
        )
        with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
            id2label = json.load(f)["id2label"]
        self.labels = [id2label[str(i)] for i in range(len(id2label))]

    def predict(self, text: str) -> dict:
        # 1. Токенізація тексту
        enc = self.tokenizer.encode(text)
        inputs = {
            "input_ids": np.array([enc.ids], dtype=np.int64),
            "attention_mask": np.array([enc.attention_mask], dtype=np.int64),
        }
        # 2. Прогін через нейромережу -> logits для 28 класів
        logits = self.session.run(None, inputs)[0][0]
        # 3. Softmax -> ймовірності
        probs = np.exp(logits - logits.max())
        probs = probs / probs.sum()
        scores = {label: float(p) for label, p in zip(self.labels, probs)}

        # 4. Топ-3 детальні емоції
        top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:3]
        details = [{"label": l, "name": LABELS_UA[l][0], "emoji": LABELS_UA[l][1],
                    "score": round(s * 100, 1)} for l, s in top]

        # 5. Базова емоція: сума ймовірностей у групі
        groups = {g: sum(scores[l] for l in labels) for g, labels in GROUPS.items()}
        if top[0][1] < NEUTRAL_THRESHOLD:
            main = "neutral"
        else:
            main = max(groups, key=groups.get)
        info = ADVICE[main]
        quote, author = random.choice(info["quotes"])
        chart = sorted(
            ({"key": g, "name": ADVICE[g]["name"], "emoji": ADVICE[g]["emoji"],
              "color": ADVICE[g]["color"], "score": round(v * 100, 1)} for g, v in groups.items()),
            key=lambda x: x["score"], reverse=True,
        )
        return {
            "text": text,
            "emotion": main,
            "name": info["name"],
            "emoji": info["emoji"],
            "color": info["color"],
            "advice": info["advice"],
            "quote": quote,
            "author": author,
            "details": details,
            "groups": chart,
        }
