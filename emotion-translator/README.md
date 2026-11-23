# 🎭 AI-перекладач емоцій

Веб-застосунок на **Flask**, який визначає емоцію тексту за допомогою нейромережі
**DistilBERT** (`distilbert-base-uncased-go-emotions-student`, 28 класів GoEmotions, формат ONNX)
і показує емодзі, пораду та цитату для натхнення.

## Запуск локально
```bash
python -m venv venv
venv\Scripts\activate        # Windows  (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt
python download_model.py     # завантажує модель (~68 МБ) і перевіряє SHA-256
python app.py                # http://127.0.0.1:5000
```

## API
```bash
curl -X POST http://127.0.0.1:5000/api/analyze -H "Content-Type: application/json" \
     -d "{\"text\": \"I am so happy today!\"}"
```

## Публікація на Render
1. Завантажити проєкт у репозиторій GitHub.
2. render.com → **New → Blueprint** → обрати репозиторій (налаштування беруться з `render.yaml`).
3. Після збірки застосунок доступний за адресою `https://<назва>.onrender.com`.
