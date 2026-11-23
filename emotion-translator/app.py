"""AI-перекладач емоцій — Flask backend."""
from flask import Flask, jsonify, render_template, request

from download_model import main as download_model
from emotions import EmotionClassifier

download_model()                 # гарантуємо, що файли моделі на місці
app = Flask(__name__)
app.json.ensure_ascii = False   # українські літери у JSON без \u-кодів
classifier = EmotionClassifier()  # модель завантажується в пам'ять один раз при старті

MAX_LEN = 1000
EXAMPLES = [
    "I passed my exam today, I am so happy!",
    "I miss my grandmother so much, everything feels empty.",
    "This is outrageous, they lied to us again!",
    "I'm scared about tomorrow's interview.",
    "Wow, I did not expect that at all!",
]


@app.route("/", methods=["GET", "POST"])
def index():
    result, error, text = None, None, ""
    if request.method == "POST":
        text = request.form.get("text", "").strip()
        if not text:
            error = "Введіть текст для аналізу."
        elif len(text) > MAX_LEN:
            error = f"Текст задовгий (максимум {MAX_LEN} символів)."
        else:
            result = classifier.predict(text)
    return render_template("index.html", result=result, error=error, text=text, examples=EXAMPLES)


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()
    if not text:
        return jsonify({"error": "Поле 'text' порожнє"}), 400
    if len(text) > MAX_LEN:
        return jsonify({"error": f"Максимум {MAX_LEN} символів"}), 400
    return jsonify(classifier.predict(text))


@app.route("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
