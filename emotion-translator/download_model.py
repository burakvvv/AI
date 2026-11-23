"""Завантаження ML-моделі (DistilBERT, GoEmotions, ONNX) перед запуском застосунку.

Файли моделі беруться з архіву-дзеркала моделі Cohee/distilbert-base-uncased-go-emotions-onnx
і перевіряються за SHA-256, щоб на сервер не потрапив пошкоджений файл.
"""
import hashlib
import io
import os
import sys
import tarfile
import urllib.request

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
ARCHIVE_URL = (
    "https://github.com/m-doughty/LLM-Classifiers-Emotions/releases/download/"
    "binaries-cohee-goemotions-d22488bc-r1/"
    "cohee-goemotions-binaries-cohee-goemotions-d22488bc-r1.tar.gz"
)
ARCHIVE_SHA256 = "322798a3face6c83dfec3dc78f6b90cb5f37633f5284dfd74f5b4e3a0ab882c1"
FILES = {
    "model.onnx": "1a4a425c70f9dc0f2da1d0cacd6fc4c6008e85bb8961fe62de46bd9e44cde464",
    "tokenizer.json": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
    "config.json": "cc7b0c0672cfd8dd003bdf5b39e56935ea7a457c1952d1d4e4dd8b9b88530cab",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def model_ready() -> bool:
    for name, digest in FILES.items():
        path = os.path.join(MODEL_DIR, name)
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            if sha256(f.read()) != digest:
                return False
    return True


def main() -> None:
    if model_ready():
        print("Модель уже завантажена.")
        return
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("Завантаження моделі...")
    with urllib.request.urlopen(ARCHIVE_URL) as resp:
        data = resp.read()
    if sha256(data) != ARCHIVE_SHA256:
        sys.exit("Помилка: контрольна сума архіву не збігається.")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for member in tar.getmembers():
            name = os.path.basename(member.name)
            if name in FILES and member.isfile():
                content = tar.extractfile(member).read()
                if sha256(content) != FILES[name]:
                    sys.exit(f"Помилка: контрольна сума {name} не збігається.")
                with open(os.path.join(MODEL_DIR, name), "wb") as f:
                    f.write(content)
    if not model_ready():
        sys.exit("Помилка: не всі файли моделі отримано.")
    print("Модель готова.")


if __name__ == "__main__":
    main()
