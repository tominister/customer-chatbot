import os
import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import joblib
try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Paths relative to repo root
ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(ROOT, 'model')
MODEL_PATH = os.path.join(MODEL_DIR, "intent_model.h5")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
INTENTS_DIR = os.path.join(ROOT, 'data', 'intents')
PERFORMANCE_DIR = os.path.join(ROOT, 'performance')
MAX_SEQUENCE_LENGTH = 30

os.makedirs(PERFORMANCE_DIR, exist_ok=True)


def load_intents_as_dataset(intents_dir=INTENTS_DIR):
    import glob, json
    X = []
    y = []
    for path in glob.glob(os.path.join(intents_dir, "*.json")):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                items = data
            else:
                items = [data]
            for it in items:
                tag = it.get("tag")
                for p in it.get("patterns", []):
                    X.append(p)
                    y.append(tag)
    return X, y


def main():
    if not os.path.exists(MODEL_PATH):
        print("Model not found at", MODEL_PATH)
        return

    model = load_model(MODEL_PATH)
    tokenizer = joblib.load(TOKENIZER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)

    X, y_true = load_intents_as_dataset()
    if len(X) == 0:
        print("No patterns found in intents directory.")
        return

    X_proc = [x.lower() for x in X]
    seqs = tokenizer.texts_to_sequences(X_proc)
    X_pad = pad_sequences(seqs, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

    preds = model.predict(X_pad, verbose=0)
    y_pred_idx = np.argmax(preds, axis=1)
    y_pred = label_encoder.inverse_transform(y_pred_idx)

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=label_encoder.classes_)

    # Save JSON report
    with open(os.path.join(PERFORMANCE_DIR, "evaluation_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Save confusion matrix figure
    if _HAS_MPL:
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion matrix')
        plt.colorbar()
        tick_marks = np.arange(len(label_encoder.classes_))
        plt.xticks(tick_marks, label_encoder.classes_, rotation=90)
        plt.yticks(tick_marks, label_encoder.classes_)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(os.path.join(PERFORMANCE_DIR, 'confusion_matrix.png'), bbox_inches='tight')
        print('Saved evaluation_report.json and confusion_matrix.png in performance/')
    else:
        print('Saved evaluation_report.json (matplotlib not installed; skipping confusion matrix image)')


if __name__ == '__main__':
    main()
