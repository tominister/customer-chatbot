import os
import json
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(ROOT, 'model')
MODEL_PATH = os.path.join(MODEL_DIR, "intent_model.h5")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
INTENTS_DIR = os.path.join(ROOT, 'data', 'intents')
PERFORMANCE_DIR = os.path.join(ROOT, 'performance')
MAX_SEQUENCE_LENGTH = 30

os.makedirs(PERFORMANCE_DIR, exist_ok=True)


def load_intents_dataset(intents_dir=INTENTS_DIR):
    import glob
    X = []
    y = []
    for path in glob.glob(os.path.join(intents_dir, "*.json")):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            items = data if isinstance(data, list) else [data]
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

    X, y_true = load_intents_dataset()
    X_proc = [x.lower() for x in X]
    seqs = tokenizer.texts_to_sequences(X_proc)
    X_pad = pad_sequences(seqs, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

    preds = model.predict(X_pad, verbose=0)
    y_pred_idx = np.argmax(preds, axis=1)
    y_pred = label_encoder.inverse_transform(y_pred_idx)

    # Collect misclassifications
    mis = []
    for text, t_true, t_pred in zip(X, y_true, y_pred):
        if t_true != t_pred:
            mis.append({"text": text, "true": t_true, "pred": t_pred})

    # Top confused pairs
    from collections import Counter
    pair_counts = Counter((m['true'], m['pred']) for m in mis)
    top_pairs = pair_counts.most_common(10)

    result = {
        "total_patterns": len(X),
        "total_misclassified": len(mis),
        "top_confused_pairs": [],
    }
    for (t_true, t_pred), cnt in top_pairs:
        examples = [m['text'] for m in mis if m['true'] == t_true and m['pred'] == t_pred][:5]
        result['top_confused_pairs'].append({
            "true": t_true,
            "pred": t_pred,
            "count": cnt,
            "examples": examples
        })

    with open(os.path.join(PERFORMANCE_DIR, 'misclassifications.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print('Wrote performance/misclassifications.json')


if __name__ == '__main__':
    main()
