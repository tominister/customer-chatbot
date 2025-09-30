import os
import json
import string
# Resolve paths relative to the repository root so these scripts can be run
# from the repo root or from inside the train/ folder.
ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(ROOT, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "intent_model.h5")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
DATA_DIR = os.path.join(ROOT, "data")
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dropout, Dense, BatchNormalization, SpatialDropout1D, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import joblib
from tensorflow.keras.optimizers import Adam

# (ROOT/ MODEL_DIR / DATA_DIR already set above)

# Hyperparameters
MAX_VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 30
EMBEDDING_DIM = 128
EPOCHS = 60
BATCH_SIZE = 32
MIN_SAMPLES_PER_CLASS = 40
RANDOM_SEED = 42

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text


def augment_example(example: str):
    """Deterministic lightweight paraphrasing to expand small classes.
    Techniques: add question words, swap phrases, append clarifiers.
    This avoids external LLMs and keeps augmentation reproducible.
    """
    variants = []
    base = example.strip()
    if not base.endswith('?'):
        variants.append(base + '?')
    # add common question prefixes
    variants.append('how ' + base)
    variants.append('what is ' + base)
    # append short clarifiers
    variants.append(base + ' please')
    variants.append('can you explain ' + base)
    # simple swaps: move 'how' to start if present
    if base.startswith('how to'):
        variants.append(base.replace('how to', 'how do i'))
    return list(dict.fromkeys([v for v in variants if v and v != example]))

def load_all_intents(data_dir=DATA_DIR):
    all_intents = []
    for root, dirs, files in os.walk(data_dir):
        for filename in files:
            if filename.endswith(".json"):
                filepath = os.path.join(root, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    try:
                        intents_part = json.load(f)
                        # Accept either a list of intents or a single intent object
                        if isinstance(intents_part, list):
                            all_intents.extend(intents_part)
                        elif isinstance(intents_part, dict):
                            all_intents.append(intents_part)
                        else:
                            print(f"Warning: {filepath} root JSON has unexpected type: {type(intents_part)}")
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON in {filepath}: {e}")
    return all_intents

def prepare_data(intents):
    texts, labels = [], []
    # Basic extraction
    for intent in intents:
        tag = intent["tag"]
        for pattern in intent.get("patterns", []):
            texts.append(preprocess_text(pattern))
            labels.append(tag)

    # If classes are small, perform deterministic augmentation/oversampling
    from collections import defaultdict, Counter
    per_tag = defaultdict(list)
    for t, l in zip(texts, labels):
        per_tag[l].append(t)

    # Augment to MIN_SAMPLES_PER_CLASS
    for tag, examples in list(per_tag.items()):
        if len(examples) >= MIN_SAMPLES_PER_CLASS:
            continue
        i = 0
        # cycle through existing examples and generate deterministic variants
        while len(per_tag[tag]) < MIN_SAMPLES_PER_CLASS:
            src = examples[i % len(examples)]
            for aug in augment_example(src):
                if len(per_tag[tag]) >= MIN_SAMPLES_PER_CLASS:
                    break
                per_tag[tag].append(preprocess_text(aug))
            i += 1

    # Flatten back to lists
    new_texts, new_labels = [], []
    for tag, exs in per_tag.items():
        for ex in exs:
            new_texts.append(ex)
            new_labels.append(tag)

    print('\nPost-augmentation label distribution:')
    for tag, cnt in Counter(new_labels).items():
        print(f'  {tag}: {cnt}')

    return new_texts, new_labels

def delete_existing_model_files():
    if os.path.exists(MODEL_DIR):
        for filename in [MODEL_PATH, TOKENIZER_PATH, LABEL_ENCODER_PATH]:
            if os.path.exists(filename):
                os.remove(filename)
                print(f"Deleted: {filename}")

def train_model():
    print("Deleting existing model files (if any)...")
    delete_existing_model_files()

    print("Loading intents...")
    intents = load_all_intents()
    texts, labels = prepare_data(intents)


    # Debug: print sample (pattern, label) pairs
    print("Sample (pattern, label) pairs:")
    for i in range(min(10, len(texts))):
        print(f"  {texts[i]!r} -> {labels[i]!r}")

    # Debug: print label distribution
    from collections import Counter
    label_counts = Counter(labels)
    print("Label distribution:")
    for tag, count in label_counts.items():
        print(f"  {tag}: {count}")

    print("Tokenizing texts...")
    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding="post")

    print("Encoding labels...")
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    num_classes = len(label_encoder.classes_)
    print(f"Number of classes: {num_classes}")
    print("Label encoder mapping:")
    for idx, tag in enumerate(label_encoder.classes_):
        print(f"  {idx}: {tag}")


    print("Building Conv1D classifier (faster for short texts)...")
    model = Sequential([
        Embedding(MAX_VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH),
        SpatialDropout1D(0.2),
        Conv1D(128, 5, activation='relu'),
        BatchNormalization(),
        GlobalMaxPooling1D(),
        Dropout(0.4),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=Adam(learning_rate=1e-3),
        metrics=["accuracy"]
    )

    # Create stratified train/validation split
    # Use a fixed random seed for reproducibility
    import numpy as _np
    _np.random.seed(RANDOM_SEED)

    X_train, X_val, y_train, y_val = train_test_split(
        padded, labels_encoded, test_size=0.2, random_state=RANDOM_SEED, stratify=labels_encoded
    )

    # Print class distribution
    import numpy as _np
    unique, counts = _np.unique(y_train, return_counts=True)
    print("Train class distribution:")
    for u, c in zip(unique, counts):
        print(f"  class {u}: {c}")
    unique_v, counts_v = _np.unique(y_val, return_counts=True)
    print("Val class distribution:")
    for u, c in zip(unique_v, counts_v):
        print(f"  class {u}: {c}")

    # Compute class weights to handle imbalance
    class_weights = compute_class_weight(
        class_weight='balanced', classes=_np.unique(labels_encoded), y=labels_encoded
    )
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    print("Using class weights:", class_weight_dict)

    print("Training model...")
    # Ensure model directory exists so the checkpoint callback can write
    os.makedirs(MODEL_DIR, exist_ok=True)
    checkpoint_cb = ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True, verbose=1)
    earlystop_cb = EarlyStopping(monitor="val_loss", patience=12, verbose=1, restore_best_weights=True)
    reduce_lr_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

    model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        class_weight=class_weight_dict,
        callbacks=[earlystop_cb, checkpoint_cb, reduce_lr_cb],
        shuffle=True
    )

    print("Saving preprocessors...")
    joblib.dump(tokenizer, TOKENIZER_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)

    print("Training complete.")

if __name__ == "__main__":
    # Always retrain — start fresh
    train_model()
