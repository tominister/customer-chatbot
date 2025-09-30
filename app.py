from chatbot.llm import get_llm_response
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import numpy as np
import os
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import string
import random
import requests
from pathlib import Path

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret")  # set via env for production

MODEL_PATH = 'model/intent_model.h5'
TOKENIZER_PATH = 'model/tokenizer.pkl'
ENCODER_PATH = 'model/label_encoder.pkl'
DATA_DIR = "data"

MAX_SEQUENCE_LENGTH = 30  # Must match training

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text


# Lightweight rule-based keyword -> intent mapping to catch short or referential queries
KEYWORD_INTENT_MAP = {
    "stock": "stocks_info",
    "stocks": "stocks_info",
    "bond": "investment_risks",
    "bonds": "bonds_info",
    "etf": "etfs_info",
    "etfs": "etfs_info",
    "brokerage": "brokerage_account",
    "brokerage account": "brokerage_account",
    "strategy": "investment_strategies",
    "strategies": "investment_strategies",
    "risk": "investment_risks",
    "risks": "investment_risks",
    "portfolio": "portfolio_management",
    "tax": "taxes_investing",
    "investment": "how_to_invest",
    "taxes": "taxes_investing",
}


def is_referential_question(text: str) -> bool:
    t = text.strip().lower()
    short_refs = {"what is that", "what's that", "what is this", "what's this", "what do you mean", "explain that", "explain this"}
    if t in short_refs:
        return True
    # Very short questions like "what is that?" or "what's that" are handled above.
    return False


def rule_based_intent(user_input: str, last_bot: str | None = None):
    """Try a very small deterministic mapping before using the NN model.

    Returns (intent_tag or None)
    """
    u = preprocess_text(user_input)
    # 1) If the user asked a referential question (what is that) and we have the last bot reply,
    #    search for known keywords inside the last bot response.
    if is_referential_question(u) and last_bot:
        last = preprocess_text(last_bot)
        for kw, tag in KEYWORD_INTENT_MAP.items():
            if kw in last:
                return tag

    # 2) Direct keyword mapping from the user input
    for kw, tag in KEYWORD_INTENT_MAP.items():
        if kw in u:
            return tag

    return None

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

# Load model and preprocessors (fail gracefully with helpful messages)
model = None
tokenizer = None
label_encoder = None


def ensure_model_local():
    """Ensure model file exists locally. If not, attempt to download from
    MODEL_STORAGE_URL environment variable. This keeps Docker images small by
    not baking the model into the image.
    """
    model_path = Path(MODEL_PATH)
    if model_path.exists():
        return True

    url = os.environ.get("MODEL_STORAGE_URL")
    if not url:
        return False

    model_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(model_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return True
    except Exception as e:
        print(f"Model download failed: {e}")
        return False


def ensure_resources_loaded():
    """Lazy-load model and preprocessors. Returns True if loaded."""
    global model, tokenizer, label_encoder
    if model is not None and tokenizer is not None and label_encoder is not None:
        return True

    # Ensure model file is present (download if MODEL_STORAGE_URL provided)
    ensure_model_local()

    try:
        model = load_model(MODEL_PATH)
        tokenizer = joblib.load(TOKENIZER_PATH)
        label_encoder = joblib.load(ENCODER_PATH)
        return True
    except Exception as e:
        print(f"Failed to load model or preprocessors: {e}")
        model = None
        tokenizer = None
        label_encoder = None
        return False

# Load all intents from hierarchical JSONs
intents_data = load_all_intents(DATA_DIR)

# Create dict for quick lookup by tag
intents_dict = {item['tag']: item for item in intents_data}

def classify_intent(user_input):
    # Try lightweight rule-based mapping first (captures short/referential queries)
    last_bot = session.get("chat_history", [])[-1]["bot"] if session.get("chat_history") else None
    rb = rule_based_intent(user_input, last_bot=last_bot)
    if rb:
        return rb, 0.99

    # Fall back to neural model — ensure resources are loaded first
    if not ensure_resources_loaded():
        # If model not available, fallback to no-intent
        return None, 0.0

    user_input = preprocess_text(user_input)
    seq = tokenizer.texts_to_sequences([user_input])
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    preds = model.predict(padded)[0]
    confidence = float(np.max(preds))
    intent_idx = int(np.argmax(preds))
    intent = label_encoder.inverse_transform([intent_idx])[0]
    return intent, confidence

def get_response(intent, confidence, current_context):
    allowed_next = []
    if current_context and current_context in intents_dict:
        allowed_next = intents_dict[current_context].get("next", [])

    # Accept if confident and contextually allowed, or if very high confidence (rule-based override)
    if (confidence >= 0.4 and (intent == current_context or intent in allowed_next or current_context is None)) or confidence >= 0.9:
        intent_obj = intents_dict.get(intent, {})
        # If this intent is flagged for LLM, use the LLM to generate a response
        if intent_obj.get("llm", False):
            # If the prompt template expects the user_input, format it. Otherwise prefer recent user message.
            prompt_template = intent_obj.get("llm_prompt") or f"Answer as an investment assistant: {intent_obj.get('tag', intent)}"
            user_message = session.get("chat_history", [])[-1]["user"] if session.get("chat_history") else ""
            if "{user_input}" in prompt_template:
                llm_input = prompt_template.format(user_input=user_message or "")
            else:
                llm_input = user_message or prompt_template
            response = get_llm_response(llm_input)
            return response, intent
        else:
            responses = intent_obj.get("responses", [])
            return random.choice(responses), intent
    else:
        fallback_responses = intents_dict.get("fallback", {}).get("responses", ["Sorry, I didn't understand that."])
        return random.choice(fallback_responses), None

@app.route("/", methods=["GET"])
def index():
    chat_history = session.get("chat_history", [])
    return render_template("index.html", chat_history=chat_history)

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.form.get("user_input")
    if not user_input:
        return redirect(url_for("index"))

    current_context = session.get("context")

    intent, confidence = classify_intent(user_input)

    bot_response, new_context = get_response(intent, confidence, current_context)

    session["context"] = new_context if new_context else None

    if "chat_history" not in session:
        session["chat_history"] = []

    session["chat_history"].append({
        "user": user_input,
        "bot": bot_response,
        "confidence": round(confidence, 2),
        "context": session.get("context")
    })
    session.modified = True

    return redirect(url_for("index"))

@app.route("/clear", methods=["POST"])
def clear():
    session.pop("chat_history", None)
    session.pop("context", None)
    return redirect(url_for("index"))


@app.route("/health", methods=["GET"])
def health():
    """Health endpoint: returns 200 if model and preprocessors are loaded, 503 if still initializing."""
    ok = ensure_resources_loaded()
    if ok:
        return jsonify({"status": "ready"}), 200
    else:
        return jsonify({"status": "loading_or_unavailable"}), 503

if __name__ == "__main__":
    app.run(debug=True)
