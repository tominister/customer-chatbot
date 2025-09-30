import json
import os
import string
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "intent_model.h5")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
INTENTS_PATH = "data/intents.json"

MAX_SEQUENCE_LENGTH = 30  # must match train.py

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Load model and preprocessors
model = load_model(MODEL_PATH)
tokenizer = joblib.load(TOKENIZER_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

# Load intents JSON for response retrieval
with open(INTENTS_PATH, "r", encoding="utf-8") as f:
    intents_data = json.load(f)

intents_dict = {intent['tag']: intent for intent in intents_data}

# Minimal keyword map (same semantics as app.py) to catch short queries
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


def rule_based_intent(user_input: str, last_bot: str | None = None):
    u = preprocess_text(user_input)
    # If user asked a short referential question and we have last bot message, try mapping from it
    short_refs = {"what is that", "what's that", "what is this", "what's this", "what do you mean", "explain that", "explain this"}
    if u.strip() in short_refs and last_bot:
        last = preprocess_text(last_bot)
        for kw, tag in KEYWORD_INTENT_MAP.items():
            if kw in last:
                return tag

    for kw, tag in KEYWORD_INTENT_MAP.items():
        if kw in u:
            return tag

    return None

def classify_intent(user_input):
    # Expose base NN classification as before
    user_input_proc = preprocess_text(user_input)
    seq = tokenizer.texts_to_sequences([user_input_proc])
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    preds = model.predict(padded)[0]
    confidence = float(np.max(preds))
    intent_idx = int(np.argmax(preds))
    intent_tag = label_encoder.inverse_transform([intent_idx])[0]
    return intent_tag, confidence


def classify_with_rules(user_input, last_bot=None):
    """Try rule-based intents first for short or referential queries, then fall back to NN."""
    rb = rule_based_intent(user_input, last_bot=last_bot)
    if rb:
        # Using high confidence for rule matches so downstream logic treats it deterministically
        return rb, 0.99
    return classify_intent(user_input)

def get_response(intent, confidence, current_context=None):
    allowed_next = []
    if current_context and current_context in intents_dict:
        allowed_next = intents_dict[current_context].get("next", [])
    
    if confidence >= 0.4 and (intent == current_context or intent in allowed_next or current_context is None):
        responses = intents_dict.get(intent, {}).get("responses", [])
        if responses:
            return np.random.choice(responses), intent
    fallback_responses = intents_dict.get("fallback", {}).get("responses", ["Sorry, I didn't understand that."])
    return np.random.choice(fallback_responses), None

if __name__ == "__main__":
    while True:
        inp = input("You: ")
        intent, conf = classify_intent(inp)
        response, _ = get_response(intent, conf)
        print(f"Intent: {intent}, Confidence: {conf:.2f}")
        print(f"Bot: {response}")
