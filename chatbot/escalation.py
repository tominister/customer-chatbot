import json
import random

with open("data/intents.json") as f:
    intents = json.load(f)

def get_response(intent_tag):
    for intent in intents:
        if intent["tag"] == intent_tag:
            return random.choice(intent["responses"])
    return "Sorry, I don't understand that yet."

def should_escalate(intent, confidence, threshold=0.7):
    """
    Decide whether to escalate to a human advisor.

    Args:
        intent (str): Predicted intent tag
        confidence (float): Confidence score from classifier
        threshold (float): Confidence threshold below which to escalate

    Returns:
        bool: True if escalation is needed, False otherwise
    """
    # For example, escalate if confidence is below threshold
    return confidence < threshold
