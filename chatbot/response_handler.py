# chatbot/response_handler.py

def get_response(intent):
    responses = {
        "greeting": "Hello! How can I help you with your investments today?",
        "goodbye": "Goodbye! Feel free to ask if you have more questions.",
        "ask_strategy": "I recommend diversifying your portfolio with a mix of stocks and bonds.",
        "risk_assessment": "Are you comfortable with high risk for potentially higher returns?",
        # add more intents and responses here...
        "default": "Sorry, I didn't understand that. Can you please rephrase?"
    }
    return responses.get(intent, responses["default"])
