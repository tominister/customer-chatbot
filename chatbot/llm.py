import os
from dotenv import load_dotenv
load_dotenv()

# Defer importing heavy/possibly-incompatible LLM client libraries until
# runtime so the Flask app can start even if the environment doesn't have
# the required OpenAI/Groq packages or has dependency conflicts (common when
# TensorFlow pins typing-extensions).

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_API_BASE = os.environ.get("GROQ_API_BASE", "https://api.groq.com/openai/v1")


def get_llm_response(prompt, system_prompt="You are a helpful investment assistant.", model="llama2-70b-4096"):
    """Return an LLM-generated response or a helpful placeholder if the LLM
    client or API key is not available.

    This function performs a lazy import of the OpenAI client. If importing
    fails (dependency conflicts) or the API key is not set, it returns a
    clear string that the app can show to the user instead of raising on
    import.
    """
    if not GROQ_API_KEY:
        return "[LLM disabled: set GROQ_API_KEY in your environment to enable the LLM.]"

    try:
        import openai
    except Exception as e:  # pragma: no cover - environment-specific
        return f"[LLM unavailable: failed to import OpenAI client: {e}]"

    try:
        openai.api_key = GROQ_API_KEY
        openai.api_base = GROQ_API_BASE
        # Use ChatCompletion for compatibility with openai-python API surface
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=256,
            temperature=0.7,
        )
        # Defensive access to nested response fields
        choices = response.get("choices") or []
        if not choices:
            return "[LLM returned no choices]"
        first = choices[0]
        # Some providers use 'message' -> 'content', others may return a string
        msg = first.get("message") or {}
        return msg.get("content") or first.get("text") or "[LLM returned an unexpected format]"
    except Exception as e:
        return f"[LLM error: {e}]"
