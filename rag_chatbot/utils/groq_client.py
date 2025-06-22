import os
from groq import Groq

def get_groq_client():
    key = os.getenv("GROQ_API_KEY")
    return Groq(api_key=key)
