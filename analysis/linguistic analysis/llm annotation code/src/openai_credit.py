# config.py
import os

# Retrieve the OPENAI_API_KEY from environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if OPENAI_API_KEY is None:
    raise ValueError("The OPENAI_API_KEY environment variable is not set.")