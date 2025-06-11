# config.py
import os

# Retrieve the ANTHROPIC_API_KEY from environment variables
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

if ANTHROPIC_API_KEY is None:
    raise ValueError("The ANTHROPIC_API_KEY environment variable is not set.")