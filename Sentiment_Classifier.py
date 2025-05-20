"""
Sentiment Classifier using Cohere's Command Model
This script demonstrates how to use Cohere's Command model to classify the sentiment of sentences.

Requires:
- cohere Python package
- dotenv package for environment variables

Usage:
1. Install the required packages:
   pip install cohere dotenv
2. Create a .env file in the same directory with your Cohere API key:
   COHERE_API_KEY=your_api_key_here
3. Run the script:
   python Sentiment_Classifier.py

Todo:
- Add more examples to improve accuracy
- Accept input from text files or user input

Author: Adam Richards
Created: May 2025
"""

import cohere
import os
from dotenv import load_dotenv

# Load API key
# Ensure you have a .env file with your API key
load_dotenv()
api_key = os.getenv("COHERE_API_KEY")
if not api_key:
    raise ValueError("Missing API key. Check your .env file.") # Ensure you have a .env file with your API key

co = cohere.Client(api_key)

# Example prompts for sentiment classification
# The prompt includes examples of sentences with their corresponding sentiments.
# The model will use these examples to understand how to classify the sentiment of new sentences.
example_prompts = """Determine the sentiment (Positive, Negative, or Neutral) of the following sentences.

Sentence: I love this product, it's fantastic!
Sentiment: Positive

Sentence: This is the worst thing I've ever bought.
Sentiment: Negative

Sentence: It’s okay, not great but not bad.
Sentiment: Neutral

Sentence: {}
Sentiment:"""

# Sentences to classify
inputs = [
    "I'm so excited about this new feature!",
    "Nothing about this impressed me.",
    "It's fine I guess, just not amazing.",
    "This makes me really angry.",
    "Wow, I’m speechless in the best way."
]

# Classify the sentiment
print("Sentiment Classification Results:\n")
for sentence in inputs:
    prompt = example_prompts.format(sentence)
    response = co.generate(
        model='command-xlarge',
        prompt=prompt,
        max_tokens=10,
        temperature=0.3,
        stop_sequences=["\n"]
    )
    sentiment = response.generations[0].text.strip()
    print(f"\"{sentence}\" → {sentiment}")
