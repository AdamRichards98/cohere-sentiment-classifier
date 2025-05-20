import cohere
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
api_key = os.getenv("COHERE_API_KEY")
if not api_key:
    raise ValueError("Missing API key. Check your .env file.")

co = cohere.Client(api_key)

# Few-shot examples in the prompt
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
