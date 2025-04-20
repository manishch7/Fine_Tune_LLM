"""
Simple prediction tool for financial sentiment analysis
"""
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Find the most recent model directory
if len(sys.argv) > 1:
    MODEL_DIR = sys.argv[1]
else:
    model_dirs = [d for d in os.listdir('./') if d.startswith('financial_sentiment_model_')]
    if not model_dirs:
        print("Error: No model directory found. Please run train_simple.py first.")
        sys.exit(1)
    MODEL_DIR = sorted(model_dirs)[-1]  # Latest model

print(f"Loading sentiment model from: {MODEL_DIR}")

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

# Sentiment labels
sentiment_labels = {0: "Bearish", 1: "Bullish", 2: "Neutral"}

# Function to predict sentiment
def predict_sentiment(text):
    # Tokenize
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        padding=True, 
        truncation=True
    )
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get predicted class and confidence
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
    predicted_class = torch.argmax(probabilities).item()
    
    return {
        "text": text,
        "sentiment": sentiment_labels[predicted_class],
        "confidence": probabilities[predicted_class].item(),
        "probabilities": {
            sentiment_labels[i]: f"{prob.item():.3f}" 
            for i, prob in enumerate(probabilities)
        }
    }

# Example texts
examples = [
    "Tesla stock surges on earnings beat",
    "Markets decline as inflation concerns grow",
    "Federal Reserve maintains current interest rates"
]

# Print header
print("\n--------- FINANCIAL SENTIMENT PREDICTOR ---------")
print("This tool predicts the sentiment of financial texts as:")
print("- Bearish (negative/pessimistic)")
print("- Bullish (positive/optimistic)")
print("- Neutral")
print("---------------------------------------------------")

# Run predictions on examples
print("\nExample predictions:")
for example in examples:
    result = predict_sentiment(example)
    print(f"\nText: {result['text']}")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Confidence: {result['confidence']:.3f}")

# Interactive mode
print("\nEnter your own financial text (or type 'exit' to quit):")
while True:
    user_text = input("> ")
    if user_text.lower() in ['exit', 'quit']:
        break
    
    if user_text.strip():
        result = predict_sentiment(user_text)
        print(f"Sentiment: {result['sentiment']} (Confidence: {result['confidence']:.3f})")
        print(f"All probabilities: {result['probabilities']}")

print("\nThanks for using the Financial Sentiment Predictor!")