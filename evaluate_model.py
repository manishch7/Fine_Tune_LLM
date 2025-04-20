"""
Simple evaluation of fine-tuned sentiment model
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# Check if model path is provided
if len(sys.argv) > 1:
    MODEL_DIR = sys.argv[1]
else:
    # Find the most recent model directory
    model_dirs = [d for d in os.listdir('./') if d.startswith('financial_sentiment_model_')]
    if not model_dirs:
        print("Error: No model directory found. Please run train_simple.py first.")
        sys.exit(1)
    MODEL_DIR = sorted(model_dirs)[-1]  # Latest model

print(f"Evaluating model from: {MODEL_DIR}")

# Create figures directory
os.makedirs("figures", exist_ok=True)

# Label mapping
sentiment_labels = {0: "Bearish", 1: "Bullish", 2: "Neutral"}

# Load dataset
print("\nLoading dataset...")
dataset = load_dataset("zeroshot/twitter-financial-news-sentiment")
train_val = dataset["train"].train_test_split(test_size=0.1, seed=42)
test_dataset = train_val["test"]
print(f"Evaluating on {len(test_dataset)} examples")

# Load model and tokenizer
print(f"\nLoading model from {MODEL_DIR}...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

# Prepare device (CPU, CUDA, or MPS)
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
model.to(device)
print(f"Using device: {device}")

# Get predictions
print("\nMaking predictions...")
texts = test_dataset["text"]
true_labels = test_dataset["label"]

# Process in smaller batches
all_predictions = []
batch_size = 32

for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i+batch_size]
    
    # Tokenize
    inputs = tokenizer(
        batch_texts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True
    ).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get predictions
    batch_predictions = torch.argmax(outputs.logits, dim=1).cpu().numpy()
    all_predictions.extend(batch_predictions)
    
    if i % (5 * batch_size) == 0:
        print(f"Processed {i}/{len(texts)} examples...")

predicted_labels = np.array(all_predictions)

# Generate classification report
print("\nClassification Report:")
report = classification_report(
    true_labels, 
    predicted_labels, 
    target_names=[sentiment_labels[i] for i in range(3)],
    digits=3
)
print(report)

# Save report to file
with open(f"{MODEL_DIR}/evaluation_report.txt", "w") as f:
    f.write(report)

# Generate confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)
cmd = ConfusionMatrixDisplay(
    confusion_matrix=cm, 
    display_labels=[sentiment_labels[i] for i in range(3)]
)

plt.figure(figsize=(8, 6))
cmd.plot(cmap="Blues", values_format="d")
plt.title("Financial Sentiment Confusion Matrix")
plt.tight_layout()
plt.savefig("figures/confusion_matrix.png")
print("Saved confusion matrix to figures/confusion_matrix.png")

# Show error examples
errors = []
for i in range(len(texts)):
    if predicted_labels[i] != true_labels[i]:
        errors.append({
            "text": texts[i],
            "true": sentiment_labels[true_labels[i]],
            "predicted": sentiment_labels[predicted_labels[i]]
        })

print(f"\nFound {len(errors)} incorrect predictions out of {len(texts)} examples")
print(f"Error rate: {len(errors)/len(texts):.1%}")

# Show a few examples of errors
print("\nError Examples:")
for i, error in enumerate(errors[:3]):  # Show first 3 errors
    print(f"\nExample {i+1}:")
    print(f"Text: {error['text']}")
    print(f"True sentiment: {error['true']}")
    print(f"Predicted sentiment: {error['predicted']}")

print("\nEvaluation complete!")