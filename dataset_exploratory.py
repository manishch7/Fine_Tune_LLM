"""
Simple exploration of Twitter financial sentiment dataset
"""
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset

# Create figures directory
import os
os.makedirs("figures", exist_ok=True)

# Load dataset
print("Loading Twitter financial sentiment dataset...")
dataset = load_dataset("zeroshot/twitter-financial-news-sentiment")

# Convert to dataframe for easier analysis
train_df = pd.DataFrame(dataset["train"])

# Define sentiment labels
sentiment_labels = {0: "Bearish", 1: "Bullish", 2: "Neutral"}

# Basic statistics
print(f"Dataset size: {len(train_df)} examples")
print(f"Average text length: {train_df['text'].str.len().mean():.1f} characters")

# Label distribution
label_counts = train_df['label'].value_counts().sort_index()
label_counts.index = [sentiment_labels[i] for i in label_counts.index]
print("\nSentiment Distribution:")
print(label_counts)
print(f"Percentages: {(label_counts / len(train_df) * 100).round(1)}")

# Visualize distribution
plt.figure(figsize=(8, 5))
label_counts.plot(kind='bar', color=['red', 'green', 'gray'])
plt.title('Financial Tweet Sentiment Distribution')
plt.ylabel('Number of Tweets')
plt.tight_layout()
plt.savefig("figures/sentiment_distribution.png")

# Print examples of each class
print("\nExample tweets:")
for label, name in sentiment_labels.items():
    examples = train_df[train_df['label'] == label]['text'].head(1).tolist()
    print(f"\n{name} example:")
    for ex in examples:
        print(f"- {ex}")

print("\nExploration complete!")