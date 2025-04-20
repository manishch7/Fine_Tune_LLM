"""
Simple fine-tuning script for financial sentiment analysis
"""
import os
import time
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer
)
from sklearn.metrics import accuracy_score, f1_score

# Create timestamp for versioned outputs
timestamp = time.strftime("%Y%m%d_%H%M%S")
MODEL_DIR = f"./financial_sentiment_model_{timestamp}"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("figures", exist_ok=True)

# Setup logging to a file
log_file = open(f"{MODEL_DIR}/training_log.txt", "w")
def log(message):
    print(message)
    log_file.write(message + "\n")

# Determine device (CPU, CUDA, or MPS for Apple Silicon)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    log("Using MPS (Apple Silicon) device")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    log("Using CUDA device")
else:
    device = torch.device("cpu")
    log("Using CPU device")

# Base model to fine-tune 
BASE_MODEL = "distilbert-base-uncased"
log(f"Starting fine-tuning of {BASE_MODEL} for financial sentiment analysis")

# Load dataset
log("\nStep 1: Loading dataset...")
dataset = load_dataset("zeroshot/twitter-financial-news-sentiment")

# Create train/validation split
train_val = dataset["train"].train_test_split(test_size=0.1, seed=42)
train_data = train_val["train"]
val_data = train_val["test"]
log(f"Training data: {len(train_data)} examples")
log(f"Validation data: {len(val_data)} examples")

# Load tokenizer
log("\nStep 2: Setting up tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Prepare data for training
def preprocess_data(examples):
    return tokenizer(
        examples["text"], 
        truncation=True, 
        padding="max_length", 
        max_length=128
    )

log("\nStep 3: Preprocessing data...")
train_encoded = train_data.map(preprocess_data, batched=True)
val_encoded = val_data.map(preprocess_data, batched=True)

# Compute metrics function
def compute_metrics(eval_pred):
    predictions = eval_pred.predictions.argmax(-1)
    labels = eval_pred.label_ids
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    return {"accuracy": accuracy, "f1": f1}

# Initialize model
log("\nStep 4: Initializing model...")
model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL, 
    num_labels=3  # Bearish, Bullish, Neutral
)

# Move model to device
model.to(device)

# Set up training arguments - simplified parameters
training_args = TrainingArguments(
    output_dir=MODEL_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=1,
    logging_steps=100,
    logging_dir=f"{MODEL_DIR}/logs",
    # For Apple Silicon
    no_cuda=True if device.type == "mps" else False,
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_encoded,
    eval_dataset=val_encoded,
    compute_metrics=compute_metrics,
)

# Train model
log("\nStep 5: Training model...")
train_results = trainer.train()

# Save evaluation results 
log("\nStep 6: Evaluating model...")
eval_results = trainer.evaluate()
log(f"Final evaluation results: {eval_results}")

# Save model
log(f"\nStep 7: Saving model to {MODEL_DIR}...")
model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)

log("\nTraining complete! Your model is ready for evaluation and inference.")
log_file.close()