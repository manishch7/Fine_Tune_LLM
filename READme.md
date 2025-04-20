# Financial Sentiment Analysis

This project fine-tunes a language model to analyze sentiment in financial tweets.

## Sentiment Classes
- Bearish (0): Negative financial sentiment
- Positive (1): Positive financial sentiment  
- Neutral (2): Neutral financial content

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Explore dataset: `python dataset_exploration.py`  
3. Train model: `python train_model.py`
4. Evaluate model: `python evaluate_model.py`
5. Run inference: `python inference.py "Markets rally today"`