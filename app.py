"""
Streamlit app for financial sentiment analysis
"""
import app as st
import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(
    page_title="Financial Sentiment Analyzer",
    page_icon="💰",
    layout="centered"
)

# App title and description
st.title("Financial Sentiment Analyzer")
st.markdown("""
This app predicts the sentiment of financial texts as:
- 📉 **Bearish** (negative/pessimistic)
- 📈 **Bullish** (positive/optimistic)
- 🔄 **Neutral**
""")

# Function to load model
@st.cache_resource
def load_model():
    # Find the most recent model directory
    model_dirs = [d for d in os.listdir('./') if d.startswith('financial_sentiment_model_')]
    if not model_dirs:
        st.error("Error: No model directory found. Please run train_model.py first.")
        st.stop()
        
    MODEL_DIR = sorted(model_dirs)[-1]  # Latest model
    st.info(f"Using model from: {MODEL_DIR}")
    
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    
    return model, tokenizer, MODEL_DIR

# Load model
try:
    model, tokenizer, model_dir = load_model()
    
    # Sentiment labels
    sentiment_labels = {0: "Bearish", 1: "Bullish", 2: "Neutral"}
    sentiment_icons = {0: "📉", 1: "📈", 2: "🔄"}
    
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
            "sentiment": sentiment_labels[predicted_class],
            "icon": sentiment_icons[predicted_class],
            "confidence": probabilities[predicted_class].item(),
            "probabilities": {
                sentiment_labels[i]: prob.item()
                for i, prob in enumerate(probabilities)
            }
        }
    
    # Example texts section
    with st.expander("Try with example texts"):
        examples = [
            "Tesla stock surges on earnings beat",
            "Markets decline as inflation concerns grow",
            "Federal Reserve maintains current interest rates"
        ]
        
        for i, example in enumerate(examples):
            if st.button(f"Example {i+1}", key=f"example_{i}"):
                st.session_state.text_input = example
    
    # User input section
    text_input = st.text_area(
        "Enter financial text to analyze:",
        value=st.session_state.get("text_input", ""),
        height=100,
        placeholder="Enter financial news or statements here...",
        key="text_input"
    )
    
    # Prediction button
    col1, col2 = st.columns([1, 3])
    with col1:
        predict_button = st.button("Analyze", type="primary")
    
    # Only show prediction if text is entered and button is clicked
    if text_input and predict_button:
        with st.spinner("Analyzing sentiment..."):
            # Get prediction
            result = predict_sentiment(text_input)
            
            # Display result
            st.markdown("### Analysis Result")
            
            # Format results in a nice UI
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown(f"# {result['icon']}")
            with col2:
                st.markdown(f"### {result['sentiment']}")
                st.progress(result['confidence'])
                st.text(f"Confidence: {result['confidence']:.2f}")
            
            # Show probability breakdown
            st.markdown("### Probability Distribution")
            for label, prob in result['probabilities'].items():
                st.markdown(f"**{label}**: {prob:.3f}")
                st.progress(prob)
    
    # About section
    with st.expander("About this model"):
        st.markdown(f"""
        **Model Details**
        - Base model: DistilBERT
        - Fine-tuned on: Twitter Financial News Sentiment dataset
        - Model directory: {model_dir}
        
        This model classifies financial texts into three sentiment categories:
        - **Bearish**: Negative sentiment, expecting prices to fall
        - **Bullish**: Positive sentiment, expecting prices to rise
        - **Neutral**: Neither positive nor negative
        """)

except Exception as e:
    st.error(f"Error loading model: {e}")
    st.markdown("""
    ## Troubleshooting
    1. Make sure you've run `train_model.py` first to create a model
    2. Ensure all required libraries are installed:
    ```
    pip install streamlit transformers torch
    ```
    """)