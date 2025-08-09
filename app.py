import streamlit as st
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import numpy as np

# Load model and tokenizer
model_path = "./final_model/results/final_model"
model = DistilBertForSequenceClassification.from_pretrained("bipolar27/fake-news-app")
tokenizer = DistilBertTokenizerFast.from_pretrained("bipolar27/fake-news-app")

model.eval()

# Prediction function
def classify_news(text):
    if not text.strip():
        return "⚠️ Please enter some text.", 0.0

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]

    fake_prob = probs[0].item()
    real_prob = probs[1].item()

    label = "🟢 Real News" if real_prob > fake_prob else "🔴 Fake News"
    confidence = max(fake_prob * 100.0, real_prob * 100.0)

    return label, confidence

# Page config
st.set_page_config(
    page_title="Fake News Detector",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
.big-font {
    font-size:26px !important;
}
</style>
""", unsafe_allow_html=True)

st.title("📰 AI-Powered Fake News Detector")
st.markdown("Paste a news article or headline to classify it as **Real** or **Fake** ")

# Input
text_input = st.text_area("📝 Input News Article", height=200)

# Buttons
col1, col2 = st.columns(2)

with col1:
    analyze = st.button(" Analyze")
with col2:
    use_example = st.button(" Use Example")

# Example article
example_article = """
NASA has confirmed that aliens have been found living on Mars and have initiated communication with Earth, according to multiple sources.
"""

if use_example:
    text_input = example_article
    st.experimental_rerun()

# Run prediction
if analyze and text_input.strip():
    label, confidence = classify_news(text_input)
    st.markdown(f"### {label}")
    st.progress(min(int(confidence), 100))  # safely cap at 100
    st.markdown(f"**Confidence:** `{confidence:.2f}%`")
elif analyze:
    st.warning("⚠️ Please enter some text.")

st.markdown("---")
st.markdown("Made by Sanchali Das, Ishani Ghosh and Souradip Chakraborty")
