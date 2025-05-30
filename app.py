# app.py

import streamlit as st
import numpy as np
import joblib
from PIL import Image
from textblob import TextBlob
import textstat
import docx
import PyPDF2
import requests
from tensorflow.keras.models import load_model

# === Load model and tokenizer ===
model = load_model("AIvsHUMAN.h5")
tokenizer = joblib.load("tfidf_tokenizer.pkl")

logical_connectors = [
    "however", "therefore", "moreover", "furthermore", "consequently",
    "nevertheless", "thus", "although", "because", "since",
    "in addition", "on the other hand", "as a result", "in contrast",
    "hence", "similarly", "for example", "for instance", "in conclusion", "despite"
]

# === Helper functions ===
def count_grammar_errors(text):
    blob = TextBlob(text)
    corrected = str(blob.correct())
    return sum(1 for a, b in zip(text, corrected) if a != b)

def repeated_word_ratio(text):
    words = text.lower().split()
    return sum(words.count(w) > 2 for w in set(words)) / len(words) if words else 0

def unique_word_ratio(text):
    words = text.lower().split()
    return len(set(words)) / len(words) if words else 0

def logical_connector_count(text):
    return sum(text.lower().count(conn) for conn in logical_connectors)

def extract_features(text):
    return {
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "grammar_error_count": count_grammar_errors(text),
        "repetition_score": sum(text.lower().split().count(word) > 3 for word in set(text.lower().split())),
        "personal_pronoun_count": sum(text.lower().split().count(p) for p in ['i', 'me', 'my', 'we', 'us', 'our']),
        "spelling_error_count": count_grammar_errors(text),
        "repeated_word_ratio": repeated_word_ratio(text),
        "unique_word_ratio": unique_word_ratio(text),
        "logical_connector_count": logical_connector_count(text)
    }

def predict_origin(text):
    vector = tokenizer.transform([text]).toarray()
    pred = model.predict(vector)[0][0]
    return (1 if pred >= 0.5 else 0), float(pred)

def generate_llm_explanation(text, origin, features, context):
    repetition_score = int(min(features['repeated_word_ratio'] * 200, 100))
    error_score = int(100 - min(features['grammar_error_count'] * 10, 100))
    creativity_score = int(min(features['unique_word_ratio'] * 100, 100))

    prompt = f"""
You are a smart AI classifier assistant. Analyze the following text and generate a natural-language explanation using the structure provided. The explanation should be dynamic and based on features like repetition, grammar errors, creativity, logical connectors, and personal pronouns. Avoid rigid rules or static suggestions.

### Context:
{context}

### Text:
{text}

### Extracted Features:
- Repetition Score: {repetition_score}/100
- Error Score: {error_score}/100
- Creativity Score: {creativity_score}/100
- Logical Connector Count: {features['logical_connector_count']}
- Personal Pronoun Count: {features['personal_pronoun_count']}
- Flesch Reading Ease: {features['flesch_reading_ease']}

### Classification:
- Predicted Origin: {"AI-generated" if origin else "Human-written"}
- AI Confidence: {features['confidence'] * 100:.0f}%
- Human Confidence: {(1 - features['confidence']) * 100:.0f}%

### Output format:
**Output (Text Origin Classification)**:
- Classification Result:
  - Origin: <Predicted Origin>, <Confidence>% confidence.
  - Rationale: [Explain based on features. Mention patterns like logical connectors, polished text, or sentence structure.]

- Recommendation: [Give one natural-sounding action or advice.]

**Output (Feature Analysis)**:
- Feature Report:
  - Repetition: <score>/100 (<AI or human>) – "[Explain why this score matters]"
  - Errors: <score>/100 (<AI or human>) – "[Natural explanation]"
  - Creativity: <score>/100 (<AI or human>) – "[Insight]"
  - Example: "[Point to a sample phrase or sentence if needed]"
- Suggestions:
  - [Dynamic suggestion 1]
  - [Dynamic suggestion 2]

Respond only with the above output structure.
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3", "prompt": prompt, "stream": False}
    )
    return response.json()["response"]

def extract_text(file):
    if file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    elif file.name.endswith(".docx"):
        doc = docx.Document(file)
        return "\n".join(p.text for p in doc.paragraphs)
    elif file.name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return ""

# === Streamlit UI ===
st.set_page_config(page_title="AI vs Human Detector", layout="centered")
st.title("🧠 AI vs Human Text Detector")

try:
    image = Image.open("20-July-web-blog.png")
    st.image(image, use_container_width=True)
except:
    pass

uploaded_file = st.file_uploader("📄 Upload (.txt, .docx, .pdf)", type=["txt", "docx", "pdf"])
context = st.text_input("📝 Context (e.g., Academic Essay, Blog Post)")

if uploaded_file:
    text_input = extract_text(uploaded_file)
else:
    text_input = st.text_area("✍️ Or paste your text here", height=250)

if st.button("🔍 Analyze"):
    if not text_input.strip():
        st.error("Text is required.")
    elif not context.strip():
        st.error("Context is required.")
    else:
        with st.spinner("Analyzing..."):
            features = extract_features(text_input)
            origin, conf = predict_origin(text_input)
            features["confidence"] = conf
            explanation = generate_llm_explanation(text_input, origin, features, context)
        st.markdown("### 🔎 Analysis Result")
        st.markdown(explanation)
