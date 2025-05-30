#  AI vs Human Text Detector

This project is a deep learning-based binary text classifier that distinguishes between **AI-generated** and **Human-written** text using an Artificial Neural Network (ANN).
The model uses structured features and outputs both the **predicted label** and **confidence score** for each input.

----------------------------------------------------------------------------------------------------------------------------------------------------------------

##  Project Overview

-  **Task**: Classify input text as either **AI-written** or **Human-written**
-  **Model Used**: Artificial Neural Network (ANN)
-  **Input Features**:
  - TF-IDF vectorized text
  - Grammar error count
  - Repetition score
  - Personal pronoun count
  - Flesch reading ease score
- **Output**:
  - Binary label (`0 = Human`, `1 = AI`)
  - Confidence score (sigmoid output)

-----------------------------------------------------------------------------------------------------------------------------------------------------------------

## 📂 File Structure

```bash
📁 AI-vs-Human-Classifier/
│
├── DL_aivshuman.ipynb          # Jupyter notebook for ANN training and evaluation
├── app.py                      # Feature engineering and grammar checks ,Streamlit app for deployment
├── README.md                   # Project documentation
└── combined_datasets/            # Human and AI-generated text datasets
