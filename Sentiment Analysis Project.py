import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import shap

# Step 1: Simulate or Load Data
def load_data():
    data = {
        "feedback": [
            "I love this product! It's amazing.",
            "This is the worst service I've ever used.",
            "Not bad, but could be better.",
            "I am extremely satisfied with the support.",
            "Terrible experience, I want my money back.",
            "Good quality, I'm happy with my purchase.",
            "It's okay, not the best but it works.",
            "Horrible, just horrible. Will never use again.",
            "Fantastic! Exceeded my expectations.",
            "Very disappointing, I expected more."
        ],
        "sentiment": [1, 0, 1, 1, 0, 1, 1, 0, 1, 0]
    }
    return pd.DataFrame(data)

# Step 2: Text Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\b\d+(?:\.\d+)?\s+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Step 3: Feature Engineering
def create_features(data):
    vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
    X = vectorizer.fit_transform(data['feedback'].apply(preprocess_text))
    return X, vectorizer

# Step 4: Model Development
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Step 5: Model Evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 6: Interpretability
def interpret_model(model, X, vectorizer):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values[1], features=X, feature_names=vectorizer.get_feature_names_out())

# Step 7: Prediction Function
def predict_sentiment(model, vectorizer, new_feedback):
    new_feedback_processed = preprocess_text(new_feedback)
    features = vectorizer.transform([new_feedback_processed])
    prediction = model.predict(features)
    return "Positive" if prediction[0] == 1 else "Negative"

# Main function to run the project
def main():
    # Load and preprocess data
    data = load_data()
    X, vectorizer = create_features(data)
    y = data['sentiment']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train and evaluate model
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

    # Interpret model
    interpret_model(model, X_train, vectorizer)

    # Test prediction on new feedback
    test_feedback = "This is the best product ever!"
    print(f"Feedback: '{test_feedback}' -> Sentiment: {predict_sentiment(model, vectorizer, test_feedback)}")

if __name__ == "__main__":
    main()
