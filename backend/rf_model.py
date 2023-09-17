import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_random_forest(df: pd.DataFrame):
    df.dropna(subset=['text'], inplace=True)

    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X = tfidf_vectorizer.fit_transform(df['text'])

    X_train, X_test, y_train, y_test = train_test_split(X, df['label'], test_size=0.2, random_state=42)

    # Initialize and train the RandomForestClassifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    # Make predictions on the testing set
    y_pred = rf_classifier.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')

    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))