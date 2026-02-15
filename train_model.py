import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Download NLTK data
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    return " ".join(tokens)

def train():
    print("Loading dataset...")
    try:
        df = pd.read_csv('dataset/archive/BBCNews.csv')
    except FileNotFoundError:
        print("Error: dataset/archive/BBCNews.csv not found.")
        return

    # Filter and Map categories
    categories = {
        'sport': 'Sports',
        'politics': 'Politics',
        'tech': 'Technology',
        'business': 'Business',
        'entertainment': 'Entertainment'
    }

    # Function to extract category from tags
    def get_category(tags):
        tags = str(tags).lower()
        for key, val in categories.items():
            if key in tags:
                return val
        return None

    df['category'] = df['tags'].apply(get_category)
    
    # Drop rows with no matching category
    df = df.dropna(subset=['category'])
    
    print(f"Dataset size after filtering: {len(df)}")
    print("Category distribution:\n", df['category'].value_counts())

    print("Preprocessing text...")
    # 'descr' seems to contain the text content based on previous inspection
    if 'descr' not in df.columns:
         # Fallback if column names are different, check previous inspect output...
         # Columns were: Index(['Unnamed: 0', 'descr', 'tags'], dtype='object')
         print("Error: 'descr' column not found.")
         return
         
    df['clean_text'] = df['descr'].apply(preprocess_text)

    X = df['clean_text']
    y = df['category']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training model...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('clf', MultinomialNB())
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    print("Saving model pipeline...")
    joblib.dump(pipeline, 'model_pipeline.pkl')
    print("Model saved as model_pipeline.pkl")

if __name__ == "__main__":
    train()
