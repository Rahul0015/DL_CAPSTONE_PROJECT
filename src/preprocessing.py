import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.model_selection import train_test_split
import numpy as np

def load_and_clean_data(input_path):
    df = pd.read_csv(input_path)
    df['threat_level'] = df['casualties'].apply(lambda x: 'High' if x >= 5 else 'Low')
    return df

def encode_and_vectorize(df, output_dir):
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['threat_level'])

    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    X = tfidf.fit_transform(df['summary'].astype(str)).toarray()
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'tfidf_vectorizer.pkl'), 'wb') as f:
        pickle.dump(tfidf, f)
    with open(os.path.join(output_dir, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)

    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)

    print("✅ Preprocessing complete. Files saved.")
