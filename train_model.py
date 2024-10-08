import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
import joblib

# Load the dataset
data = pd.read_csv('cleaned_dataset.csv')

# Feature extraction function
def preprocess_url(url):
    length_url = len(url)
    length_hostname = len(url.split('/')[2]) if len(url.split('/')) > 2 else 0
    ip = 1 if url.startswith('http://') or url.startswith('https://') else 0
    nb_dots = url.count('.')
    nb_slash = url.count('/')
    http_in_path = 1 if 'http' in url else 0
    return np.array([length_url, length_hostname, ip, nb_dots, nb_slash, http_in_path])

# Preprocess URLs
X = data['url'].apply(preprocess_url).tolist()
y = data['status'].values

# Convert to NumPy arrays
X = np.array(X.tolist())

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'model.pkl')

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f'Model Accuracy: {accuracy:.2f}')
