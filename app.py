from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

# Function to check if a URL is indexed by Google
def is_url_indexed(url):
    search_url = f"https://www.google.com/search?q=site:{url}"
    response = requests.get(search_url)

    if response.status_code != 200:
        print("Failed to fetch data from Google.")
        return False
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    if 'did not match any documents' in soup.text:
        return False
    else:
        return True

# Preprocessing function to extract features from URL
def preprocess_url(url):
    length_url = len(url)
    length_hostname = len(url.split('/')[2]) if len(url.split('/')) > 2 else 0
    ip = 1 if url.startswith('http://') or url.startswith('https://') else 0
    nb_dots = url.count('.')
    nb_slash = url.count('/')
    http_in_path = 1 if 'http' in url else 0
    google_index = is_url_indexed(url)
    
    # Placeholder for page rank
    page_rank = 0

    return np.array([length_url, length_hostname, ip, nb_dots, nb_slash, http_in_path, google_index, page_rank]).reshape(1, -1)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        url = request.form['url']
        features = preprocess_url(url)
        prediction = model.predict(features)
        result = 'Legitimate' if prediction[0] == 0 else 'Phishing'
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
