from flask import Flask, render_template, request, jsonify
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np

app = Flask(__name__)

# Load dataset
newsgroups = fetch_20newsgroups(subset='all')
documents = newsgroups.data

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Perform LSA
svd = TruncatedSVD(n_components=100)  # You can adjust the number of components
lsa_matrix = svd.fit_transform(tfidf_matrix)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    query_vec = vectorizer.transform([query])
    query_lsa = svd.transform(query_vec)
    cosine_similarities = np.dot(lsa_matrix, query_lsa.T).flatten()
    top_indices = cosine_similarities.argsort()[-5:][::-1]

    results = [{'document': documents[i], 'similarity': cosine_similarities[i]} for i in top_indices]
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, port=3000)
