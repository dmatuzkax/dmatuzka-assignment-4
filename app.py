from flask import Flask, render_template, request, jsonify
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.datasets import fetch_20newsgroups

nltk.download('stopwords')

app = Flask(__name__)

# TODO: Fetch dataset, initialize vectorizer and LSA here
newsgroups = fetch_20newsgroups(subset='all')
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.95, max_features=1000, ngram_range=(1, 2))
term_doc_matrix = vectorizer.fit_transform(newsgroups.data).toarray()

U, S, VT = np.linalg.svd(term_doc_matrix, full_matrices=False)

k = 100

U_k = U[:, :k]
S_k = np.diag(S[:k])
VT_k = VT[:k, :]

X_reduced = np.dot(U_k, S_k)


def search_engine(query):
    """
    Function to search for top 5 similar documents given a query
    Input: query (str)
    Output: documents (list), similarities (list), indices (list)
    """
    # TODO: Implement search engine here
    # return documents, similarities, indices
    # Vectorize the query
    query_vec = vectorizer.transform([query]).toarray()

    # Reduce the query using the same SVD transformation
    query_reduced = np.dot(query_vec, VT_k.T)

    # Compute cosine similarity between query and document vectors
    similarities = cosine_similarity(query_reduced, X_reduced)

    # Get top 5 most similar documents
    indices = np.argsort(similarities[0])[::-1][:5]
    documents = [newsgroups.data[i] for i in indices]
    
    return documents, similarities[0][indices], indices


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    documents, similarities, indices = search_engine(query)
    return jsonify({'documents': documents, 'similarities': similarities.tolist(), 'indices': indices.tolist()}) 

if __name__ == '__main__':
    app.run(debug=True, port=3000)
