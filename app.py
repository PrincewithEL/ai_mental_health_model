import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify

# Load the dataset
data = pd.read_csv('Dataset.csv')

# Remove rows with NaN values in 'statement' column (or replace NaN with empty string)
data = data.dropna(subset=['statement'])  # Option 1: Remove NaN rows
# data['statement'] = data['statement'].fillna('')  # Option 2: Replace NaN with empty string

# Initialize vectorizer and vectorize contexts
vectorizer = TfidfVectorizer()
context_vectors = vectorizer.fit_transform(data['statement'])

# Create Flask app
app = Flask(__name__)

# Define API route
@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.json.get('user_input', '')
    if not user_input:
        return jsonify({'error': 'User input is missing'}), 400

    # Calculate similarity and fetch the best response
    user_vector = vectorizer.transform([user_input])
    similarity_scores = cosine_similarity(user_vector, context_vectors)
    best_match_index = similarity_scores.argmax()
    response = data.iloc[best_match_index]['status']
    return jsonify({'response': response})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
