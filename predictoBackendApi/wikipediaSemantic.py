import numpy as np
import tensorflow as tf
import wikipedia
from sklearn.metrics.pairwise import cosine_similarity

# Path to the downloaded model directory on your local computer
model_path = r"C:\Users\Administrator\Documents\uninversal"

# Load the model using TensorFlow
embed = tf.saved_model.load(model_path)

# User query processing
user_query = "what is cryptocurrency"

# Search Wikipedia for the user query
search_results = wikipedia.search(user_query)
print(search_results)
if search_results:
    # Choose the first search result
    selected_page = search_results[0]
    print(selected_page)
    # Get the content of the Wikipedia page
    page_content = wikipedia.page(selected_page).content
    # Split content into sentences
    wiki_sentences = page_content.split(". ")
    # Preprocess sentences by removing newline characters
    wiki_sentences = [sentence.strip() for sentence in wiki_sentences if sentence.strip()]
else:
    print("No relevant Wikipedia page found for the query.")

if wiki_sentences:
    # Encode Wikipedia sentences into embeddings
    wiki_embeddings = embed(wiki_sentences)

    # Calculate cosine similarity between query embedding and Wikipedia embeddings
    similarity_scores = cosine_similarity([embed([user_query])[0]], wiki_embeddings)[0]

    # Combine the sentences and similarity scores
    ranked_sentences = sorted(zip(wiki_sentences, similarity_scores), key=lambda x: x[1], reverse=True)

    # Display the first 5 ranked sentences
    for sentence, score in ranked_sentences[:5]:
        print(f"Similarity Score: {score:.4f}")
        print(f"Sentence: {sentence}")
        print("=" * 50)
else:
    print("No relevant Wikipedia sentences found for the query.")
