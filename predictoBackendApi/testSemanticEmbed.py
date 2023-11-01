from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import tensorflow as tf

#load embeddings of the dataset
loaded_embeddings  = np.load(r"C:\Users\Administrator\Desktop\Icog Project\predictochain\predictoBackend\sentence_embeddings.model.npy")

# Path to the downloaded model directory on your local computer
model_path = r"C:\Users\Administrator\Documents\uninversal"

# Load the model using TensorFlow
embed = tf.saved_model.load(model_path)

# Path to your text file containing AI-related sentences
file_path = r"C:\Users\Administrator\Documents\semanticDataset.txt"

# Read sentences from the text file
with open(file_path, "r", encoding="utf-8") as file:
    dataset = file.readlines()

# Preprocess sentences by removing newline characters
dataset = [sentence.strip() for sentence in dataset]

# User query processing
user_query = "what is artificial intelligence(AI)"

# Preprocess user query and encode it into an embedding
query_embedding = embed([user_query])[0]

# Calculate cosine similarity between query embedding and loaded embeddings
similarity_scores = cosine_similarity([query_embedding], loaded_embeddings)[0]

# Combine the sentences and similarity scores
ranked_sentences = sorted(zip(dataset, similarity_scores), key=lambda x: x[1], reverse=True)

# Display the first 5 ranked sentences (as shown in previous responses)
for sentence, score in ranked_sentences[:5]:
    print(f"Similarity Score: {score:.4f}")
    print(f"Sentence: {sentence}")
    print("=" * 50)
