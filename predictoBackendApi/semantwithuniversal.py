import tensorflow as tf
import numpy as np
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

# Generate embeddings for each sentence
sentence_embeddings = embed(dataset)

# Save sentence embeddings in a binary file
np.save("sentence_embeddings.model", sentence_embeddings)

