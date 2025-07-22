import torch
import torch.nn as nn

# Super simple example first
vocab = ["hello", "hi", "dog", "cat", "pizza", "burger"]

# Without embeddings (one-hot encoding)
one_hot_example = {
    "hello": [1, 0, 0, 0, 0, 0],
    "hi":    [0, 1, 0, 0, 0, 0],
    "dog":   [0, 0, 1, 0, 0, 0],
    "cat":   [0, 0, 0, 1, 0, 0],
}

# With embeddings (learned meaningful vectors)
# These numbers would be learned during training
embedding_example = {
    "hello": [0.2, 0.1],     # 2 numbers represent "hello"
    "hi":    [0.3, 0.2],     # similar to "hello" (both are greetings)
    "dog":   [0.6, 0.8],     # similar to "cat" (both are animals)
    "cat":   [0.5, 0.7],     # similar to "dog"
    "pizza": [0.9, 0.1],     # similar to "burger" (both are food)
    "burger":[0.8, 0.2],     # similar to "pizza"
}

# Let's implement this:
class SimpleEmbeddingExample:
    def __init__(self):
        # Create vocabulary
        self.word_to_id = {
            "<PAD>": 0,
            "hello": 1,
            "hi": 2,
            "dog": 3,
            "cat": 4,
            "pizza": 5,
            "burger": 6
        }
        
        # Create embedding layer
        vocab_size = len(self.word_to_id)
        embedding_dim = 2  # we'll use 2 dimensions for easy visualization
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
    
    def get_embedding(self, word):
        # Convert word to ID
        word_id = self.word_to_id.get(word, 0)
        # Convert to tensor
        word_tensor = torch.tensor([word_id])
        # Get embedding
        return self.embedding(word_tensor)

# Let's visualize it!
def plot_embeddings(model):
    import matplotlib.pyplot as plt
    
    # Get embeddings for all words
    words = list(model.word_to_id.keys())
    embeddings = [model.get_embedding(word).detach().numpy()[0] for word in words]
    
    # Split into x and y coordinates
    x_coords = [e[0] for e in embeddings]
    y_coords = [e[1] for e in embeddings]
    
    # Plot
    plt.figure(figsize=(10, 10))
    plt.scatter(x_coords, y_coords)
    
    # Add word labels
    for i, word in enumerate(words):
        plt.annotate(word, (x_coords[i], y_coords[i]))
    
    plt.title("Word Embeddings Visualization")
    plt.show()

# Let's try it out!
if __name__ == "__main__":
    model = SimpleEmbeddingExample()
    
    # Look at some embeddings
    print("Embedding for 'hello':", model.get_embedding("hello").detach().numpy())
    print("Embedding for 'hi':", model.get_embedding("hi").detach().numpy())
    print("Embedding for 'dog':", model.get_embedding("dog").detach().numpy())
    
    # Visualize
    try:
        plot_embeddings(model)
    except ImportError:
        print("Matplotlib not installed")

    # Let's see how embeddings work in a sequence
    sentence = "hello dog"
    
    # Convert to IDs
    token_ids = [model.word_to_id.get(word, 0) for word in sentence.split()]
    print("\nToken IDs:", token_ids)
    
    # Convert to tensor
    token_tensor = torch.tensor(token_ids)
    
    # Get embeddings for whole sequence
    sequence_embeddings = model.embedding(token_tensor)
    print("\nSequence embedding shape:", sequence_embeddings.shape)
    print("Embeddings for sequence:\n", sequence_embeddings.detach().numpy())