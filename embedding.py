import torch
import torch.nn as nn
import matplotlib.pyplot as plt

'''
In this code, the embeddings are just randomly initialized and have no semantic meaning yet. 
'''
class TextEmbedding:  
    def __init__(self, vocab_size, embedding_dim):
        self.weight = torch.randn(vocab_size, embedding_dim)
        
        # Vocabulary mapping
        self.word_to_id = {
            "<PAD>": 0,
            "hello": 1,
            "hi": 2,
            "dog": 3,
            "cat": 4,
            "pizza": 5,
            "burger": 6
        }
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
    
    def get_embedding(self, word):
        # Convert word to ID
        word_id = self.word_to_id.get(word, 0)
        # Get embedding by indexing into weight matrix
        return self.weight[word_id]
    
    def embed_sequence(self, sentence):
        # Convert sentence to token IDs
        token_ids = [self.word_to_id.get(word, 0) for word in sentence.split()]
        # Convert to tensor
        token_tensor = torch.tensor(token_ids)
        # Get embeddings by indexing into weight matrix
        return self.weight[token_tensor]
    
    def visualize_embeddings(self):
        if self.weight.shape[1] != 2:
            print("Warning: Can only visualize 2D embeddings")
            return
            
        # Get embeddings for all words
        words = list(self.word_to_id.keys())
        embeddings = [self.weight[self.word_to_id[word]] for word in words]
        
        # Split into x and y coordinates
        x_coords = [e[0].item() for e in embeddings]
        y_coords = [e[1].item() for e in embeddings]
        
        # Plot
        plt.figure(figsize=(10, 10))
        plt.scatter(x_coords, y_coords)
        
        # Add word labels
        for i, word in enumerate(words):
            plt.annotate(word, (x_coords[i], y_coords[i]))
        
        plt.title("Word Embeddings Visualization")
        plt.show()

    def forward(self, x):  
        return self.weight[x]

if __name__ == "__main__":
    # Create model with 2D embeddings for visualization

    '''# With embeddings (learned meaningful vectors) 
    # These numbers would be learned during training
    # 1) the examples below are only 2 dimensional embeddings - In the original "Attention is All You Need" paper, 
    # the model's base configuration uses embeddings with dimension size 512 - try to vizualize a 512 dimensional space that represents a super position
    # of a word
    # 2) these embedding values are hard coded for demo purposes

    embedding_example = {
        "hello": [0.2, 0.1],     # 2 numbers represent "hello"
        "hi":    [0.3, 0.2],     # similar to "hello" (both are greetings)
        "dog":   [0.6, 0.8],     # similar to "cat" (both are animals)
        "cat":   [0.5, 0.7],     # similar to "dog"
        "pizza": [0.9, 0.1],     # similar to "burger" (both are food)
        "burger":[0.8, 0.2],     # similar to "pizza"
    }'''

    word_embedder = TextEmbedding(vocab_size=7, embedding_dim=2)
    
    # Look at some individual embeddings
    print("Embedding for 'hello':", word_embedder.get_embedding("hello"))
    print("Embedding for 'dog':", word_embedder.get_embedding("dog"))
    
    # Embed a sequence
    sentence = "hello dog"
    sequence_embeddings = word_embedder.embed_sequence(sentence)
    print("\nSequence embedding shape:", sequence_embeddings.shape)
    print("Embeddings for sequence:\n", sequence_embeddings)
    
    # Visualize embeddings
    word_embedder.visualize_embeddings()
    
    # Save model weights
    torch.save(word_embedder.state_dict(), "text_embedding_model.pth")
    
    # Load model weights
    loaded_model = TextEmbedding(vocab_size=7, embedding_dim=2)
    loaded_model.load_state_dict(torch.load("text_embedding_model.pth"))
    print("\nLoaded model parameters:")
    print(loaded_model.weight)