import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class SimpleTokenizer:
    def __init__(self):
        self.word_to_id = {"<PAD>": 0, "<UNK>": 1}
        self.id_to_word = {0: "<PAD>", 1: "<UNK>"}
        
    def tokenize(self, text):
        words = text.lower().split()
        current_id = len(self.word_to_id)
        
        tokens = []
        for word in words:
            if word not in self.word_to_id:
                self.word_to_id[word] = current_id
                self.id_to_word[current_id] = word
                current_id += 1
            tokens.append(self.word_to_id[word])
        return tokens
    
    def decode(self, token_ids):
        return [self.id_to_word.get(id, "<UNK>") for id in token_ids]
    
    def vocab_size(self):
        return len(self.word_to_id)

class SimpleTextEmbedding:
    def __init__(self, embedding_dim=2):
        self.tokenizer = SimpleTokenizer()
        self.embedding_dim = embedding_dim
        self.weight = None  # Will be initialized when we know vocab size
        
    def _ensure_embeddings(self):
        """Initialize or resize embeddings based on current vocabulary size"""
        vocab_size = self.tokenizer.vocab_size()
        
        if self.weight is None:
            # First time initialization
            self.weight = torch.randn(vocab_size, self.embedding_dim)
        elif self.weight.shape[0] < vocab_size:
            # Need to add embeddings for new words
            old_size = self.weight.shape[0]
            new_embeddings = torch.randn(vocab_size - old_size, self.embedding_dim)
            self.weight = torch.cat([self.weight, new_embeddings], dim=0)
    
    def get_embedding(self, word):
        """Get embedding for a single word"""
        # Tokenize the word (this will add it to vocab if new)
        token_ids = self.tokenizer.tokenize(word)
        self._ensure_embeddings()
        
        # Return embedding for the word
        return self.weight[token_ids[0]]
    
    def embed_sequence(self, text):
        """Convert text to embeddings"""
        # Tokenize the text
        token_ids = self.tokenizer.tokenize(text)
        self._ensure_embeddings()
        
        # Convert to tensor and get embeddings
        token_tensor = torch.tensor(token_ids)
        return self.weight[token_tensor]
    
    def visualize_embeddings(self):
        """Visualize embeddings (only works for 2D embeddings)"""
        if self.weight is None:
            print("No embeddings to visualize yet!")
            return
            
        if self.embedding_dim != 2:
            print(f"Warning: Can only visualize 2D embeddings, but embedding_dim is {self.embedding_dim}")
            return
        
        # Get all words and their embeddings
        words = list(self.tokenizer.word_to_id.keys())
        embeddings = [self.weight[self.tokenizer.word_to_id[word]] for word in words]
        
        # Split into x and y coordinates
        x_coords = [e[0].item() for e in embeddings]
        y_coords = [e[1].item() for e in embeddings]
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.scatter(x_coords, y_coords, s=100, alpha=0.7)
        
        # Add word labels
        for i, word in enumerate(words):
            plt.annotate(word, (x_coords[i], y_coords[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        plt.title("Word Embeddings Visualization")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.grid(True, alpha=0.3)
        plt.show()

if __name__ == "__main__":
    # Create embedding model with 2D embeddings for visualization
    embedder = SimpleTextEmbedding(embedding_dim=2)
    
    # Test with some text
    text1 = "hello world"
    text2 = "hello dog cat pizza"
    text3 = "the quick brown fox jumps"
    
    print("=== Testing Simple Text Embeddings ===")
    
    # Process first text
    print(f"\nText 1: '{text1}'")
    embeddings1 = embedder.embed_sequence(text1)
    print(f"Token IDs: {embedder.tokenizer.tokenize(text1)}")
    print(f"Embeddings shape: {embeddings1.shape}")
    print(f"Vocabulary size: {embedder.tokenizer.vocab_size()}")
    
    # Process second text (adds new words)
    print(f"\nText 2: '{text2}'")
    embeddings2 = embedder.embed_sequence(text2)
    print(f"Token IDs: {embedder.tokenizer.tokenize(text2)}")
    print(f"Embeddings shape: {embeddings2.shape}")
    print(f"Vocabulary size: {embedder.tokenizer.vocab_size()}")
    
    # Process third text (adds more new words)
    print(f"\nText 3: '{text3}'")
    embeddings3 = embedder.embed_sequence(text3)
    print(f"Token IDs: {embedder.tokenizer.tokenize(text3)}")
    print(f"Embeddings shape: {embeddings3.shape}")
    print(f"Vocabulary size: {embedder.tokenizer.vocab_size()}")
    
    # Show vocabulary
    print(f"\nFinal vocabulary: {embedder.tokenizer.word_to_id}")
    
    # Test individual word embedding
    print(f"\nEmbedding for 'hello': {embedder.get_embedding('hello')}")
    print(f"Embedding for 'world': {embedder.get_embedding('world')}")
    
    # Visualize embeddings
    print("\nVisualizing embeddings...")
    embedder.visualize_embeddings()
