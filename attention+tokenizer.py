import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

class SimpleTokenizer:
    def __init__(self):
        self.word_to_id = defaultdict(lambda: len(self.word_to_id))
        self.id_to_word = {}
        
        # Add special tokens
        self.PAD_TOKEN = self.word_to_id["<PAD>"]
        self.UNK_TOKEN = self.word_to_id["<UNK>"]
        
        # Update id_to_word mapping for special tokens
        self.id_to_word[self.PAD_TOKEN] = "<PAD>"
        self.id_to_word[self.UNK_TOKEN] = "<UNK>"

    def tokenize(self, text):
        # Simple tokenization by splitting on spaces
        words = text.lower().split()
        tokens = [self.word_to_id[word] for word in words]
        
        # Update id_to_word mapping
        for word in words:
            idx = self.word_to_id[word]
            self.id_to_word[idx] = word
            
        return tokens
    
    def decode(self, token_ids):
        return [self.id_to_word[id] for id in token_ids]

    def vocab_size(self):
        return len(self.word_to_id)

'''class SimpleAttention(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.query = nn.Linear(input_size, input_size)
        self.key = nn.Linear(input_size, input_size)
        self.value = nn.Linear(input_size, input_size)
        
    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / (Q.size(-1) ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights

class TextAttentionModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.attention = SimpleAttention(embedding_dim)
        
    def forward(self, x):
        # Convert token IDs to embeddings
        embedded = self.embedding(x)
        output, weights = self.attention(embedded)
        return output, weights

# Helper function to prepare batch
def prepare_batch(tokens, max_len=None):
    if max_len is None:
        max_len = max(len(seq) for seq in tokens)
    
    # Pad sequences to max length
    padded = [seq + [tokenizer.PAD_TOKEN] * (max_len - len(seq)) for seq in tokens]
    return torch.LongTensor(padded)
'''
# Example usage
if __name__ == "__main__":
    # Initialize tokenizer
    tokenizer = SimpleTokenizer()
    
    # Example sentences
    sentences = [
        "hello world",
        "attention is awesome",
        "deep learning is fun"
    ]
    
    # Tokenize sentences
    tokenized = [tokenizer.tokenize(sent) for sent in sentences]
    print("\nTokenized sequences:")
    for sent, tokens in zip(sentences, tokenized):
        print(f"Original: {sent}")
        print(f"Tokens: {tokens}")
        print(f"Decoded: {' '.join(tokenizer.decode(tokens))}\n")
    
    '''# Prepare batch
    batch = prepare_batch(tokenized)
    print("Batch shape:", batch.shape)
    
    # Initialize model
    vocab_size = tokenizer.vocab_size()
    embedding_dim = 32
    model = TextAttentionModel(vocab_size, embedding_dim)
    
    # Forward pass
    output, attention_weights = model(batch)
    
    print("\nOutput shape:", output.shape)
    print("Attention weights shape:", attention_weights.shape)
    
    # Print attention weights for first sentence
    print("\nAttention weights for first sentence:")
    print(attention_weights[0].round(decimals=2))'''
    
    # Visualize attention (optional, requires matplotlib)
    '''try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            attention_weights[0].detach(),
            annot=True,
            fmt='.2f',
            xticklabels=tokenizer.decode(batch[0]),
            yticklabels=tokenizer.decode(batch[0])
        )
        plt.title("Attention Weights")
        plt.show()
    except ImportError:
        print("Matplotlib not installed. Skipping visualization.")'''