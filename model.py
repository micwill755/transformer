import torch
import torch.nn as nn
import torch.nn.functional as F
from simple_tokenizer import SimpleTokenizer
from tokenizer_with_embeddings import prepare_batch, TextEmbedding

'''
# Complete pipeline
text -> tokens -> embeddings -> attention -> output
'''
class SimpleAttention(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
    
    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / (Q.size(-1) ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights

class CompleteModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = TextEmbedding(vocab_size, embedding_dim)
        self.attention = SimpleAttention(embedding_dim)
    
    def forward(self, x):
        embedded = self.embedding(x)
        output, weights = self.attention(embedded)
        return output, weights

if __name__ == "__main__":
    # Test complete model
    tokenizer = SimpleTokenizer()
    
    # Example sentences
    sentences = [
        "hello world",
        "this is a test",
        "attention is cool"
    ]
    
    # Process text
    tokenized = [tokenizer.tokenize(sent) for sent in sentences]
    batch = prepare_batch(tokenized)
    
    # Create model
    embedding_dim = 8
    model = CompleteModel(tokenizer.vocab_size(), embedding_dim)
    
    # Forward pass
    output, attention_weights = model(batch)
    
    print("Input sentences:", sentences)
    print(f"Batch shape: {batch.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    # Print attention weights for first sentence
    print("\nAttention weights for first sentence:")
    print(attention_weights[0].detach().round(decimals=2))
    
    # Optional: Visualize attention
    try:
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
        print("Matplotlib not installed for visualization")