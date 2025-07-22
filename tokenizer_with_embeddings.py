import torch
import torch.nn as nn
from simple_tokenizer import SimpleTokenizer

'''
In this code, the embeddings are just randomly initialized and have no semantic meaning yet. To get meaningful embeddings, we need:


'''

def prepare_batch(tokens, max_len=None):
    if max_len is None:
        max_len = max(len(seq) for seq in tokens)
    padded = [seq + [0] * (max_len - len(seq)) for seq in tokens]
    return torch.LongTensor(padded)

class TextEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
    
    def forward(self, x):
        return self.embedding(x)

if __name__ == "__main__":
    # Test embeddings
    tokenizer = SimpleTokenizer()
    
    # Example sentences
    sentences = ["hello world", "embeddings are cool"]
    tokenized = [tokenizer.tokenize(sent) for sent in sentences]
    
    # Create batch
    batch = prepare_batch(tokenized)
    
    # Create embedding layer
    embedding_dim = 8
    model = TextEmbedding(tokenizer.vocab_size(), embedding_dim)
    
    # Get embeddings
    embedded = model(batch)
    
    print(f"Input shape: {batch.shape}")
    print(f"Embedding shape: {embedded.shape}")
    print(f"\nFirst sentence embeddings:\n{embedded[0].detach().round(decimals=2)}")