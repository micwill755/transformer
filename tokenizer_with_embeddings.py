import torch
import torch.nn as nn
from simple_tokenizer import SimpleTokenizer


class TextEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        '''
        padding_idx=0 tells the embedding layer to handle padding tokens (represented by index 0) in a special way. 
        Here's what it does:
        - It initializes the embedding vector for padding tokens (index 0) to all zeros
        - It freezes this embedding vector - it won't be updated during training
        - It helps the model distinguish between real tokens and padding

        This is useful because:
        - Padding tokens shouldn't carry meaning
        - Zero embeddings for padding help with masking and attention mechanisms
        - It's more efficient as gradients don't need to be computed for padding tokens
        '''
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )

        # Initialize weights using Glorot uniform initialization
        nn.init.xavier_uniform_(self.embedding.weight)
    
    def forward(self, x):
        '''
        during a forward pass through the training process - we use the word sequence represented as numbers as the index to look up the embedding
        eg. x = torch.tensor([1, 2])  # our sequence "hello dog" as numbers - these numbers are created during the tokenization process
        embeddings = embedding_matrix[x]  # gets rows 1 and 2 from matrix
        '''
        embedded = self.embedding(x)
        return embedded

'''
the embeddings are just randomly initialized and have no semantic meaning yet. We will see when we train the model these
embeddings will generate semantic meaning

'''
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