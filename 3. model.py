import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

'''
The SimpleTokenizer takes a sentence, splits it into individual words, 
and converts each unique word into a numeric ID (starting with special 
tokens like PAD and UNK), creating a dictionary that maps words to IDs and vice versa, 
allowing us to transform text into numbers that our neural network can process.
'''
class SimpleTokenizer:
    def __init__(self):
        # We use two dictionaries for bidirectional mapping
        self.word_to_id = {}
        self.id_to_word = {}
        
        # Add special tokens
        self.word_to_id["<PAD>"] = 0
        self.word_to_id["<UNK>"] = 1
        self.id_to_word[0] = "<PAD>"
        self.id_to_word[1] = "<UNK>"
        
        self.PAD_TOKEN = 0
        self.UNK_TOKEN = 1

    def tokenize(self, text):
        words = text.lower().split()
        current_id = len(self.word_to_id)
        
        tokens = []
        for word in words:
            # If word not in dictionary, add it
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
    
    # Helper function to prepare batch
    def prepare_batch(self, tokens, max_len=None):
        if max_len is None:
            max_len = max(len(seq) for seq in tokens)
        
        # Pad sequences to max length - We need padding because neural networks expect fixed-size inputs, 
        # but sentences can have different lengths - here we are simply adding the <PAD> token at the end of sentences that
        # are less than the largest
        padded = [seq + [self.PAD_TOKEN] * (max_len - len(seq)) for seq in tokens]
        # LLMs do use padding in some contexts (especially during training), they employ many sophisticated 
        # techniques to minimize its impact on memory and computation efficiency!
        return torch.LongTensor(padded)

class SimpleAttention(nn.Module):
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

class Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.attention = SimpleAttention(embedding_dim)
        
    def forward(self, x):
        # Convert token IDs to embeddings
        embedded = self.embedding(x)
        output, weights = self.attention(embedded)
        return output, weights


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
    tokenized = [tokenizer.tokenize(sentence) for sentence in sentences]

    print("\nTokenized sequences:")
    for sent, tokens in zip(sentences, tokenized):
        print(f"Original: {sent}")
        print(f"Tokens: {tokens}")
        print(f"Decoded: {' '.join(tokenizer.decode(tokens))}\n")
    
    # Prepare batch
    batch = tokenizer.prepare_batch(tokenized)
    print("Batch shape:", batch.shape)
    
    # Initialize model
    vocab_size = tokenizer.vocab_size()
    embedding_dim = 32
    model = Model(vocab_size, embedding_dim)
    
    # Forward pass
    output, attention_weights = model(batch)
    
    print("\nOutput shape:", output.shape)
    print("Attention weights shape:", attention_weights.shape)
    
    # Print attention weights for first sentence
    print("\nAttention weights for first sentence:")
    print(attention_weights[0].round(decimals=2))
    
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