import torch
import torch.nn as nn
import torch.nn.functional as F

'''
# Complete pipeline
text -> tokens -> embeddings -> attention -> output
'''

# -------------- 1. Tokenizer

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
    
# -------------- 1. Tokenizer

# -------------- 2. TextEmbedding

class TextEmbedding:  
    def __init__(self, vocab_size, embedding_dim):
        # Just create the embedding weight matrix
        self.weight = torch.randn(vocab_size, embedding_dim)
        print("Embedding matrix shape:", self.weight.shape)  # (10, 8)
    
    def forward(self, x):
        '''

        here is how the indexing to the embedding table looks:

        # Your weight matrix (10 rows, 8 columns):
        weight = tensor([
            # row 0: embedding for <PAD> (index 0)
            [-0.6351,  0.1994,  0.7733, -0.3779, -1.1220,  1.3301,  0.4687, -0.1523],
            # row 1: embedding for <UNK> (index 1)
            [-1.1897, -0.9326, -0.8552,  0.7444,  0.4291, -1.5263,  0.8666, -1.1037],
            # row 2: embedding for first word (index 2)
            [-1.0172, -0.2445, -1.0905,  0.3260, -0.3313, -1.3757,  0.2354, -0.0960],
            # ... and so on
        ])

        # Your input x (3 sequences, 4 tokens each):
        x = tensor([
            [2, 3, 0, 0],  # first sequence of 4 words, remember 0 is the padding <PAD> (index 0) x
            [4, 5, 6, 7],  # second sequence
            [8, 5, 9, 0]   # third sequence
        ])

        # When you do self.weight[x], it looks up each index in x
        # and gets the corresponding row from weight:

        # For first sequence [2, 3, 0, 0]:
        # - index 2 -> gets row 2 of weight matrix
        # - index 3 -> gets row 3 of weight matrix
        # - index 0 -> gets row 0 of weight matrix
        # - index 0 -> gets row 0 of weight matrix

        # For second sequence [4, 5, 6, 7]:
        # - index 4 -> gets row 4 of weight matrix
        # - index 5 -> gets row 5 of weight matrix
        # - index 6 -> gets row 6 of weight matrix
        # - index 7 -> gets row 7 of weight matrix
        '''

        return self.weight[x]
    
# -------------- 2. TextEmbedding

# -------------- 3. Attention

class SimpleAttention():
    def __init__(self, embedding_dim):
        '''
        # Let's say we have the word "bank"
        bank_embedding = [0.5, -0.2, 0.7]  # original embedding

        # Without linear transformations:
        # "bank" only has one representation

        # But "bank" could mean:
        # - financial institution
        # - river bank
        # - to tilt/turn

        # With transformations:
        Q = query(bank_embedding)   # might learn to focus on context
        K = key(bank_embedding)     # might learn to provide context
        V = value(bank_embedding)   # might learn to provide meaning

        1) so q*k creates a new matrix of dot products which words should pay attention to other words, and this produces a weight, 
        2) that weight is then mulitplied by the v vector to produce a new embedding for that word?

        # For "hello world":
        scores = Q * K  
        We're doing matrix multiplication
        But each element in the resulting matrix is a dot product between word vectors
        That's why we need to transpose K (to align the vectors for dot products)
        # Shows how much each word should attend to others
        [
            [0.8, 0.3],  # hello->hello (0.8), hello->world (0.3)
            [0.4, 0.7]   # world->hello (0.4), world->world (0.7)
        ]

        # In attention:
        Q = [[q11, q12],    # 2x2 matrix (2 words, 2 features)
            [q21, q22]]

        K = [[k11, k12],    # 2x2 matrix
            [k21, k22]]

        # Q × K.transpose gives us dot products between rows of Q and rows of K
        scores = Q × K.transpose = [[q1·k1, q1·k2],
                                    [q2·k1, q2·k2]]

        Softmax converts to weights:
        weights = softmax(scores)
        [
            [0.7, 0.3],  # hello pays 70% attention to hello, 30% to world
            [0.4, 0.6]   # world pays 40% attention to hello, 60% to world
        ]

        Multiply by V to get new embeddings:
        # V contains transformed value vectors for each word
        new_embeddings = weights * V

        # For "hello":
        # new_hello = (0.7 * V_hello) + (0.3 * V_world)
        # Combines information from both words based on attention weights

        # For "world":
        # new_world = (0.4 * V_hello) + (0.6 * V_world)

        So each word gets a new embedding that's a weighted mixture of all the value vectors, where the weights come from how much attention it should pay to each word!

        '''
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
    
    def forward(self, x):
        '''
        This (b,t,c) format is important because:

        b: allows processing multiple sequences at once - in this example we are using
        t: represents the sequence of words
        c: represents the features/embedding for each word
        '''
        # x shape: (batch_size, sequence_length, embedding_dim) - 3D tensor or a rank-3 tensor.

        # 1. Each word creates: 
        # Query: "what we're looking for" - A query to search for relevant context
        # Key: "what we have": A key to be searched by other words
        # Value: "what we'll return" A value that gets passed to other words based on relevance
        Q = self.query(x)  # Query: "what we're looking for"
        K = self.key(x)    # Key: "what we have"
        V = self.value(x)  # Value: "what we'll return"

        # 2. Calculate attention scores
        # matmul(Q, K.transpose): How similar each query is to each key
        scores = torch.matmul(Q, K.transpose(-2, -1))
        # 3. Scale scores to prevent softmax from having very small gradients
        scores = scores / (Q.size(-1) ** 0.5)  # divide by sqrt(d_k)
        # 4. Convert scores to probabilities
        attention_weights = F.softmax(scores, dim=-1)
        # 5. Weight values by attention weights
        output = torch.matmul(attention_weights, V)
        return output, attention_weights

# -------------- 3. Attention

class Model():
    def __init__(self, vocab_size, embedding_dim):
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
        self.embedding = TextEmbedding(vocab_size=vocab_size, 
                                       embedding_dim=embedding_dim)
        self.attention = SimpleAttention(embedding_dim)
    
    def forward(self, x):
        '''
        during a forward pass through the training process - we use the word sequence represented as numbers as the index to look up the embedding
        eg. x = torch.tensor([1, 2])  # our sequence "hello dog" as numbers - these numbers are created during the tokenization process
        embeddings = embedding_matrix[x]  # gets rows 1 and 2 from matrix
        '''
        embedded = self.embedding.forward(batch)
        output, weights = self.attention.forward(embedded)
        return output, weights

def prepare_batch(tokens, max_len=None):
    '''
    # Example usage:
    tokens = [
        [1, 2, 3],        # sequence of length 3
        [4, 5],           # sequence of length 2
        [6, 7, 8, 9]      # sequence of length 4
    ]

    padded_batch = prepare_batch(tokens)
    print(padded_batch)

    # Output would look like:
    # tensor([[1, 2, 3, 0],
    #         [4, 5, 0, 0],
    #         [6, 7, 8, 9]])
    '''
    # If no max_len provided, find the longest sequence in the batch
    if max_len is None:
        max_len = max(len(seq) for seq in tokens)
    # Add padding (zeros) to make all sequences the same length - Neural networks expect fixed-size inputs
    # zeros are added to the right side of each sequence to make them all the same length as the longest sequence in the batch
    padded = [seq + [0] * (max_len - len(seq)) for seq in tokens]
    return torch.tensor(padded)

if __name__ == "__main__":
    tokenizer = SimpleTokenizer()
    
    sentences = [
        "hello world",
        "this is a test",
        "attention is cool"
    ]
    
    # create tokens for each sentence
    tokenized = [tokenizer.tokenize(sent) for sent in sentences]
    print("Tokenized sequences:")
    for sent, tokens in zip(sentences, tokenized):
        print(f"{sent}: {tokens}")

    # a batch will have the shape (3, 4) - 3 different sentences, 4 words in each sentence
    batch = prepare_batch(tokenized)
    # After padding
    print("\nBatch shape:", batch.shape)
    print("Padded batch:\n", batch)

    # In practice, when you pass a batch of n sequences through the model, the forward function processes all sequences simultaneously in parallel.
    # The GPU can perform these operations in parallel because:
    # - Matrix operations are parallelizable
    # - Modern GPUs have thousands of cores
    # - PyTorch/CUDA handles the parallelization automatically

    # current example with a batch of 3 sequences, even though we're using the batch format, 
    # it's still running sequentially because we're not using a GPU or enabling parallel processing.
    
    # Create model with a feature dimension of 8, this means each word will have a ten
    embedding_dim = 8
    model = Model(tokenizer.vocab_size(), embedding_dim)
    
    # Forward pass - here we are calling a single forward pass to train to show how we begin to start training
    output, attention_weights = model.forward(batch)
    
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