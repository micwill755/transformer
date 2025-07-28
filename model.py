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
        ''' Creates a matrix of random numbers
        vocab_size rows (one for each word in vocabulary)
        embedding_dim columns (features for each word)''' 

        self.weight = torch.randn(vocab_size, embedding_dim)

        '''
        In this case:
        vocab_size = 10 (number of unique words in your vocabulary - all unique words in sentences + <PAD> & <UNK>)
        embedding_dim = 8 (each word gets 8 random numbers)

        So self.weight looks like:
        [
            [-0.6351, 0.1994, 0.7733, -0.3779, -1.1220, 1.3301, 0.4687, -0.1523], # word_id 0 (<PAD>)
            [-1.1897, -0.9326, -0.8552, 0.7444, 0.4291, -1.5263, 0.8666, -1.1037], # word_id 1 (<UNK>)
            [-1.0172, -0.2445, -1.0905, 0.3260, -0.3313, -1.3757, 0.2354, -0.0960], # word_id 2 ("hello")
            ...  # and so on for each word
        ]
        
        The key points:

        Weight matrix is initialized once
        Same weights are used for all forward passes
        In training, these weights would be updated through backpropagation
        But in our example, they stay fixed as random values
        '''

        print("Embedding matrix shape:", self.weight.shape) 
    
    def forward(self, x):
        # x contains word IDs, like [2, 3, 0, 0] for "hello world <PAD> <PAD>"
        # self.weight[x] looks up the corresponding rows from the weight matrix
        return self.weight[x]
    
# -------------- 2. TextEmbedding

# -------------- 3. Attention

class SimpleAttention():
    '''
    Each instance of SimpleAttention has its own Q, K, V matrices that are:

    Initialized once when the model is created
    Used repeatedly in forward passes and updated during training 
    Each instance can learn to focus on different aspects of the input

    In an real world example a model will have n number of attention heads, and each can learn to focus on different types of semantics, relationships 
    or patterns in the data through training 

    '''
    def __init__(self, embedding_dim):
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        '''
        nn.Linear(embedding_dim, embedding_dim) creates:
            1. A weight matrix of shape (embedding_dim, embedding_dim)
            2. A bias vector of shape (embedding_dim)

        In this example model:
        Each of these creates:
            - Weight matrix shape: (8, 8)
            - Bias vector shape: (8)
        self.query = nn.Linear(8, 8)  # Weight_Q(8,8) and bias_Q(8)
        self.key = nn.Linear(8, 8)    # Weight_K(8,8) and bias_K(8)
        self.value = nn.Linear(8, 8)  # Weight_V(8,8) and bias_V(8)
        '''
    
    def forward(self, x):
        '''
        When we do self.query(x), it performs:
        Q = x * Weight_Q + bias_Q

        example:

        Input x (one word embedding):
        x = [-1.0172, -0.2445, -1.0905, 0.3260, -0.3313, -1.3757, 0.2354, -0.0960]

        Query transformation:
        1. x * Weight_Q (8x8 matrix)
        2. Add bias_Q (8 values)
        Results in new 8-dimensional vector
        '''
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