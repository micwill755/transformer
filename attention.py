import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleAttention(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        # Simple linear transformations for Q, K, V
        self.query = nn.Linear(input_size, input_size)
        self.key = nn.Linear(input_size, input_size)
        self.value = nn.Linear(input_size, input_size)
        
    def forward(self, x):
        # Generate Q, K, V
        Q = self.query(x)  # Query
        K = self.key(x)    # Key
        V = self.value(x)  # Value

        # Calculate attention scores
        # (batch_size, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1))
        
        # Scale the scores
        scores = scores / (Q.size(-1) ** 0.5)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention weights to values
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights

# Let's try it out!
if __name__ == "__main__":
    # Create a simple example
    batch_size = 2
    seq_length = 3
    input_size = 4
    
    # Create random input
    x = torch.randn(batch_size, seq_length, input_size)
    
    # Initialize the attention module
    attention = SimpleAttention(input_size)
    
    # Get output
    output, weights = attention(x)
    
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("Attention weights shape:", weights.shape)
    
    # Print the attention weights for the first batch
    print("\nAttention weights for first batch:")
    print(weights[0].round(decimals=2))