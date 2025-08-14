import torch
import torch.nn as nn

# Input sentence: "hello world this is a test"
sentence = ["hello", "world", "this", "is", "a", "test"]
batch_size = 1
seq_len = 6
d_model = 4

# Set seed for reproducible results
torch.manual_seed(42)

# Simulated input embeddings
x = torch.randn(batch_size, seq_len, d_model)
print("Input x shape:", x.shape)  # [1, 6, 4]

# Create the linear layers
query_layer = nn.Linear(d_model, d_model)
key_layer = nn.Linear(d_model, d_model)

# Generate Q and K matrices
Q = query_layer(x)  # [1, 6, 4] - what each token is looking for
K = key_layer(x)    # [1, 6, 4] - what each token offers

print("\n" + "="*60)
print("Q MATRIX (Queries - what each token is looking for)")
print("="*60)
print("Q shape:", Q.shape)  # [1, 6, 4]
Q_2d = Q.squeeze(0)  # Remove batch dimension for easier viewing
print("\nQ matrix values:")
print("       dim0    dim1    dim2    dim3")
for i, token in enumerate(sentence):
    row_str = f"{token:>5}"
    for j in range(d_model):
        row_str += f"{Q_2d[i,j].item():8.3f}"
    print(row_str)

print("\n" + "="*60)
print("K MATRIX (Keys - what each token offers)")
print("="*60)
print("K shape:", K.shape)  # [1, 6, 4]
K_2d = K.squeeze(0)
print("\nK matrix values:")
print("       dim0    dim1    dim2    dim3")
for i, token in enumerate(sentence):
    row_str = f"{token:>5}"
    for j in range(d_model):
        row_str += f"{K_2d[i,j].item():8.3f}"
    print(row_str)

print("\n" + "="*60)
print("K TRANSPOSED (for matrix multiplication)")
print("="*60)
K_transposed = K.transpose(-2, -1)  # [1, 4, 6]
K_T_2d = K_transposed.squeeze(0)  # [4, 6]
print("K transposed shape:", K_transposed.shape)
print("\nK^T matrix values:")
print("       hello   world    this      is       a    test")
for i in range(d_model):
    row_str = f"dim{i}"
    for j in range(seq_len):
        row_str += f"{K_T_2d[i,j].item():8.3f}"
    print(row_str)

# Calculate attention scores
scores = torch.matmul(Q, K_transposed)  # [1, 6, 4] @ [1, 4, 6] = [1, 6, 6]
scores_2d = scores.squeeze(0)  # [6, 6]

print("\n" + "="*60)
print("FINAL SCORES MATRIX")
print("="*60)
print("Rows = queries (what each token is looking for)")
print("Cols = keys (what each token offers)")
print("\n       hello   world    this      is       a    test")
for i, token in enumerate(sentence):
    row_str = f"{token:>5}"
    for j in range(seq_len):
        row_str += f"{scores_2d[i,j].item():8.3f}"
    print(row_str)

print(f"\nInterpretation examples:")
print(f"scores[0,1] = {scores_2d[0,1].item():.3f} = how much 'hello' query matches 'world' key")
print(f"scores[2,5] = {scores_2d[2,5].item():.3f} = how much 'this' query matches 'test' key")
print(f"scores[3,4] = {scores_2d[3,4].item():.3f} = how much 'is' query matches 'a' key")

print(f"\nNote: These are raw scores before scaling (÷√d_k) and softmax!")