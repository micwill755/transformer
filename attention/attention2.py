import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
value_layer = nn.Linear(d_model, d_model)

# Generate Q, K, and V matrices
Q = query_layer(x)  # [1, 6, 4] - what each token is looking for
K = key_layer(x)    # [1, 6, 4] - what each token offers
V = value_layer(x)  # [1, 6, 4] - what each token contains (actual information)

print("\n" + "="*70)
print("STEP 1: Q, K, V MATRICES")
print("="*70)

# Print Q matrix
print("\nQ MATRIX (Queries - what each token is looking for)")
print("-" * 50)
Q_2d = Q.squeeze(0)
print("       dim0    dim1    dim2    dim3")
for i, token in enumerate(sentence):
    row_str = f"{token:>5}"
    for j in range(d_model):
        row_str += f"{Q_2d[i,j].item():8.3f}"
    print(row_str)

# Print K matrix
print("\nK MATRIX (Keys - what each token offers)")
print("-" * 50)
K_2d = K.squeeze(0)
print("       dim0    dim1    dim2    dim3")
for i, token in enumerate(sentence):
    row_str = f"{token:>5}"
    for j in range(d_model):
        row_str += f"{K_2d[i,j].item():8.3f}"
    print(row_str)

# Print V matrix
print("\nV MATRIX (Values - actual information each token contains)")
print("-" * 50)
V_2d = V.squeeze(0)
print("       dim0    dim1    dim2    dim3")
for i, token in enumerate(sentence):
    row_str = f"{token:>5}"
    for j in range(d_model):
        row_str += f"{V_2d[i,j].item():8.3f}"
    print(row_str)

print("\n" + "="*70)
print("STEP 2: CALCULATE RAW ATTENTION SCORES (Q @ K^T)")
print("="*70)

# Calculate raw attention scores
K_transposed = K.transpose(-2, -1)  # [1, 4, 6]
raw_scores = torch.matmul(Q, K_transposed)  # [1, 6, 6]
raw_scores_2d = raw_scores.squeeze(0)

print("Raw scores matrix (before scaling):")
print("       hello   world    this      is       a    test")
for i, token in enumerate(sentence):
    row_str = f"{token:>5}"
    for j in range(seq_len):
        row_str += f"{raw_scores_2d[i,j].item():8.3f}"
    print(row_str)

print("\n" + "="*70)
print("STEP 3: SCALE SCORES BY √d_k")
print("="*70)

# Scale by sqrt(d_k) to prevent softmax saturation
d_k = d_model  # In this case, d_k = d_model = 4
scale_factor = math.sqrt(d_k)
print(f"d_k = {d_k}")
print(f"Scale factor = √d_k = √{d_k} = {scale_factor:.3f}")

scaled_scores = raw_scores / scale_factor
scaled_scores_2d = scaled_scores.squeeze(0)

print(f"\nScaled scores matrix (raw_scores / {scale_factor:.3f}):")
print("       hello   world    this      is       a    test")
for i, token in enumerate(sentence):
    row_str = f"{token:>5}"
    for j in range(seq_len):
        row_str += f"{scaled_scores_2d[i,j].item():8.3f}"
    print(row_str)

print("\nComparison of scaling effect:")
print(f"Before scaling - hello->world: {raw_scores_2d[0,1].item():.3f}")
print(f"After scaling  - hello->world: {scaled_scores_2d[0,1].item():.3f}")

print("\n" + "="*70)
print("STEP 4: APPLY SOFTMAX TO GET ATTENTION WEIGHTS")
print("="*70)

# Apply softmax to get attention probabilities
attention_weights = F.softmax(scaled_scores, dim=-1)  # Apply softmax along last dimension
attention_weights_2d = attention_weights.squeeze(0)

print("Attention weights matrix (after softmax):")
print("       hello   world    this      is       a    test   SUM")
for i, token in enumerate(sentence):
    row_str = f"{token:>5}"
    row_sum = 0
    for j in range(seq_len):
        weight = attention_weights_2d[i,j].item()
        row_str += f"{weight:8.3f}"
        row_sum += weight
    row_str += f"{row_sum:8.3f}"
    print(row_str)

print("\nKey properties of attention weights:")
print("- Each row sums to 1.0 (probability distribution)")
print("- All values are between 0 and 1")
print("- Higher values = stronger attention")

print("\n" + "="*70)
print("STEP 5: WEIGHTED SUM OF VALUES (ATTENTION OUTPUT)")
print("="*70)

# Calculate the final output by weighting values with attention weights
output = torch.matmul(attention_weights, V)  # [1, 6, 6] @ [1, 6, 4] = [1, 6, 4]
output_2d = output.squeeze(0)

print("Final output matrix (attention_weights @ V):")
print("       dim0    dim1    dim2    dim3")
for i, token in enumerate(sentence):
    row_str = f"{token:>5}"
    for j in range(d_model):
        row_str += f"{output_2d[i,j].item():8.3f}"
    print(row_str)

print("\n" + "="*70)
print("STEP 6: DETAILED EXAMPLE - HOW 'HELLO' OUTPUT IS CALCULATED")
print("="*70)

print("For token 'hello' (index 0):")
print("\n1. Attention weights for 'hello':")
for j, target_token in enumerate(sentence):
    weight = attention_weights_2d[0, j].item()
    print(f"   Attention to '{target_token}': {weight:.3f}")

print("\n2. Value vectors being weighted:")
for j, target_token in enumerate(sentence):
    print(f"   V[{target_token}] = [{V_2d[j,0].item():.3f}, {V_2d[j,1].item():.3f}, {V_2d[j,2].item():.3f}, {V_2d[j,3].item():.3f}]")

print("\n3. Weighted sum calculation:")
print("   output[hello] = Σ(attention_weight[i] * V[i])")

# Manual calculation for verification
manual_output = torch.zeros(d_model)
for j in range(seq_len):
    weight = attention_weights_2d[0, j]
    value_vec = V_2d[j, :]
    weighted_value = weight * value_vec
    manual_output += weighted_value
    print(f"   + {weight.item():.3f} * [{value_vec[0].item():.3f}, {value_vec[1].item():.3f}, {value_vec[2].item():.3f}, {value_vec[3].item():.3f}] = [{weighted_value[0].item():.3f}, {weighted_value[1].item():.3f}, {weighted_value[2].item():.3f}, {weighted_value[3].item():.3f}]")

print(f"\n   Final result: [{manual_output[0].item():.3f}, {manual_output[1].item():.3f}, {manual_output[2].item():.3f}, {manual_output[3].item():.3f}]")
print(f"   Matrix result: [{output_2d[0,0].item():.3f}, {output_2d[0,1].item():.3f}, {output_2d[0,2].item():.3f}, {output_2d[0,3].item():.3f}]")
print(f"   ✓ Match: {torch.allclose(manual_output, output_2d[0, :], atol=1e-6)}")

print("\n" + "="*70)
print("SUMMARY: COMPLETE ATTENTION MECHANISM")
print("="*70)
print("1. Input embeddings → Q, K, V matrices (different learned transformations)")
print("2. Raw scores = Q @ K^T (compatibility between queries and keys)")
print("3. Scaled scores = raw_scores / √d_k (prevents softmax saturation)")
print("4. Attention weights = softmax(scaled_scores) (probability distribution)")
print("5. Output = attention_weights @ V (weighted sum of values)")
print("\nEach token's output is a weighted combination of all tokens' values,")
print("where the weights are determined by query-key compatibility!")
