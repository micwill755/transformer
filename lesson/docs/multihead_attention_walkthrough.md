# Multi-Head Attention: Complete Visual Walkthrough

## Setup Parameters
```python
# Example configuration
batch_size = 2
num_tokens = 3  
d_in = 4        # Input embedding dimension
d_out = 8       # Output dimension
num_heads = 2   # Number of attention heads
head_dim = d_out // num_heads = 4  # Dimension per head
context_length = 4
dropout = 0.1
```

## Input Data
```python
# Input tensor shape: (batch_size, num_tokens, d_in) = (2, 3, 4)
x = [[[1.0, 2.0, 3.0, 4.0],      # Batch 0, Token 0: "The"
      [5.0, 6.0, 7.0, 8.0],      # Batch 0, Token 1: "cat"  
      [9.0, 10.0, 11.0, 12.0]],  # Batch 0, Token 2: "sat"
     
     [[13.0, 14.0, 15.0, 16.0],  # Batch 1, Token 0: "A"
      [17.0, 18.0, 19.0, 20.0],  # Batch 1, Token 1: "dog"
      [21.0, 22.0, 23.0, 24.0]]] # Batch 1, Token 2: "ran"

print("Input shape:", x.shape)  # torch.Size([2, 3, 4])
```

---

## Step 1: Linear Transformations (Q, K, V)

### Create Query, Key, Value matrices
```python
# Each linear layer: (d_in=4) → (d_out=8)
keys = self.W_key(x)     # Shape: (2, 3, 8)
queries = self.W_query(x) # Shape: (2, 3, 8)  
values = self.W_value(x)  # Shape: (2, 3, 8)

# Example output after linear transformation:
queries = [[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],      # Batch 0, Token 0
            [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8],      # Batch 0, Token 1
            [2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]],     # Batch 0, Token 2
           
           [[3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8],      # Batch 1, Token 0
            [4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8],      # Batch 1, Token 1
            [5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8]]]     # Batch 1, Token 2

print("After linear transformation:")
print("queries.shape:", queries.shape)  # torch.Size([2, 3, 8])
print("keys.shape:", keys.shape)        # torch.Size([2, 3, 8])
print("values.shape:", values.shape)    # torch.Size([2, 3, 8])
```

---

## Step 2: Split into Multiple Heads

### Reshape to add head dimension
```python
# Reshape: (batch, tokens, d_out) → (batch, tokens, num_heads, head_dim)
# (2, 3, 8) → (2, 3, 2, 4)

queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
values = values.view(b, num_tokens, self.num_heads, self.head_dim)

# Visual representation of queries after view():
queries_split = [[[[0.1, 0.2, 0.3, 0.4],  [0.5, 0.6, 0.7, 0.8]],      # Batch 0, Token 0: [Head 0, Head 1]
                  [[1.1, 1.2, 1.3, 1.4],  [1.5, 1.6, 1.7, 1.8]],      # Batch 0, Token 1: [Head 0, Head 1]
                  [[2.1, 2.2, 2.3, 2.4],  [2.5, 2.6, 2.7, 2.8]]],     # Batch 0, Token 2: [Head 0, Head 1]
                 
                 [[[3.1, 3.2, 3.3, 3.4],  [3.5, 3.6, 3.7, 3.8]],      # Batch 1, Token 0: [Head 0, Head 1]
                  [[4.1, 4.2, 4.3, 4.4],  [4.5, 4.6, 4.7, 4.8]],      # Batch 1, Token 1: [Head 0, Head 1]
                  [[5.1, 5.2, 5.3, 5.4],  [5.5, 5.6, 5.7, 5.8]]]]     # Batch 1, Token 2: [Head 0, Head 1]

print("After view() - split into heads:")
print("queries.shape:", queries.shape)  # torch.Size([2, 3, 2, 4])
```

### Head Separation Visualization
```
Original 8-dimensional vector per token:
[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
 └─────── Head 0 ──────┘ └─────── Head 1 ──────┘
    [0.1, 0.2, 0.3, 0.4]   [0.5, 0.6, 0.7, 0.8]

Each head now has its own 4-dimensional representation!
```

---

## Step 3: Transpose for Efficient Computation

### Rearrange dimensions for batch processing
```python
# Transpose: (batch, tokens, heads, head_dim) → (batch, heads, tokens, head_dim)
# (2, 3, 2, 4) → (2, 2, 3, 4)

queries = queries.transpose(1, 2)  # Swap dimensions 1 and 2
keys = keys.transpose(1, 2)
values = values.transpose(1, 2)

# After transpose - queries organized by head:
queries_transposed = [[[[0.1, 0.2, 0.3, 0.4],      # Batch 0, Head 0, Token 0
                        [1.1, 1.2, 1.3, 1.4],      # Batch 0, Head 0, Token 1
                        [2.1, 2.2, 2.3, 2.4]],     # Batch 0, Head 0, Token 2
                       
                       [[0.5, 0.6, 0.7, 0.8],      # Batch 0, Head 1, Token 0
                        [1.5, 1.6, 1.7, 1.8],      # Batch 0, Head 1, Token 1
                        [2.5, 2.6, 2.7, 2.8]]],    # Batch 0, Head 1, Token 2
                      
                      [[[3.1, 3.2, 3.3, 3.4],      # Batch 1, Head 0, Token 0
                        [4.1, 4.2, 4.3, 4.4],      # Batch 1, Head 0, Token 1
                        [5.1, 5.2, 5.3, 5.4]],     # Batch 1, Head 0, Token 2
                       
                       [[3.5, 3.6, 3.7, 3.8],      # Batch 1, Head 1, Token 0
                        [4.5, 4.6, 4.7, 4.8],      # Batch 1, Head 1, Token 1
                        [5.5, 5.6, 5.7, 5.8]]]]    # Batch 1, Head 1, Token 2

print("After transpose:")
print("queries.shape:", queries.shape)  # torch.Size([2, 2, 3, 4])
```

### Why Transpose?
```
Before transpose: (batch, tokens, heads, head_dim)
- Hard to process all heads simultaneously
- Each head scattered across the tokens dimension

After transpose: (batch, heads, tokens, head_dim)  
- Easy to process all heads in parallel
- Each head has its own contiguous block
- Perfect for matrix multiplication: heads × (tokens × head_dim)
```

---

## Step 4: Compute Attention Scores

### Matrix multiplication Q @ K^T
```python
# queries: (2, 2, 3, 4)
# keys: (2, 2, 3, 4)
# keys.transpose(2, 3): (2, 2, 4, 3)  # Transpose last two dimensions

attn_scores = queries @ keys.transpose(2, 3)
# Result shape: (2, 2, 3, 3) = (batch, heads, tokens, tokens)

# Example attention scores for Batch 0, Head 0:
attn_scores[0][0] = [[score_00, score_01, score_02],    # Token 0 attending to all tokens
                     [score_10, score_11, score_12],    # Token 1 attending to all tokens  
                     [score_20, score_21, score_22]]    # Token 2 attending to all tokens

print("Attention scores shape:", attn_scores.shape)  # torch.Size([2, 2, 3, 3])
```

### Attention Score Matrix Visualization
```
For each head, we get a 3×3 attention matrix:

        Token_0  Token_1  Token_2
Token_0   0.8     0.2     0.1      ← "The" attending to ["The", "cat", "sat"]
Token_1   0.3     0.9     0.4      ← "cat" attending to ["The", "cat", "sat"]  
Token_2   0.1     0.6     0.7      ← "sat" attending to ["The", "cat", "sat"]

Each head learns different attention patterns!
```

---

## Step 5: Apply Causal Mask

### Prevent future token attention
```python
# Create causal mask for num_tokens=3
mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
# mask_bool = [[False, True,  True ],
#              [False, False, True ],  
#              [False, False, False]]

# Apply mask - set future positions to -inf
attn_scores.masked_fill_(mask_bool, -torch.inf)

# After masking:
attn_scores[0][0] = [[0.8,  -inf, -inf],    # Token 0 can only see itself
                     [0.3,  0.9,  -inf],    # Token 1 can see tokens 0,1
                     [0.1,  0.6,  0.7 ]]    # Token 2 can see tokens 0,1,2

print("After causal masking:")
print("attn_scores.shape:", attn_scores.shape)  # torch.Size([2, 2, 3, 3])
```

### Causal Mask Visualization
```
Before masking (can see future):
[[0.8, 0.2, 0.1],     ← Token 0 sees future tokens (bad!)
 [0.3, 0.9, 0.4],     ← Token 1 sees future tokens (bad!)
 [0.1, 0.6, 0.7]]

After masking (causal):
[[0.8, -∞,  -∞ ],     ← Token 0 only sees itself
 [0.3, 0.9, -∞ ],     ← Token 1 sees tokens 0,1  
 [0.1, 0.6, 0.7]]     ← Token 2 sees tokens 0,1,2
```

---

## Step 6: Softmax and Scaling

### Apply scaled softmax
```python
# Scale by sqrt(head_dim) and apply softmax
scale_factor = keys.shape[-1]**0.5  # sqrt(4) = 2.0
attn_weights = torch.softmax(attn_scores / scale_factor, dim=-1)

# After softmax, each row sums to 1.0:
attn_weights[0][0] = [[1.0,  0.0,  0.0 ],    # Token 0: 100% attention to itself
                      [0.4,  0.6,  0.0 ],    # Token 1: 40% to token 0, 60% to token 1
                      [0.2,  0.3,  0.5 ]]    # Token 2: 20% to token 0, 30% to token 1, 50% to token 2

# Apply dropout
attn_weights = self.dropout(attn_weights)

print("Attention weights shape:", attn_weights.shape)  # torch.Size([2, 2, 3, 3])
```

---

## Step 7: Apply Attention to Values

### Weighted sum of values
```python
# attn_weights: (2, 2, 3, 3)
# values: (2, 2, 3, 4)
# Result: (2, 2, 3, 4)

context_vec = attn_weights @ values

# For each token, we get a weighted combination of all value vectors:
# context_vec[batch][head][token] = Σ(attention_weight[i] × value[i])

print("Context vectors shape:", context_vec.shape)  # torch.Size([2, 2, 3, 4])
```

### Context Vector Calculation Example
```python
# For Batch 0, Head 0, Token 1 ("cat"):
# attention_weights = [0.4, 0.6, 0.0]  # 40% "The", 60% "cat", 0% "sat"
# values = [[v0_head0], [v1_head0], [v2_head0]]

context_vec[0][0][1] = 0.4 × values[0][0][0] + 0.6 × values[0][0][1] + 0.0 × values[0][0][2]
#                      ↑ 40% of "The"        ↑ 60% of "cat"        ↑ 0% of "sat"

# Result: A contextualized representation of "cat" that incorporates information from "The"
```

---

## Step 8: Transpose Back

### Prepare for head concatenation
```python
# Transpose: (batch, heads, tokens, head_dim) → (batch, tokens, heads, head_dim)
# (2, 2, 3, 4) → (2, 3, 2, 4)

context_vec = context_vec.transpose(1, 2)

# After transpose - organized by token:
context_vec_transposed = [[[[h0_t0], [h1_t0]],      # Batch 0, Token 0: [Head 0 result, Head 1 result]
                           [[h0_t1], [h1_t1]],      # Batch 0, Token 1: [Head 0 result, Head 1 result]
                           [[h0_t2], [h1_t2]]],     # Batch 0, Token 2: [Head 0 result, Head 1 result]
                          
                          [[[h0_t0], [h1_t0]],      # Batch 1, Token 0: [Head 0 result, Head 1 result]
                           [[h0_t1], [h1_t1]],      # Batch 1, Token 1: [Head 0 result, Head 1 result]
                           [[h0_t2], [h1_t2]]]]     # Batch 1, Token 2: [Head 0 result, Head 1 result]

print("After transpose back:")
print("context_vec.shape:", context_vec.shape)  # torch.Size([2, 3, 2, 4])
```

---

## Step 9: Concatenate Heads

### Combine all head outputs
```python
# Reshape: (batch, tokens, heads, head_dim) → (batch, tokens, d_out)
# (2, 3, 2, 4) → (2, 3, 8)

context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)

# Concatenation visualization for one token:
# Head 0 output: [a, b, c, d]
# Head 1 output: [e, f, g, h]  
# Combined:      [a, b, c, d, e, f, g, h]  ← 8-dimensional output

print("After head concatenation:")
print("context_vec.shape:", context_vec.shape)  # torch.Size([2, 3, 8])
```

### Head Concatenation Visualization
```
Before concatenation (separate heads):
Token 0: Head 0: [0.1, 0.2, 0.3, 0.4]
         Head 1: [0.5, 0.6, 0.7, 0.8]

After concatenation (combined):
Token 0: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
         └─── Head 0 ────┘ └─── Head 1 ────┘

Each token now has a rich 8-dimensional representation combining insights from both heads!
```

---

## Step 10: Output Projection

### Final linear transformation
```python
# Optional projection to mix head outputs
context_vec = self.out_proj(context_vec)  # (2, 3, 8) → (2, 3, 8)

print("Final output shape:", context_vec.shape)  # torch.Size([2, 3, 8])
```

---

## Complete Shape Transformation Summary

```python
# Input:                    (2, 3, 4)    # (batch, tokens, d_in)
# ↓ Linear Q,K,V
# After linear:             (2, 3, 8)    # (batch, tokens, d_out)
# ↓ Split heads  
# After view:               (2, 3, 2, 4) # (batch, tokens, heads, head_dim)
# ↓ Transpose
# After transpose:          (2, 2, 3, 4) # (batch, heads, tokens, head_dim)
# ↓ Attention computation
# Attention scores:         (2, 2, 3, 3) # (batch, heads, tokens, tokens)
# Context vectors:          (2, 2, 3, 4) # (batch, heads, tokens, head_dim)
# ↓ Transpose back
# After transpose:          (2, 3, 2, 4) # (batch, tokens, heads, head_dim)
# ↓ Concatenate heads
# After view:               (2, 3, 8)    # (batch, tokens, d_out)
# ↓ Output projection  
# Final output:             (2, 3, 8)    # (batch, tokens, d_out)
```

## Key Insights

1. **Weight Splitting**: One large matrix multiplication instead of multiple small ones
2. **Parallel Processing**: All heads computed simultaneously  
3. **Memory Efficiency**: `view()` operations are free (no data copying)
4. **Causal Masking**: Prevents information leakage from future tokens
5. **Head Specialization**: Each head learns different attention patterns
6. **Rich Representations**: Final output combines insights from all heads

The beauty of multi-head attention is that it allows the model to attend to different aspects of the input simultaneously - one head might focus on syntax, another on semantics, etc.
