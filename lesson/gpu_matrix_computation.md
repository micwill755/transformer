# GPU Matrix Computation: From CUDA Cores to Tensor Cores

## GPU Parallelization Hierarchy

GPUs use a hierarchical approach to parallelize large matrix computations:

```
Grid (entire computation)
├── Block 1 (shared memory)
│   ├── Thread 1
│   ├── Thread 2
│   └── ... (up to 1024 threads)
├── Block 2
└── ...
```

## Matrix Multiplication Decomposition

For large matrix multiplication `C = A @ B`:

### Thread Level (Finest Granularity)
```python
# Each thread computes ONE element of output matrix C
thread_id = blockIdx.x * blockDim.x + threadIdx.x
row = thread_id // C_width
col = thread_id % C_width

# Thread computes C[row][col] = sum(A[row][k] * B[k][col])
result = 0
for k in range(A_width):
    result += A[row][k] * B[k][col]
C[row][col] = result
```

### Block Level (Shared Memory Optimization)
```python
# Block processes a TILE of the output matrix (e.g., 16x16)
# Threads in a block cooperate using shared memory

__shared__ float A_tile[TILE_SIZE][TILE_SIZE]
__shared__ float B_tile[TILE_SIZE][TILE_SIZE]

# Each block computes a 16x16 submatrix of C
block_row = blockIdx.y * TILE_SIZE
block_col = blockIdx.x * TILE_SIZE
```

## Practical Example: 4096×4096 Matrix Multiplication

### Grid Configuration:
```python
# Divide into 16x16 tiles
TILE_SIZE = 16
grid_dim = (4096 // 16, 4096 // 16) = (256, 256)  # 65,536 blocks
block_dim = (16, 16) = 256 threads per block

# Total threads: 256 × 256 × 256 = 16.7M threads
# Each thread computes one output element
```

### Memory Access Pattern:
```python
# Block (bx, by) computes C[by*16:(by+1)*16, bx*16:(bx+1)*16]

for tile_k in range(0, 4096, 16):  # Iterate through K dimension
    # Load A_tile and B_tile into shared memory
    A_tile = A[by*16:(by+1)*16, tile_k:tile_k+16]
    B_tile = B[tile_k:tile_k+16, bx*16:(bx+1)*16]
    
    # All threads in block compute partial results
    # using shared memory (much faster than global memory)
```

## Why One Thread Per Element Works

### GPU Threads are Lightweight
- Creating 16M threads takes ~microseconds (not milliseconds like CPU threads)
- No OS context switching overhead
- Threads are just register allocations + instruction pointers

### SIMD Execution Model
```python
# GPU doesn't actually create individual threads
# Instead: 32 threads execute in lockstep (called a "warp")
warp_threads = [thread_0, thread_1, ..., thread_31]
# All 32 threads execute the SAME instruction simultaneously
# Just on different data (SIMD = Single Instruction, Multiple Data)
```

## Memory Hierarchy Utilization

**Global Memory** (slow, large):
- Store full Q, K, V matrices
- ~1TB/s bandwidth

**Shared Memory** (fast, small):
- Cache tiles of Q, K for block computation  
- ~19TB/s bandwidth on A100

**Registers** (fastest, tiny):
- Store individual thread's partial results
- ~100TB/s effective bandwidth

## Row-Per-Thread Challenges

### Memory Access Issues
```python
# One element per thread (good):
Thread 0: C[0][0] = A[0][:] @ B[:][0]  # Accesses A row 0, B col 0
Thread 1: C[0][1] = A[0][:] @ B[:][1]  # Accesses A row 0, B col 1
# Threads in same warp access consecutive memory → coalesced access

# One row per thread (problematic):
Thread 0: C[0][:] = A[0][:] @ B  # Needs entire B matrix
Thread 1: C[1][:] = A[1][:] @ B  # Needs entire B matrix again
# Massive memory bandwidth waste
```

### Register Pressure
```python
# One element: Each thread needs ~10 registers
# One row: Each thread needs row_size × registers
# For 4096-wide row: 40,960 registers per thread!
# GPU has limited registers → fewer active threads → lower utilization
```

## Tensor Cores: The Game Changer

### Traditional CUDA Core vs Tensor Core

**Traditional CUDA Core**:
```python
# One operation per clock cycle
result = a * b + c  # One multiply-accumulate (MAC) operation
```

**Tensor Core**:
```python
# 64+ operations per clock cycle
C_tile[4×4] = A_tile[4×4] @ B_tile[4×4] + C_tile[4×4]  # 64 MAC operations simultaneously!
```

### Performance Comparison

**A100 GPU Specs**:
- CUDA Cores: ~19 TFLOPS (float32)
- Tensor Cores: ~312 TFLOPS (mixed precision)
- **16x speedup** for matrix operations!

### Matrix Multiplication Scaling

**For 4096×4096 matrix**:

**CUDA Cores**: 
- 16M elements = 16M separate operations
- Time: ~16M / (19T ops/sec) = 0.84ms

**Tensor Cores**:
- 16M elements = 1M tile operations (16 elements per tile)
- Time: ~1M / (312T ops/sec) = 0.003ms
- **280x faster!**

## Tensor Core Implementation

### Traditional Approach (one element per thread):
```python
# 16 threads needed for 4×4 matrix tile
Thread 0: C[0][0] = A[0][:] @ B[:][0]
Thread 1: C[0][1] = A[0][:] @ B[:][1]
...
Thread 15: C[3][3] = A[3][:] @ B[:][3]
# 16 separate computations
```

### Tensor Core Approach:
```python
# Single Tensor Core instruction
wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
# Computes entire 4×4 tile in ONE operation
# Same latency as single element!
```

## Real CUDA Kernel Structure

```cuda
__global__ void attention_kernel(float* Q, float* K, float* output) {
    // Block and thread indices
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    // Shared memory tiles
    __shared__ float Q_tile[16][128];
    __shared__ float K_tile[128][16];
    
    // Each thread computes one attention score
    float result = 0.0f;
    
    // Tile across head_dim
    for (int k = 0; k < head_dim; k += 16) {
        // Cooperatively load tiles
        Q_tile[ty][tx] = Q[...];  // Load Q tile
        K_tile[ty][tx] = K[...];  // Load K tile
        __syncthreads();
        
        // Compute partial dot product
        for (int i = 0; i < 16; i++) {
            result += Q_tile[ty][i] * K_tile[i][tx];
        }
        __syncthreads();
    }
    
    output[...] = result;
}
```

## Transformer Attention Optimization

### Attention-Specific GPU Layout
```python
# Shape: (batch, heads, seq_len, head_dim) @ (batch, heads, head_dim, seq_len)
# Result: (batch, heads, seq_len, seq_len)

# Grid layout for seq_len=2048, head_dim=128:
batch_blocks = batch_size
head_blocks = num_heads  
seq_blocks = (2048 + TILE_SIZE - 1) // TILE_SIZE  # e.g., 128 blocks for 16x16 tiles

# Each block handles one (batch, head, seq_tile, seq_tile) computation
```

### Real Transformer Impact
```python
# Attention Computation Q @ K^T:
# Shape: (batch=32, heads=32, seq=2048, dim=128)
# Traditional: 32×32×2048×2048×128 = 274B operations
# Tensor Cores: Same result in ~1/16th the time

# This is why H100s can train massive models efficiently
```

## Modern Optimizations

### Warp-Level Primitives
```python
# Threads in a warp cooperate on larger chunks
for i in range(0, row_size, 32):  # 32 = warp size
    # Each thread in warp handles one element of 32-element chunk
    partial_sum = A[row][i + thread_id] * B[i + thread_id][col]
    # Warp reduction to sum all 32 partial results
    total += warp_reduce_sum(partial_sum)
```

### Hybrid Approach in Modern Libraries
Modern libraries like **cuBLAS** and **cutlass** use:

```python
# Block-level: 128×128 output tile
# Warp-level: 32×32 subtile  
# Thread-level: 8×8 micro-tile using Tensor Cores

# So each thread actually computes 64 elements (8×8)
# But still maintains memory coalescing benefits
```

## Tensor Core Requirements

### Data Type Requirements
- Tensor Cores work best with **mixed precision** (float16/bfloat16)
- Matrix dimensions must be **multiples of tile size** (8, 16, etc.)
- Need special programming (WMMA API or libraries like cuBLAS)

### PyTorch Integration
```python
# Automatically uses Tensor Cores when possible
with torch.cuda.amp.autocast():  # Mixed precision
    result = torch.matmul(A, B)   # Uses Tensor Cores under the hood
```

## Key Insights

**Memory Bandwidth > Compute**:
- Modern GPUs: ~1000 TFLOPS compute, ~1 TB/s memory
- Bottleneck is getting data to compute units, not the computation itself
- One-element-per-thread maximizes memory throughput
- "Excessive" thread creation is essentially free

**Tensor Cores Revolution**:
- Process **64+ matrix elements in the same time** a traditional core processes **1 element**
- Specialized hardware that makes transformers computationally feasible at scale
- **16-280x speedup** for matrix operations depending on the workload

**Architecture Matters**:
- GPUs are fundamentally different from CPUs
- Massive parallelism is the path to efficiency
- What seems wasteful (millions of threads) is actually optimal
