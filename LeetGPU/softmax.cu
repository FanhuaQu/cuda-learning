#include <cuda_runtime.h>
#include <vector>
#include <cmath>

// Kernel A: 每 block 计算该 block 范围内的最大值，写入 block_max[blockIdx.x]
__global__ void kernel_block_max(const float* __restrict__ input, float* block_max, int N) {
    extern __shared__ float sdata[]; // 动态共享内存
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    float v = -INFINITY;
    // 每个线程遍历 stride 步长的元素，取局部最大
    for (int i = gid; i < N; i += gridDim.x * blockDim.x) {
        v = fmaxf(v, input[i]);
    }
    sdata[tid] = v;
    __syncthreads();

    // block 内归约求最大值
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_max[blockIdx.x] = sdata[0];
    }
}

// Kernel B: 使用 global_max 计算 exp(input - global_max) 写入 output，同时每 block 求 block_sum -> 写入 block_sum[blockIdx.x]
__global__ void kernel_exp_and_blocksum(const float* __restrict__ input, float* output, float* block_sum, int N, float global_max) {
    extern __shared__ float sdata[]; // 用于 block 内求和
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    float sum = 0.0f;
    // 每个线程累加它负责的元素的 exp
    for (int i = gid; i < N; i += gridDim.x * blockDim.x) {
        float v = expf(input[i] - global_max);
        output[i] = v;
        sum += v;
    }

    // 把每线程的局部和写到共享内存，然后归约
    sdata[tid] = sum;
    __syncthreads();

    // block 内归约求和
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_sum[blockIdx.x] = sdata[0];
    }
}

// Kernel C: 全局 sum 已知，做最终归一化 output[i] /= global_sum
__global__ void kernel_normalize(float* output, int N, float global_sum) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = gid; i < N; i += stride) {
        output[i] = output[i] / global_sum;
    }
}


// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = std::min((N + threadsPerBlock - 1) / threadsPerBlock, 1024);
    size_t shared_bytes = threadsPerBlock * sizeof(float);

    // Step 1: 每个 block 计算局部最大值
    float* d_block_max;
    cudaMalloc(&d_block_max, blocksPerGrid * sizeof(float));
    kernel_block_max<<<blocksPerGrid, threadsPerBlock, shared_bytes>>>(input, d_block_max, N);

    // copy block_max to host and find global max
    // 因为比较短，在cpu上搞定
    std::vector<float> h_block_max(blocksPerGrid);
    cudaMemcpy(h_block_max.data(), d_block_max, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);
    float global_max = -INFINITY;
    for (int i = 0; i < blocksPerGrid; ++i) {
        if (h_block_max[i] > global_max) global_max = h_block_max[i];
    }

    // kernel B: 计算 exp(input - global_max) 并求 block_sum
    float* d_block_sum;
    cudaMalloc(&d_block_sum, blocksPerGrid * sizeof(float));
    kernel_exp_and_blocksum<<<blocksPerGrid, threadsPerBlock, shared_bytes>>>(input, output, d_block_sum, N, global_max);

    // copy block_sum to host and find global sum
    std::vector<float> h_block_sum(blocksPerGrid);
    cudaMemcpy(h_block_sum.data(), d_block_sum, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);
    float global_sum = 0.0f;
    for (int i = 0; i < blocksPerGrid; ++i) {
        global_sum += h_block_sum[i];
    }

    // kernel C: 归一化
    kernel_normalize<<<blocksPerGrid, threadsPerBlock>>>(output, N, global_sum);

    // softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}