#include <cuda_runtime.h>

__global__ void softmax_kernel(const float* input, float* output, int N) {
    // native softmax
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx >= N) 
        return;

    // 每个线程遍历整个数组找到最大值（简单但低效）
    float max_val = input[0];
    for (int i = 1; i < N; ++i) {
        if (input[i] > max_val) max_val = input[i];
    }

    float exp_val = expf(input[idx] - max_val);
    output[idx] = exp_val;

    // 确保当前线程在这之前对gmem和smem的写操作，对同一GPU上其他线程都是可见的
    __threadfence();

    float sum = 0.0f;
    for(int i=0; i<N; i++){
        sum += output[i];
    }

    output[idx] = exp_val / sum;

}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}