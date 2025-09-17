#include <cuda_runtime.h>

__global__ void count_2d_equal_kernel(const int* input, int* output, int N, int M, int K) {
     int g_x = threadIdx.x + blockIdx.x * blockDim.x;
     int g_y = threadIdx.y + blockIdx.y * blockDim.y;
     int tid = g_x + g_y * M;

     if(g_x < M && g_y < N){
        if(input[tid] == K)
            atomicAdd(output, 1);
     }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int M, int K) {
    int *d_input, *d_output;
    int size = N * M * sizeof(int);

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, sizeof(int));

    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);

    cudaMemset(d_output, 0, sizeof(int));

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                              (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    count_2d_equal_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N, M, K);
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}
