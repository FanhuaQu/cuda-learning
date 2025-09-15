#include <cuda_runtime.h>

__global__ void copy_matrix_kernel(const float* A, float* B, int N) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid >= N * N)
        return;
    B[tid] = A[tid];
}

// A, B are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, float* B, int N) {
    float *d_A, *d_B;
    int total = N * N;

    cudaMalloc(&d_A, total * sizeof(float));
    cudaMalloc(&d_B, total * sizeof(float));

    cudaMemcpy(d_A, A, total * sizeof(float), cudaMemcpyHostToDevice);

    
    int threadsPerBlock = 256;
    int blocksPerGrid = (total + threadsPerBlock - 1) / threadsPerBlock;
    copy_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, N);
    cudaDeviceSynchronize();

    cudaMemcpy(B, d_B, total * sizeof(float), cudaMemcpyDeviceToHost);
} 
