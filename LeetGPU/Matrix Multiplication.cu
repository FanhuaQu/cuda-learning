#include <cuda_runtime.h>

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;

    if(row < M && col < K){
        float accumulate = 0.0f;
        for(int i=0; i<N; i++){
            accumulate += A[row * N + i] * B[i * K + col]; 
        }
        C[row * K + col] = accumulate;
    }

}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    float *d_A, *d_B, *d_C;

    // malloc device memory
    cudaMalloc(&d_A, M * N * sizeof(float));
    cudaMalloc(&d_B, N * K * sizeof(float));
    cudaMalloc(&d_C, M * K * sizeof(float));

    // copy from host to device
    cudaMemcpy(d_A, A, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * K * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    // copy from device to host
    cudaMemcpy(C, d_C, M * K * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
