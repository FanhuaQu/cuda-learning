#include <cuda_runtime.h>

__global__ void reverse_array(float* input, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < N/2){
        float left = input[tid];
        input[tid] = input[N-tid];
        input[N-tid] = left;
    }
}

// input is device pointer
extern "C" void solve(float* input, int N) {
    float *d_input;

    cudaMalloc(&d_input, N * sizeof(float));
    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(d_input, N);
    cudaDeviceSynchronize();

    cudaMemcpy(input, d_input, N * sizeof(float), cudaMemcpyDeviceToHost);
}