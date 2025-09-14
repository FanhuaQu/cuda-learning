#include <cuda_runtime.h>

__device__ unsigned int fnv1a_hash(int input) {
    const unsigned int FNV_PRIME = 16777619;
    const unsigned int OFFSET_BASIS = 2166136261;
    
    unsigned int hash = OFFSET_BASIS;
    
    for (int byte_pos = 0; byte_pos < 4; byte_pos++) {
        unsigned char byte = (input >> (byte_pos * 8)) & 0xFF;
        hash = (hash ^ byte) * FNV_PRIME;
    }
    
    return hash;
}

__global__ void fnv1a_hash_kernel(const int* input, unsigned int* output, int N, int R) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid >= N)
        return;
    int tmp = input[tid];
    for(int i = 0; i < R; i++){
        tmp = fnv1a_hash(tmp);
    }
    output[tid] = tmp;
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, unsigned int* output, int N, int R) {
    int *d_input;
    unsigned int *d_output;

    cudaMalloc(&d_input, N * sizeof(int));
    cudaMalloc(&d_output, N * sizeof(int));

    cudaMemcpy(d_input, input, N * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    fnv1a_hash_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N, R);
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, N * sizeof(int), cudaMemcpyDeviceToHost);
}
