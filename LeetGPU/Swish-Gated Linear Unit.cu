#include <cuda_runtime.h>

__global__ void swiglu_kernel(const float* input, float* output, int halfN) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < halfN){
        float x1 = input[tid];
        float x2 = input[tid + halfN];

        output[tid] = x1 / (1 + expf(-x1)) * x2;
    }

}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int halfN = N / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

    swiglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    cudaDeviceSynchronize();
}
