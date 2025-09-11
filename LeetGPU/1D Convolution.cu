#include <cuda_runtime.h>

__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output,
                                      int input_size, int kernel_size) {
    // get the thread_idx
    int thread_idx = threadIdx.x + blockDim.x * blockIdx.x;
    int output_size = input_size - kernel_size + 1;
    if(thread_idx < output_size){
        float result = 0.0f;
        for(int i = 0; i < kernel_size; i++){
            result += input[thread_idx + i] * kernel[i];
        }
        output[thread_idx] = result;
    }
}

// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, const float* kernel, float* output, int input_size, int kernel_size) {
    float* d_input, *d_kernel, *d_output;
    int output_size = input_size - kernel_size + 1;

    // allocate device memory
    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));
    cudaMalloc(&d_kernel, kernel_size * sizeof(float));

    // H2D copy
    cudaMemcpy(d_input, input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernel_size * sizeof(float), cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_kernel, d_output, input_size, kernel_size);
    cudaDeviceSynchronize();

    // D2H copy
    cudaMemcpy(output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);
}