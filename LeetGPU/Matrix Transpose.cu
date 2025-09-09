#include <cuda_runtime.h>

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if(i < cols && j < rows)
        output[j + i * rows] = input[i + j * cols];
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    float *d_input, *d_output;

    // allocate device memory
    cudaMalloc(&d_input, rows * cols * sizeof(float));
    cudaMalloc(&d_output, rows * cols * sizeof(float));

    // H2D copy
    cudaMemcpy(d_input, input, rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    // launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, rows, cols);
    cudaDeviceSynchronize();

    // D2H copy
    cudaMemcpy(output, d_output, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    // free
    cudaFree(d_input);
    cudaFree(d_output);
}
