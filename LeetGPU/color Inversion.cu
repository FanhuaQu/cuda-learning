#include <cuda_runtime.h>

__global__ void invert_kernel(unsigned char* image, int width, int height) {
    // each thread hand 1 pixel, use vector read
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid < width * height){
        uchar4 val = ((const uchar4*)image)[tid];
        val.x = ~val.x;
        val.y = ~val.y;
        val.z = ~val.z;

        // write back
        ((uchar4*)image)[tid] = val;
    }
}
// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(unsigned char* image, int width, int height) {
    unsigned char *d_image;
    int bytes = width * height * 4 * sizeof(unsigned char);
    
    // allocate device memory
    cudaMalloc(&d_image, bytes);

    // H2D copy
    cudaMemcpy(d_image, image, bytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;
    // launch kernel
    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_image, width, height);
    cudaDeviceSynchronize();

    // D2H copy
    cudaMemcpy(image, d_image, bytes, cudaMemcpyDeviceToHost);
}