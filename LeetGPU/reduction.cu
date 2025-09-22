#include <cuda_runtime.h>

__global__ void reduction_kernel(const float* input, float* output, int N){
    // share mem
    extern __shared__ float s_data[];

    // 每个线程加载一个元素到共享内存,优化点：向量读取，tma
    int glo_tid = threadIdx.x + blockDim.x * blockIdx.x;
    int tid = threadIdx.x;
    
    s_data[tid] = glo_tid < N ? input[glo_tid] : 0.0f;

    // 块内规约
    for(int s = blockDim.x /2; s > 0; s>>=1){
        __syncthreads();
        if(tid < s){
            s_data[tid] += s_data[tid + s];
        }
    }

    // smem to gmem
    if(tid == 0){
        atomicAdd(output, s_data[0]);
    }

}


// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {  
    int threads = 1024;
    int blocks = (N + threads - 1) / threads;
    reduction_kernel<<<blocks, threads, threads * sizeof(float)>>>(input, output, N);
    cudaDeviceSynchronize();
}