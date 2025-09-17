#include <cuda_runtime.h>

__global__ void sigmoid_linear_kernel(const float* input, float* output, int N){
    int vec_tid = threadIdx.x + blockIdx.x * blockDim.x;
    int base = vec_tid * 4;

    if(base + 3 < N){
        float4 in4 = reinterpret_cast<const float4*>(input)[vec_tid];

        float4 out4;

        out4.x = in4.x / (1.0f + expf(-in4.x));
        out4.y = in4.y / (1.0f + expf(-in4.y));
        out4.z = in4.z / (1.0f + expf(-in4.z));
        out4.w = in4.w / (1.0f + expf(-in4.w));

        reinterpret_cast<float4*>(output)[vec_tid] = out4;
    }
    else{
        for(int i = 0; i < 4 && base + i < N; i++){
            float v = input[base + i];
            output[base + i] = v / (1.0f + expf(-v));
        }
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {

    int threadsPerBlock = 256;
    int vecN = (N + 3) / 4;
    int blocksPerGrid = (vecN + threadsPerBlock - 1) / threadsPerBlock;

    sigmoid_linear_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
