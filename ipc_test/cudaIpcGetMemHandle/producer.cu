// producer.cu
#include <stdio.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>

int main() {
    cudaSetDevice(0);

    int *d_data;
    size_t N = 10;
    cudaMalloc(&d_data, N * sizeof(int));

    // 初始化显存数据（全设成 123）
    int init_val = 123;
    cudaMemset(d_data, init_val, N * sizeof(int));

    cudaIpcMemHandle_t handle;
    cudaError_t err = cudaIpcGetMemHandle(&handle, d_data);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaIpcGetMemHandle failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // 把句柄写入文件
    int fd = open("memhandle.bin", O_WRONLY | O_CREAT, 0666);
    write(fd, &handle, sizeof(handle));
    close(fd);

    printf("进程 A: 已写入句柄到 memhandle.bin，等待进程 B 使用...\n");

    // 为了演示方便，这里阻塞，等待 B 完成
    getchar();

    cudaFree(d_data);
    return 0;
}
