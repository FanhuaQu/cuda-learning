// consumer.cu
#include <stdio.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>

int main() {
    cudaSetDevice(1);

    cudaIpcMemHandle_t handle;

    // 从文件读取句柄
    int fd = open("memhandle.bin", O_RDONLY);
    read(fd, &handle, sizeof(handle));
    close(fd);

    int canAccessPeer = 0;
    if(cudaSuccess != cudaDeviceCanAccessPeer(&canAccessPeer, 1, 0)){
        printf("cudaDeviceCanAccessPeer failed\n");
    }
    else{
        printf("cudaDeviceCanAccessPeer success, canAccessPeer=%d\n", canAccessPeer);
    }

    int *d_data;
    cudaError_t err = cudaIpcOpenMemHandle((void**)&d_data, handle, cudaIpcMemLazyEnablePeerAccess);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaIpcOpenMemHandle failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    else{
        printf("cudaIpcOpenMemHandle success, canAccessPeer=%d\n", canAccessPeer);
    }
    printf("canAccessPeer=%d\n", canAccessPeer);

    // 读取数据
    size_t N = 10;
    int h_data[10] = {0};
    cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("进程 B: 从共享显存读取到数据：\n");
    for (size_t i = 0; i < N; i++) {
        printf("%d ", h_data[i]);
    }
    printf("\n");

    cudaIpcCloseMemHandle(d_data);
    return 0;
}
