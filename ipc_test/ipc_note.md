主机线程创建的任何[设备内存指针](https://zhida.zhihu.com/search?content_id=174095862&content_type=Article&match_order=1&q=设备内存指针&zhida_source=entity)或[事件句柄](https://zhida.zhihu.com/search?content_id=174095862&content_type=Article&match_order=1&q=事件句柄&zhida_source=entity)都可以被同一进程内的任何其他线程直接引用。然而，它在此进程之外无效，因此不能被属于不同进程的线程直接引用。

要跨进程共享设备内存指针和事件，应用程序必须使用**[进程间通信](https://zhida.zhihu.com/search?content_id=174095862&content_type=Article&match_order=1&q=进程间通信&zhida_source=entity) API**

IPC API **仅支持 Linux 上的 64 位进程和计算能力 2.0 及更高版本的设备**。

使用此 API，应用程序可以使用 `cudaIpcGetMemHandle()` 获取给定设备内存指针的 IPC 句柄，使用标准 IPC 机制（例如，进程间共享内存或文件）将其传递给另一个进程，并使用 `cudaIpcOpenMemHandle()` 检索设备来自 IPC 句柄的指针，它是另一个进程中的有效指针。可以使用类似的入口点共享事件句柄。

# **cudaIpcGetMemHandle**

获取现有设备内存分配的进程间内存句柄，用于**进程间共享 GPU 内存**。作用是：给定一块在当前进程中分配的 GPU 内存（`cudaMalloc` 分配的），获取一个 **可跨进程传递的内存句柄（handle）**，这样其他进程就可以通过这个句柄访问这块显存

```c++
/*
handle：是指向cudaIpcMemHandle_t 结构的指针，函数会在这里填入生成的内存句柄
dev_Ptr：由 cudaMalloc 分配的 GPU 内存指针（必须是 device memory）
*/
cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t *handle, void *devPtr);
// 如果使用cudaFree释放内存区域并且随后调用cudaMalloc返回具有相同设备地址的内存，则cudaIpcGetMemHandle将返回新内存的唯一句柄。
```

# **cudaIpcOpenMemHandle**

打开从另一个进程导出的进程间内存句柄并返回可用于本地进程的设备指针

```c++
__host__ cudaError_t cudaIpcOpenMemHandle ( void** devPtr, cudaIpcMemHandle_t handle, unsigned int  flags )
//打开从另一个进程导出的进程间内存句柄并返回可用于本地进程的设备指针。
//参数
//devPtr - 返回设备指针
//handle - cudaIpcMemHandle 打开
//flags - 此操作的标志。必须指定为cudaIpcMemLazyEnablePeerAccess
```

举个跨进程访问显存的例子

```c++
// producer.cu
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


// consumer.cu
// consumer.cu
#include <stdio.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>

int main() {
    cudaSetDevice(0);

    cudaIpcMemHandle_t handle;

    // 从文件读取句柄
    int fd = open("memhandle.bin", O_RDONLY);
    read(fd, &handle, sizeof(handle));
    close(fd);

    int *d_data;
    cudaError_t err = cudaIpcOpenMemHandle((void**)&d_data, handle, cudaIpcMemLazyEnablePeerAccess);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaIpcOpenMemHandle failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

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
```



# **cudaDeviceCanAccessPeer**

查询设备是否可以直接访问对等设备的内存。

```c++
__host__ cudaError_t cudaDeviceCanAccessPeer ( int* canAccessPeer, int  device, int  peerDevice )
//Parameters
//canAccessPeer - 返回访问能力
//device - 可以直接访问peerDevice上的分配的设备。
//peerDevice - 设备直接访问的分配所在的设备。
    
//如果设备设备能够直接从peerDevice访问内存，则在*canAccessPeer 中返回值 1 ，否则返回0。
//如果可以从设备直接访问peerDevice，则可以通过调用cudaDeviceEnablePeerAccess()来启用访问。
```

刚才的例子

```c++
cudaSetDevice(1);
...
int canAccessPeer = 0;
if(cudaSuccess != cudaDeviceCanAccessPeer(&canAccessPeer, 1, 0)){
    printf("cudaDeviceCanAccessPeer failed\n");
}
printf("canAccessPeer=%d\n", canAccessPeer);

/*
cudaDeviceCanAccessPeer success, canAccessPeer=1
canAccessPeer=1
*/

```

在实际使用的时候，通常是在启动kernel之前通过`cudaIpcGetMemHandle`以及`cudaIpcOpenMemHandle`以及通过all-gather操作来获得每张卡开辟显存的指针。这样在每张卡就能通过这个指针访问其他rank的数据，就和访问本地显存一样。



# ipc和NCCL以及NVSHMEM是什么关系？

首先**cuda ipc**是指cuda 进程间通信，作用是同一台机器上的多个进程共享GPU内存，只能在同机器上进行，底层是依赖于**UVA(统一虚拟寻址)**和GPU peer access。提供**基础的跨进程 GPU 内存共享能力**，相当于最原始的“拼积木”接口

**NCCL**全称是NVIDIA Collective Communication Library。作用是实现多 GPU/多节点的高效通信（AllReduce、Broadcast、Gather 等）。不仅能在同机多 GPU 间通信，还能跨节点（多机）通信。底层会根据拓扑（NVLink、PCIe、InfiniBand 等）选择最快的传输路径。

* 在 **单机多 GPU** 情况下，NCCL 可以用 **CUDA IPC + P2P 访问 (NVLink/PCIe)** 来实现 GPU↔GPU 数据交换
* 在 **多机** 情况下，NCCL 会用 **InfiniBand/RDMA** 等网络传输
* **定位**：构建在 CUDA IPC 等机制之上的**分布式通信框架**，主要面向深度学习。

**NVSHMEM**是更通用的 **PGAS (Partitioned Global Address Space) 编程模型**

功能：

* 提供类似 **OpenSHMEM** 的接口，把多 GPU/多进程的显存抽象成一块“对称共享内存”。
* 支持 GPU 内核直接发起 **one-sided 通信**（put/get/atomic），而不是必须依赖 CPU 协调
* 支持 **单机多 GPU**，也支持 **跨机多 GPU**（走 IB/RDMA）

底层实现

* 单机时：可能使用 **CUDA IPC + CUDA P2P** 来访问远程 GPU 内存。
* 跨机时：走 **GPUDirect RDMA**。





