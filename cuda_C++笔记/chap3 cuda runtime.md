## 3.1 NVCC编译

`nvcc -arch=sm_86`

## 3.2 cuda runtime

### 1、初始化

使用通过这两个函数调用初始化runtime和指定设备关联的上下文`cudaInitDevice()`，`cudaSetDevice()`

`cudaDeviceReset()`销毁主机线程当前的操作设备的上下文

### 2、设备内存

线性内存地址空间使用`cudaMalloc()`分配内存，使用`cudaFree()`释放内存，使用`cudaMemcpy()`进行H2D拷贝

``` c++
int N = ...;
size_t size = N * sizeof(float);

float* h_A = (float*)malloc(size);
float* d_A;
cudaMalloc(&d_A, size);

cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);	// H2D copy

```

先行内存也可以使用`cudaMallocPitch()`和`cudaMalloc3D()`进行分配，通常用于2D或者3D数组的分配，会进行适当填充以满足对齐要求，提高复制时的性能。

```c++
// Host code
int width = 64, height = 64;
float* devPtr;
size_t pitch;
cudaMallocPitch(&devPtr, &pitch,
                width * sizeof(float), height);
MyKernel<<<100, 512>>>(devPtr, pitch, width, height);

// Device code
__global__ void MyKernel(float* devPtr,
                         size_t pitch, int width, int height)
{
    for (int r = 0; r < height; ++r) {
        float* row = (float*)((char*)devPtr + r * pitch);
        for (int c = 0; c < width; ++c) {
            float element = row[c];
        }
    }
}
```

```c++
__constant__ float constData[256];		// 常量内存数组
float data[256];
cudaMemcpyToSymbol(constData, data, sizeof(data));			// 将主机数据复制到GPU常量内存中
cudaMemcpyFromSymbol(data, constData, sizeof(data));

__device__ float devData;
float value = 3.14f;
cudaMemcpyToSymbol(devData, &value, sizeof(float));

__device__ float* devPointer;
float* ptr;
cudaMalloc(&ptr, 256 * sizeof(float));
cudaMemcpyToSymbol(devPointer, &ptr, sizeof(ptr)); // 将设备内存地址ptr传递给全局指针devPoiter，设备内的程序可以通过这个指针访问这块地址。
```

### 3、设备内存L2访问管理

cuda内核频繁访问的全局内存数据会被L2缓存，而那些只访问一次的数据(流式)应该减小缓存。L2缓存的一部分可以用于持久化对全局内存的数据访问，流式全局内存访问只能在持久访问未使用L2时利用这部分L2.

我们可以调整为持久访问预留的L2缓存大小

```c++
cudaGetDeviceProperties(&prop, device_id);	// 获取指定device的cuda设备属性，填充到cudaDeviceProp结构体中
size_t size = min(int(prop.l2CacheSize * 0.75), prop.persistingL2CacheMaxSize);	// persistingL2CacheMaxSize是设备允许为持久性缓存分配的最大空间
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size); /* set-aside 3/4 of L2 cache for persisting accesses or the max allowed*/
```

关于这个设置还有一些其他限制

**使用持久化访问的L2策略**:

下面代码为cuda流设置访问策略窗口(Access Policy Window)，以帮助GPU，更好地管理L2缓存的数据访问策略。提升数据访问的性能。



```c++
cudaStreamAttrValue stream_attribute;                                         // Stream level attributes data structure
// 指定缓存优化的目标内存地址区域的起始地址。
stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(ptr); // Global Memory data pointer
// 要优化的内存区域大小
stream_attribute.accessPolicyWindow.num_bytes = num_bytes;                    // Number of bytes for persistence access.
                                                                              // (Must be less than cudaDeviceProp::accessPolicyMaxWindowSize)
// 提示编译器该区域大概会有60%的访问命中，是一种优化提示
stream_attribute.accessPolicyWindow.hitRatio  = 0.6;                          // Hint for cache hit ratio
// 在缓存命中时，希望将数据尽可能保留（持久化）在 L2 缓存中
stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting; // Type of access property on cache hit
// 在缓存未命中时，该区域按流式访问处理（容易被驱逐）
stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;  // Type of access property on cache miss.
// 设置访问策略到 CUDA 流中
//Set the attributes to a CUDA stream of type cudaStream_t
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
```

这样访问全局内存[ptr...ptr+num_bytes]比访问其他全局内存更有可能被保留在L2缓存中。

也可以为cuda图形内核节点设置L2持久性.（***什么是图形内核节点？***）

```c++
cudaKernelNodeAttrValue node_attribute;                                     // Kernel level attributes data structure
node_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(ptr); // Global Memory data pointer
node_attribute.accessPolicyWindow.num_bytes = num_bytes;                    // Number of bytes for persistence access.
                                                                            // (Must be less than cudaDeviceProp::accessPolicyMaxWindowSize)
node_attribute.accessPolicyWindow.hitRatio  = 0.6;                          // Hint for cache hit ratio
node_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting; // Type of access property on cache hit
node_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;  // Type of access property on cache miss.

//Set the attributes to a CUDA Graph Kernel node of type cudaGraphNode_t
cudaGraphKernelNodeSetAttribute(node, cudaKernelNodeAttributeAccessPolicyWindow, &node_attribute);
```

L2策略控制常用于不同流并发的情况，来保证不同流高效利用L2缓存，避免不同流逐出彼此的缓存行，导致缓存命中降低。

不同的hitProp

* cudaAccessPropertyStreaming：告诉 CUDA 驱动：“这段内存是**一次性读取或临时性数据**，不需要长时间保留在 L2 缓存中。”**更容易被驱逐**出缓存。**不会影响其他更重要（需要常驻）的数据**的缓存留存。对于不需要缓存的流式数据，可以用这个配置
* cudaAccessPropertyPersisting：尝试在 **L2 缓存中尽可能长时间保留**这段数据，用做热点数据/频繁访问的数据
* cudaAccessPropertyNormal：默认行为，可用于重置hitProp配置

使用示例：

```c++
cudaStream_t stream;
cudaStreamCreate(&stream);                              // Create CUDA stream

cudaDeviceProp prop;                                    // CUDA device properties variable
cudaGetDeviceProperties( &prop, device_id);             // Query GPU properties
size_t size = min( int(prop.l2CacheSize * 0.75) , prop.persistingL2CacheMaxSize );
// set-aside 3/4 of L2 cache for persisting accesses or the max allowed
cudaDeviceSetLimit( cudaLimitPersistingL2CacheSize, size); 
// Select minimum of user defined num_bytes and max window size.
size_t window_size = min(prop.accessPolicyMaxWindowSize, num_bytes);

cudaStreamAttrValue stream_attribute;  // Stream level attributes data structure
stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(data1);               // Global Memory data pointer
stream_attribute.accessPolicyWindow.num_bytes = window_size;                                // Number of bytes for persistence access
stream_attribute.accessPolicyWindow.hitRatio  = 0.6;                                        // Hint for cache hit ratio
stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;// 命中的就持久缓存
stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;// 没有命中就快速驱逐

cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);   // Set the attributes to a CUDA Stream

for(int i = 0; i < 10; i++) {
    
    cuda_kernelA<<<grid_size,block_size,0,stream>>>(data1);
}                 
// 虽然是多个kernel,但是在同一个stream中，会受益于前面的缓存
cuda_kernelB<<<grid_size,block_size,0,stream>>>(data1); 

// 清除上面设置的L2缓存策略
stream_attribute.accessPolicyWindow.num_bytes = 0;                                          
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);   
// 应该就是设置成normal
cudaCtxResetPersistingL2Cache();                                                            

// data2可以用完整的L2缓存，不再受前面的持久性策略限制
cuda_kernelC<<<grid_size,block_size,0,stream>>>(data2);                                     
```

**将持久访问重置为正常状态**

1. 显示使用access属性为cudaAccessPropertyNormal
2. 调用cudaCtxResetPersistingL2Cache()
3. 依赖自动重置。未触及的线条会自动重置为正常。强烈建议**不要依赖自动重置**，因为自动重置所需的时间长度不确定。

**管理L2预留缓存的利用率**

需要考虑的因素包括

1. L2预留缓冲区的大小
2. 并发执行的cuda内核
3. 可以同时执行所有cuda内核的访问策略窗口
4. 及时重置L2配置，来释放预留的L2缓存，使其能被流式访问正常的使用

**查询L2缓存属性**

相关的属性存在cudaDeviceProp结构体中，可以用cudaGetDeviceProperties查询

* l2CacheSize GPU可用的L2缓存量
* persistingL2CacheMaxSize 可留作持久内存访问预留的最大 L2 缓存量
* accessPolicyMaxWindowSize 访问策略窗口的最大L2缓存量

### 4、共享内存

共享内存使用`__shared__`内存空间说明符分配；共享内存被block内的所有线程共享，所以会存在线程同步问题和bank冲突问题。前者需要借助一些同步机制解决，后者需要以合理的方式读取数据，避免一个warp里面的不同线程访问同一个bank的数据引起的竞争。

使用共享内存加速矩阵乘法

```c++
typedef struct {
    int width;		// 矩阵列数
    int height;		// 矩阵行数
    int stride;		// 跨度
    float* elements;// 数据指针
} Matrix;
// Get a matrix element
__device__ float GetElement(const Matrix A, int row, int col)
{
    return A.elements[row * A.stride + col];
}
// Set a matrix element
__device__ void SetElement(Matrix A, int row, int col,
                           float value)
{
    A.elements[row * A.stride + col] = value;
}
// 从矩阵中提取一个BLOCK_SIZE x BLOCK_SIZE的子矩阵Asub，row和col小矩阵在大矩阵中的索引，
 __device__ Matrix GetSubMatrix(Matrix A, int row, int col)
{
    Matrix Asub;
    Asub.width    = BLOCK_SIZE;
    Asub.height   = BLOCK_SIZE;
    Asub.stride   = A.stride;		// 因为计算小矩阵元素索引的时候，和大矩阵跨度是一样的，都是大矩阵的列数
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row
                                         + BLOCK_SIZE * col];	// 计算小矩阵的数据起始地址
    return Asub;
}
// Thread block size
#define BLOCK_SIZE 16
// 前置声明
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);
// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = d_A.stride = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size,
               cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = d_B.stride = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size,
    cudaMemcpyHostToDevice);
    // Allocate C in device memory
    Matrix d_C;
    d_C.width = d_C.stride = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);
    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE); // 一个block负责一小块的计算
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    // Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size,
               cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}
// Matrix multiplication kernel called by MatMul()
 __global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    // 给每个block分配数据
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);
    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    float Cvalue = 0;
    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;
    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
        // Get sub-matrix Asub of A
        Matrix Asub = GetSubMatrix(A, blockRow, m);
        // Get sub-matrix Bsub of B
        Matrix Bsub = GetSubMatrix(B, m, blockCol);
        // Shared memory used to store Asub and Bsub respectively
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);
        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();
        // Multiply Asub and Bsub together
        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += As[row][e] * Bs[e][col];
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }
    // Write Csub to device memory
    // Each thread writes one element
    SetElement(Csub, row, col, Cvalue);
}
```

### 5、线程块簇(Thread Block Clusters)和分布式共享内存(DSM)

Compute Feature 9.0 中引入了线程块簇，一个簇是由多个线程块组成的组，**在一个线程块簇内，线程可以访问其他线程块的共享内存**，不再局限于自己块内的共享内存。相应的地址空间称为分布式共享内存地址空间，只要线程属于同一个线程块簇，它们就可以对分布式共享内存中的任意地址执行读取、写入、原子操作。**无论目标地址是自己线程块的共享内存，还是别的线程块的共享内存，都可以直接访问。**

分布式共享内存的分配依然遵循按照线程块分配的规则，整个DSM的大小 = 每个线程块的SMEM大小 * 簇内线程块数量。

要访问DSM中的数据，需要确保簇中的所有线程块都处于活跃状态，使用`cluster.sync()`来等待同一个簇中的所有线程块都完成启动。同理，需要确保内存访问发生在线程块退出之前。

举例计算直方图应用中分布式共享内存的使用：

```c++
#include <cooperative_groups.h>

// 
__global__ void clusterHist_kernel(int *bins, const int nbins, const int bins_per_block, const int *__restrict__ input,
                                   size_t array_size)
{
  extern __shared__ int smem[];
  namespace cg = cooperative_groups;
  int tid = cg::this_grid().thread_rank();		// 全局线程编号

  // Cluster initialization, size and calculating local bin offsets.
  cg::cluster_group cluster = cg::this_cluster();
  unsigned int clusterBlockRank = cluster.block_rank();	// 当前线程块在cluster中的编号
  int cluster_size = cluster.dim_blocks().x;			// cluster中线程块的总数
  // 每个线程块将自己的共享内存的数据清零
  for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x)
  {
    smem[i] = 0; //Initialize shared memory histogram to zeros
  }

  // 同步cluster中的所有线程保证：线程块已经启动并完成初始化
  cluster.sync();

  // 每个线程处理输入数据的一部分
  for (int i = tid; i < array_size; i += blockDim.x * gridDim.x)
  {
    int ldata = input[i];

    // 映射一下越界的数据
    int binid = ldata;
    if (ldata < 0)
      binid = 0;
    else if (ldata >= nbins)
      binid = nbins - 1;

    // dst_block_rank是计算当前bin属于哪个线程块，
    // dst_offset目标线程块在共享内存中的位置
    // 例如nbins=64，bins_per_block = 16，那么总共有4个block
    // 当前线程拿到的输入是35，那么就知道是放在dst_block_rank=2，dst_offset=3的位置
    int dst_block_rank = (int)(binid / bins_per_block);
    int dst_offset = binid % bins_per_block;

    // 找到分布式共享内存中对应块的地址
    int *dst_smem = cluster.map_shared_rank(smem, dst_block_rank);

    // 原子更新这个bin计数器
    atomicAdd(dst_smem + dst_offset, 1);
  }

  // 确保在s2g以及block退出之前结束所有共享内存的访问
  cluster.sync();

  // Perform global memory histogram, using the local distributed memory histogram
  int *lbins = bins + cluster.block_rank() * bins_per_block;
  for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x)
  {
    atomicAdd(&lbins[i], smem[i]);
  }
}
```

DSM + cluster 模型让线程块间可以**直接在共享内存层次协同计算**，因此相较于通过GMEM协作效率更高

DSM 提供的不是“统一共享内存”，而是**受管理的跨线程块内存窗口**，速度远快于通过全局内存传递数据

### 6、**页面锁定主机内存**

使用`cudaHostAlloc（）` 和 `cudaFreeHost（）` 分配和释放页面锁定的主机内存，或者使用`cudaHostRegister（）`锁定由`cudamalloc（）`分配的内存。

页面锁定主机内存的好处

1. 在某些设备中，页面锁定主机内存和设备内存之间的拷贝可以和内核执行同时发生
2. 在某些设备中，页面锁定的主机内存可以映射到设备的地址空间，而无需发生内存拷贝(零拷贝)
3. 在具有前端总线的系统上，可以提高主机内存和设备内存之间的带宽

### 7、**内存同步域**

a 是一个 cuda::atomic<T, cuda::thread_scope_device> 变量，作用域是 device-scope，如果 thread 1 向 `x` 写入，使用 `a.release()` 来发布数据，只能确保 thread 2 在之后使用 `a.acquire()` 能看到 `x` 的值。

`b` 是一个 `cuda::atomic<T, cuda::thread_scope_system>`，是 system-scope 原子变量。所以，如果 thread 2 用 `b.release()` 发出一个信号，thread 3 使用 `b.acquire()` 接收，它必须确保：**1.** thread 2 自己的写入（例如写 `x`）对 thread 3 可见； **2. **所有在 thread 2 **之前被观察到的其他线程的写入**也要对 thread 3 可见！这就是**累积性**，GPU在处理这种问题的时候是保守的，可能会影响程序的性能。

```
Thread 1:
    x = 42;
    a.release();    // device-scope atomic
Thread 2:
    a.acquire();    // sees x == 42
    b.release();    // system-scope atomic
Thread 3:
    b.acquire();    // must see x == 42 too! (even though it didn't see a)
    
// a 是 device-scope，只保证 Thread 2 可见 x。
// b 是 system-scope，cumulativity 要求 Thread 3 也必须能看到 x
```

### 8、异步并发执行

 **流的创建和销毁**

下面例子通过两个cuda流实现异步的数据传输和内核执行，从而overlap主机与设备之间的数据传输和计算过程

```c++
// 创建cuda流
cudaStream_t stream[2];
for (int i = 0; i < 2; ++i)
    cudaStreamCreate(&stream[i]);

// 分配页锁定的主机内存，支持异步传输和更高的带宽
float* hostPtr;
cudaMallocHost(&hostPtr, 2 * size);

for (int i = 0; i < 2; ++i) {
    // 异步复制
    cudaMemcpyAsync(inputDevPtr + i * size, hostPtr + i * size,
                    size, cudaMemcpyHostToDevice, stream[i]);
    // 计算
    MyKernel <<<100, 512, 0, stream[i]>>>
          (outputDevPtr + i * size, inputDevPtr + i * size, size);
    // 回传
    cudaMemcpyAsync(hostPtr + i * size, outputDevPtr + i * size,
                    size, cudaMemcpyDeviceToHost, stream[i]);
}

// 销毁流
// 如果在调用 cudaStreamDestroy（） 时设备仍在流中工作，则该函数将立即返回
// 并且与流关联的资源将在设备完成流中的所有工作后自动释放
for (int i = 0; i < 2; ++i)
    cudaStreamDestroy(stream[i]);
```

**默认流**

未指定任何 stream 参数的内核启动和托管 `<->` 设备内存副本，或者等效地将 stream 参数设置为零，将颁发给默认流。

**流的显示同步**

* `cudaDeviceSynchronize()`等待所有流上的任务完成，会阻塞主机线程，直到GPU上的所有已提交任务执行完成
* `cudaStreamSynchronize()`，接收stream参数，只同步指定流。当主机希望等等某个流执行完成，但是不想影响其他流的时候使用
* `cudaStreamWaitEvent()`让指定的流等待某个事件完成之后，再继续执行该流后续的命令
* `cudaStreamQuery()` 提供了一种方法，用于检查一个指定的 stream 中所有之前提交的命令是否已经执行完成。    

**流的隐式同步**

当不显示指定cudastream_t的时候，cuda会使用一个默认流，叫做null stream，也就是cudastream_t = 0, 默认流与其他非默认流之间是强同步的，在下面的例子中，kernelA和kernelB不能并发执行

```c++
cudaStream_t s1, s2;
cudaStreamCreate(&s1);  // 默认阻塞流
cudaStreamCreate(&s2);  // 默认阻塞流

kernelA<<<..., ..., 0, s1>>>();     // 在流 s1 中执行
someKernel<<<...>>>();              // 提交到 NULL stream（阻塞其他流）
kernelB<<<..., ..., 0, s2>>>();     // 在流 s2 中执行

```

除非显示地将s1和s2设置为非阻塞流

```c++
cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);

kernelA<<<..., ..., 0, s1>>>();
someKernel<<<...>>>();  // NULL stream
kernelB<<<..., ..., 0, s2>>>();
```

写一个简单的demo验证

```c++
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void sleepKernel(int ms)
{
    unsigned long long start = clock64();
    unsigned long long wait = ms * 1e6; // approximate wait
    while (clock64() - start < wait)
        ;
}

void check(cudaError_t result)
{
    if (result != cudaSuccess)
    {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        exit(-1);
    }
}

int main()
{
    cudaStream_t stream1, stream2;
    cudaEvent_t start, stop;
    float elapsedTime = 0.0f;

    check(cudaStreamCreate(&stream1)); // 默认阻塞流
    check(cudaStreamCreate(&stream2)); // 默认阻塞流
    check(cudaEventCreate(&start));
    check(cudaEventCreate(&stop));

    check(cudaEventRecord(start));

    // 启动两个 kernel，在两个不同的流上
    sleepKernel<<<1, 1, 0, stream1>>>(500); // 500ms
    sleepKernel<<<1, 1>>>(10);              // 小 kernel
    sleepKernel<<<1, 1, 0, stream2>>>(500); // 500ms

    check(cudaEventRecord(stop));
    check(cudaEventSynchronize(stop));
    check(cudaEventElapsedTime(&elapsedTime, start, stop));

    printf("Elapsed time (with implicit sync): %.2f ms\n", elapsedTime);

    // 清理
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    return 0;
}

```

![image-20250601174352407](C:\Users\Qufanhua\AppData\Roaming\Typora\typora-user-images\image-20250601174352407.png)

流之间的overlap

```c++
for (int i = 0; i < 2; ++i)
    cudaMemcpyAsync(inputDevPtr + i * size, hostPtr + i * size,
                    size, cudaMemcpyHostToDevice, stream[i]);
for (int i = 0; i < 2; ++i)
    MyKernel<<<100, 512, 0, stream[i]>>>
          (outputDevPtr + i * size, inputDevPtr + i * size, size);
for (int i = 0; i < 2; ++i)
    cudaMemcpyAsync(hostPtr + i * size, outputDevPtr + i * size,
                    size, cudaMemcpyDeviceToHost, stream[i]);
```

与stream1之间发生的H2D会与stream0的内核启动重叠，

![image-20250601181256914](C:\Users\Qufanhua\AppData\Roaming\Typora\typora-user-images\image-20250601181256914.png)

**主机函数(回调)**

通过`cudaLaunchHostFunc()`在任何时间点将CPU函数插入到cuda流

```c++
void CUDART_CB MyCallback(void *data){
    printf("Inside callback %d\n", (size_t)data);
}
...
for (size_t i = 0; i < 2; ++i) {
    cudaMemcpyAsync(devPtrIn[i], hostPtr[i], size, cudaMemcpyHostToDevice, stream[i]);
    MyKernel<<<100, 512, 0, stream[i]>>>(devPtrOut[i], devPtrIn[i], size);
    cudaMemcpyAsync(hostPtr[i], devPtrOut[i], size, cudaMemcpyDeviceToHost, stream[i]);
    cudaLaunchHostFunc(stream[i], MyCallback, (void*)i);
}
```

**流的优先级**

流的优先级在创建时通过`cudaStreamCreateWithPriority()`指定，可以使用`cudaDeviceGetStreamPriorityRange()`查看允许的优先级范围，数字越小优先级越高，例如-1>0，在运行时，会根据流优先级来确定任务执行顺序。优先级只是一种提示而非保证，并不会抢占已经在处理中的任务

```c++
// get the range of stream priorities for this device
int leastPriority, greatestPriority;
cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
// create streams with highest and lowest available priorities
cudaStream_t st_high, st_low;
cudaStreamCreateWithPriority(&st_high, cudaStreamNonBlocking, greatestPriority));
cudaStreamCreateWithPriority(&st_low, cudaStreamNonBlocking, leastPriority);
```

**编程相关的启动与同步**

![image-20250607170555634](C:\Users\Qufanhua\AppData\Roaming\Typora\typora-user-images\image-20250607170555634.png)

![image-20250607170607705](C:\Users\Qufanhua\AppData\Roaming\Typora\typora-user-images\image-20250607170607705.png)

上图演示了对于secondary kernel 依赖于primary kernel的情况下，依旧可以并行部分，preamble期间执行诸如将缓冲区归零或加载常量值之类的任务，这部分是没有数据依赖的。

上面的并发启动和执行使用*Programmatic Dependent Launch*来实现

**API**

主内核和辅助内核在同一个 CUDA 流中启动。当主内核准备好启动辅助内核时，它应该与所有线程块一起执行 `cudaTriggerProgrammaticLaunchCompletion`

```c++
__global__ void primary_kernel() {
   // Initial work that should finish before starting secondary kernel

   // Trigger the secondary kernel
   // 告诉secondary_kernel，可以开始执行了
   cudaTriggerProgrammaticLaunchCompletion();

   // Work that can coincide with the secondary kernel
}

__global__ void secondary_kernel()
{
   // Independent work

   // Will block until all primary kernels the secondary kernel is dependent on have completed and flushed results to global memory
   cudaGridDependencySynchronize();

   // Dependent work
}

// 配置依赖属性
cudaLaunchAttribute attribute[1];
attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
attribute[0].val.programmaticStreamSerializationAllowed = 1;
configSecondary.attrs = attribute;
configSecondary.numAttrs = 1;

primary_kernel<<<grid_dim, block_dim, 0, stream>>>();
// 使用这种配置后，可以将 secondary kernel 和前面的 primary kernel 串联起来建立程序逻辑依赖
cudaLaunchKernelEx(&configSecondary, secondary_kernel);
```

**CUDA Graphs**

**CUDA Graph** 是 CUDA 10 引入的一种 **任务表示和执行模型**，可以用来表示一系列 GPU 操作（如 kernel 调用、内存拷贝、同步等）以及它们之间的依赖关系，并将这些操作一次性提交给 GPU 执行。

与传统的 CUDA 模型不同，CUDA Graph **将任务的定义和执行分离**，使得任务执行更加高效、结构更清晰

使用cuda graph分成以下三个阶段

1. 定义，图中有哪些操作以及依赖关系
2. 实例化一个图形模板的快照，对其进行验证，执行设置和初始化工作。最大限度地减少启动时需要完成的工作。生成的实例称为可执行图(executable graph)
3. 将可执行图启动到流中

**CUDA Graphs Structure**

一个操作形成图中的一个节点(**node**)，操作之间的依赖关系形成节点之间的边(**edges**)

**Node Type**

节点可以是以下之一

1. kernel
2. CUDA Function Call
3. Memory Copy
4. Memset
5. Empty Node
6. Waiting on an event
7. recording an event
8. signalling an external semaphore（向外部信号量发出信号）
9. waiting on an external semaphore（等待外部信号量）
10. conditianal node（条件节点）
11. Graph Memory Nodes
12. 嵌套的子图

**Edge Data**

每条边由三部分组成：出端、入端和类型。出端指定了何时触发关联的边，入端指定了节点的哪个部分依赖这条边，类型表示两个节点之间的依赖关系

**创建CUDA Graph**

有两种方式

![image-20250607210127941](C:\Users\Qufanhua\AppData\Roaming\Typora\typora-user-images\image-20250607210127941.png)

1. **显式方式创建**

```c++
cudaGraph_t graph;
cudaGraphCreate(&graph, 0);	// 0是保留位，目前固定0

// 内存拷贝节点
// cudaGraphAddMemcpyNode(...);

// 添加kernel调用节点
cudaGraphAddKernelNode(&a, graph, NULL, 0, &nodeParams); // nodeParams是内核参数
cudaGraphAddKernelNode(&b, graph, NULL, 0, &nodeParams); // Null是上游依赖，也就是可以在创建时指定依赖关系
cudaGraphAddKernelNode(&c, graph, NULL, 0, &nodeParams);
cudaGraphAddKernelNode(&d, graph, NULL, 0, &nodeParams);

// 指定依赖关系
cudaGraphAddDependencies(graph, &a, &b, 1);     // A->B
cudaGraphAddDependencies(graph, &a, &c, 1);     // A->C
cudaGraphAddDependencies(graph, &b, &d, 1);     // B->D
cudaGraphAddDependencies(graph, &c, &d, 1);     // C->D

/*
cudaGraphAddDependencies(
    cudaGraph_t graph,         // 图对象
    const cudaGraphNode_t *from, // 上游节点（依赖的前置）
    const cudaGraphNode_t *to,   // 下游节点（依赖的后置）
    size_t numDependencies     // 添加多少对 from→to
);

*/

// 实例化成可执行图
cudaGraphExec_t graphExec;
cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

// 在某个流中执行上面的图
cudaStream_t stream;
cudaStreamCreate(&stream);

cudaGraphLaunch(graphExec, stream);


```

2. 基于stream Capture创建

```c++
// 适合已有 stream 流程的代码自动生成图
cudaGraph_t graph;

cudaStreamBeginCapture(stream);

kernel_A<<< ..., stream >>>(...);
kernel_B<<< ..., stream >>>(...);
libraryCall(stream);
kernel_C<<< ..., stream >>>(...);

cudaStreamEndCapture(stream, &graph);
```

跨流捕获(先跳过)

 
