https://arxiv.org/html/2507.04786?_immersive_translate_auto_translate=1

https://www.zhihu.com/search?type=content&q=NCCL

NCCL(The NVIDIA Collective Communication Library)是GPU集群之间高性能通讯的关键软件层。**是开源的**，但是其内部设计不太透明，通信管道是怎么编排的？协议选择？跨设备的内存移动怎么处理？这些都可能成为我们跨设备开发的性能瓶颈，一旦出问题，怎么才能定位到问题，而不是局限于hang住来解释？

这篇文章主要阐述了NCCL的通信协议变体，节点内部以及节点之间的数据移动机制以及环状/树状通信算法

## I Introduction

与 MPI [[3](https://arxiv.org/html/2507.04786v2#bib.bib3)] 等通用消息传递框架不同，NCCL 专门针对 GPU 到 GPU 的交互，利用 NVLink、PCIe 和 InfiniBand（IB）等互连技术来实现高带宽和低延迟。

在本文中，我们对 NCCL 的内部架构进行了彻底而系统的探索。我们的分析专门针对 NCCL 实施的四个主要方面：（1） 总体概述，包括 API 结构和通信渠道管理;（2） 详细检查通信协议（Simple、LL、LL128）;（3） 对其数据传输模型的分析;（4）对其集体通信算法进行综合分析，全面分析了基于环（Ring）和树（Tree）的集合通信算法。



# II NCCL Overview

NCCL 专为 GPU 集群提供高度优化的集体通信作而设计，强调**低延迟和高带宽**。NCCL 的核心是通过清晰高效的 API 管理 GPU 到 GPU 的通信，该 API 抽象出复杂的技术细节。NCCL 主要为用户提供四类功能。

## A、NCCL API

### 1、 **Communicator Management (通信器管理)**

与 MPI 类似，NCCL 中的所有通信作都是在**通信器的**上下文中执行的。参与通信的每个 GPU 都维护一个通信器对象，用于调用 NCCL 作。用户必须首先初始化通信器并定义所涉及的 GPU 集。当所有设备在单个进程中管理时，可使用ncclCommInitAll 统一创建 ；对于多进程或多线程环境，每个进程需调用ncclCommInit Rank 并传入一个共享的唯一标识符来建立通信器 。通信任务完成后，应使用ncclCommDestroy（安全销毁，会等待挂起操作完成）或 **ncclCommAbort**（立即终止，用于错误恢复以避免死锁） 来释放资源。

### 2、**Collective Communication (集合通信)**

NCCL 提供五种集合操作：ncclAllReduce、ncclBroadcast、ncclReduce、ncclAllGather 和 ncclReduceScatter

### 3、**Point-to-Point Communication (点对点通信)**

NCCL 通过 neclSend 和 ncclRecv 支持点对点操作

### 4、**Group Calls (组调用)**

为聚合操作并减少开销，NCCL 提供了 ncclGroupStart 和 ncclGroupEnd 。这些函数包裹一系列 NCCL 调用，并将它们的执行延迟到组结束时 。分组操作可以包含多个 Send/Recv 调用（以模拟 SendRecv、All-to-One 等模式）或一组集合操作 。这种聚合方式能将所有分组操作作为单个 NCCL 启动的一部分一起执行，从而显著减少启动开销和延迟 

## B、**Launching Strategies**

NCCL 支持三种启动操作的执行模型，各有优劣 ：

* **One CPU process per GPU (每GPU一进程)**

此模型能更好地控制进程放置 。通过将每个GPU绑定到独立进程，相关的CPU代码可以被调度在本地NUMA域上，从而改善数据局部性并降低内存访问延迟 

* **One CPU thread per GPU (每GPU一线程)**：

当单个CPU进程通过多线程管理多个GPU时，可以实现高效的进程内内存共享 。这种设置允许跨rank直接访问内存，包括GPU缓冲区，从而减少通信时的内存拷贝开销

* **One CPU thread for multiple GPUs (单线程管理多GPU)**

虽然此模型存在顺序内核启动和并发性降低的问题，但它提供了简单性、最小的CPU开销和确定性执行的优点 。这使其适用于小规模部署或优先考虑实现简便性的原型环境 



## C、**Communication Channels**

* **硬件协同**

NCCL 通过 GPU、CPU 和网络接口（NIC）三个硬件组件协同通信。GPU 执行归约和数据移动，CPU 启动内核并管理主机端协调，NIC 在节点间传输数据包 。当只有一个流式多处理器（SM）处理 GPU 工作时，大消息会使其过载，导致其他 SM 未被充分利用，也无法饱和 NVLink 或 InfiniBand 等链路 

* **信道并行化**

为避免此瓶颈，NCCL 将每个集合操作细分为多个通信信道 。每个信道作为一个独立的 CUDA 块在各自的 SM 上运行，并且库会对输入缓冲区进行分区，使各信道能并行处理不相交的数据块 。这种细粒度的并行性提高了总吞吐量，尤其对于大数据负载 。将工作分散到多个信道还有助于在 NVLink 平台上的多个 NIC 之间平衡流量，因为每个信道可以独立地通过不同的 NIC 离开节点 。

* **信道数量的权衡**

然而，过度使用多信道可能对网络效率产生负面影响 。当每个信道的数据块大小变得小于 NIC 传输所用的 512 KiB FIFO 缓冲区大小时，代理线程会发送部分填充的缓冲区 。这种利用不足会降低 PCIe 和网络吞吐量，尤其是在为启用 ECMP 负载均衡而激活多个队列对（QP）时 。NCCL 通过启发式地为小消息减少nChannels 的数量来解决此问题（参考 [http://enqueue.cc](https://link.zhihu.com/?target=http%3A//enqueue.cc) 中的 calcP2pChunkSize 函数）。因此，选择最佳信道数是在 GPU 端并行性与网络利用效率之间的一种权衡 。

* **信道管理与拓扑**

信道管理在通信器级别进行协调 。在通信器初始化期间，NCCL 建立一组初始信道结构 。当调用集合操作时，NCCL 会动态选择算法和协议，其内部调优模型再根据所选策略、消息大小等因素决定使用多少个预建信道 。尽管早期版本允许用户通过环境变量（如NCCL_NTHREADS）影响信道行为，但现在已不鼓励手动调优，这些设置通常会被忽略，甚至可能导致不正确的行为 。分配给每个信道的逻辑通信拓扑直接决定了数据流。在**环形拓扑**中，每个 GPU 识别其前驱和后继 。在**树形拓扑**中，每个 GPU 跟踪其父节点和子节点 。为提高带宽利用率，NCCL 采用了**双二叉树**结构（double binary tree structure），该思想由 Hoefler 等人提出 (【Energy, Memory, and Runtime Tradeoffs for Implementing Collective Communication Operations, 2014, Journal of Supercomputing Frontiers and Innovations】与 【Full bandwidth broadcast, reduction and scan with only two trees, 2007, PVM/MPI’07】) 。这些拓扑在通信器初始化时建立，并被所有集合操作复用 。



## A3、方法和细节

### III、**COMMUNICATION PROTOCOLS**(通信协议)

![img](https://pic2.zhimg.com/v2-2431ee26cfc22f671ea17f36eaf874bd_r.jpg)

* **Simple Protocol**
  * **为高带宽设计**：Simple 协议旨在最大化带宽利用率，主要用于大消息传输 。它通过将数据分割成较大的数据块并在通信信道间分发，从而充分利用网络接口和 GPU 内存系统的高吞吐量
  * **基于内存屏障的同步**：为保证内存一致性，该协议使用内存屏障（memory fences）来强制执行正确的操作顺序和数据可见性 。接收方必须等待一个完整的数据块传输完毕后才能访问它 。虽然这种方法能确保正确性，但内存屏障引入了显著的开销，这成为小消息传输的限制因素，因为同步成本主导了总传输时间 。因此，Simple 协议虽然能为大消息实现接近峰值的带宽，但在处理小负载时延迟较高
* **LL (Low Latency) Protocol**
  * **为低延迟优化**：为解决 Simple 协议的延迟问题，NCCL 引入了 LL 协议，该协议专为小消息优化，因为小消息场景下带宽通常未被充分利用
  * **基于标志的轻量级同步**：LL 协议不依赖内存屏障，而是使用轻量级的基于标志的同步（flag-based synchronization）。一个小的标志（flag）与数据一同传输以表示其有效性，使接收方一旦数据可用就能立即处理，无需昂贵的内存屏障
  * **实现细节与性能权衡**：每次 LL 协议传输包含 4 字节数据和 4 字节标志，通过 8 字节原子操作一同发送 。这种方法显著降低了同步开销 。然而，它强制要求中间缓冲区必须位于主机内存中，以便 CPU 可以轮询标志并检测数据何时准备好通过 NIC 发送 。这是因为通过 PCIe 轮询 GPU 内存比访问 DRAM 慢得多，并且需要显式同步来确保数据在主机上的可见性 。此设计虽然实现了低延迟，但也
  * **阻止了 GPUDirect RDMA 的使用**，严重限制了带宽 。因此，LL 协议通常只能达到峰值带宽的 25-50% ，仅在延迟至关重要且带宽利用率次要的小数据量传输中被优先选择

* **LL128 Protocol**
  * **兼顾低延迟与高带宽**：LL128 协议在保持 LL 协议低延迟特性的同时，显著提高了带宽效率，尤其是在 NVLink 等高性能互连上 。它同样采用基于标志的同步来避免内存屏障，但以 128 字节为单位传输数据 。
  * **传输单元与性能**：在这 128 字节中，120 字节用于数据，8 字节保留给标志，使得该协议能够利用大约 95% 的峰值带宽 。在网络路径上，LL128 类似于 Simple 协议，发送方 GPU 会聚合一个相对较大的数据块，然后通知 CPU 发送 。虽然这种基于块的聚合限制了跨节点的流水线操作，但由于其较小的传输粒度，LL128 仍然能在节点内部实现细粒度的流水线 
  * **硬件依赖性**：LL128 对硬件有更严格的要求，它依赖于 128 字节的原子写操作，这些操作不能被内存系统或互连拆分或重排 。在由于 PCIe 限制或其他架构约束而无法保证此类操作的系统中，NCCL 会禁用 LL128 以避免数据损坏 。
* 动态选择机制：
* NCCL 在运行时根据用户设置（NCCL_PROTO）、集合算法以及内部性能启发式动态地在 Simple、LL 和 LL128 协议中进行选择 。若未明确指定，NCCL 会使用一个调优模型，该模型综合考虑系统拓扑、GPU 架构、消息大小和预定义的性能指标来选择最佳的算法-协议对 。该选择受到内存等资源的限制 。通常，LL/LL128 被用于小消息以降低延迟，而 Simple 被用于大消息以最大化吞吐量 。

