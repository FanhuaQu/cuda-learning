# cutlass pipeline

官方文档https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/pipeline.md

cuda pipeline https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/pipeline.html

## cuda同步方法概述

cuda编程提供的3个抽象

* 分层并行：分层单元比如block或者是cluster之间的同步
* 共享内存：同一分层单元内的线程可以通过共享内存通信
* 线程级别的同步

cuda针对不同粒度的同步，引入了不同级别的同步原语，包括

* block级别的同步，如`__syncthreads()`
* warp级别的同步，如`__syncthreads()`
* 线程级别的fence操作

hopper架构开始，引入了clusters的概念

* clusters：表示一组可以协调和共享数据的线程块
* 与此对应的，引入了clusters级别的同步以及clusters内部线程的同步



## cutlass 对hopper功能的抽象

* clusters级别的同步和查询
* 新的barrier指令用于clusters内的线程同步

### Asynchronous pipelines

* ##### Producer acquire 

  `Producer acquire()`方法是由生产者在执行特定的pipeline阶段(例如数据复制、写入)之前调用的，是一种同步手段，可以理解为**等待需要的资源已经准备好**了，例如**等待共享内存可写**。

  `Producer acquire()`会阻塞线程，如果 **某一 pipeline stage 没有被 consumer release（释放）**，那么调用 `producer_acquire` 会**阻塞执行**，直到可以继续生产

  如果 pipeline **一开始是空的**，意味着 **生产者不需要等待消费者释放**，可以直接执行，此时调用 `producer_acquire` 会**立即成功**，不会阻塞。

  如果你确定 pipeline 初始是空的，有两种方式可以正确初始化：第一次迭代时**跳过调用 `producer_acquire`**；使用 `make_producer_start_state()` 方法来初始化 pipeline，确保 `producer_acquire` 会在初始时成功

* **Producer commit**(某些情况下是非必须的)

  `producer_commit()` 是**生产者线程用来通知消费者线程**某个 pipeline 阶段已经完成的标志性操作。

  `producer_commit()` 是**非阻塞的**，也就是说它**不会等待任何其他线程**，直接发出通知即可

* **Consumer wait**

  `consumer_wait()` 是**消费者线程**调用的，目的：在**读取（消费）某个 pipeline stage 中的数据之前**，先确保这个 stage 的数据已经被生产者线程写好。这是一个**阻塞操作**。等待的信号就是来自上面的`producer_commit()` ，对于使用TMA时，结束的时候会自动完成commit

* **Consumer release**

  该方法的作用是告诉 **正在等待的生产者线程（producer threads）**，这个流水线阶段的数据已经被消费完了，**可以重用这一阶段所占用的资源**（比如 shared memory buffer）.这是一个 **非阻塞操作**。调用后会立即继续执行，而不会停下来等待其他线程或事件。上面的acquire会等待消费者发出release信号

  

### 使用pipeline实现生产者消费者之间的同步

```c++
// 生产者线程
pipeline.producer_acquire(stage_id);  // 等待 pipeline 阶段空出来
...  // 执行 TMA 或 shared memory copy
pipeline.producer_commit(stage_id);   // 通知消费者该阶段已准备好

// 消费者线程
pipeline.consumer_wait(stage_id);     // 等待对应阶段的数据可用
...  // 使用该数据进行计算
pipeline.consumer_release(stage_id);  // 通知 pipeline 这个阶段可以被复用
```

四阶段流水线的例子

```c++
// 4-stage Pipeline
static constexpr int NumStages = 4;
using MainloopPipeline = typename cutlass::PipelineAsync<NumStages>;
using PipelineState = typename cutlass::PipelineState<NumStages>;	// 用于记录当前在哪个阶段

// 2 producer threads and 1 consumer thread 
// 初始化流水线，设置两个生产者线程和一个消费者线程
// shared_storage.storage是共享内存中用于同步和数据交换的缓冲区
typename MainloopPipeline::Params params;
params.producer_arv_count = 2;
params.consumer_arv_count = 1;
MainloopPipeline pipeline(shared_storage.storage, params);
  
// Producer threads
if (thread_idx == 0 or thread_idx == 1) {
    // make_producer_start_state初始化，因为这个时候是空的
  PipelineState smem_pipe_write = cutlass::make_producer_start_state<MainloopPipeline>();
  for ( ; iter > 0; --iter) {
    pipeline.producer_acquire(smem_pipe_write);

    // Producer ops
    // If any memory operations are involved, then we also need
    // to guarantee that writes are completed and visible to consumer(s).

    pipeline.producer_commit(smem_pipe_write);
    ++smem_pipe_write;		// 移动到下一个stage
  }
}
else if (thread_idx == 2) {
  PipelineState smem_pipe_read;
  for (; iter > 0; --iter) {
    pipeline.consumer_wait(smem_pipe_read);

    // Consumer ops

    pipeline.consumer_release(smem_pipe_read);
    ++smem_pipe_read;
  }
}
```

再看一个例子：

* Prologue，对于一个流水线而言，必须有个开始，才能流动起来

  为了让 consumer 在最开始就有数据可读，**producer 必须先把前几级填好**，这就是“Prologue”

  **目的是当流水线能够正常启动，而不会死锁**，假如开始就producer_acquire，这个时候消费者也在consumer_wait，这两个阻塞操作就会导致死锁

```c++
template <class ClusterShape, uint32_t NumStages>     // NumStages流水线阶段数
__global__ static 
void pipeline_async_basic_device(uint32_t const num_iterations)
{

  extern __shared__ char shared_memory[];
  using MainloopPipeline = typename cutlass::PipelineAsync<NumStages>;
  using PipelineState = typename cutlass::PipelineState<NumStages>;

    // 所有流水线状态和标志都存储在共享内存中（PipelineAsync::SharedStorage）
  using SharedStorage = SharedStorage<NumStages>;
  SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(shared_memory);

    // 每个warp的索引
  int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
  int lane_predicate = cute::elect_one_sync();    // 选出一个线程
  dim3 block_id_in_cluster = cute::block_id_in_cluster();
  
  // This example showcases 2 producer 1 consumer example 
  typename MainloopPipeline::Params params;
  params.producer_arv_count = 2;
  params.consumer_arv_count = 1;
  MainloopPipeline pipeline(shared_storage.storage, params);    // 初始化

  // Ensure All CTAs in Cluster have completed init before issuing commits
  cute::cluster_arrive_relaxed();  // 保证集群内的CTA初始化完成
  cute::cluster_wait();
  __syncthreads();


  if (lane_predicate) {
    // Producer Warps
    if (warp_idx==0 || warp_idx==1) {

      PipelineState smem_pipe_write = cutlass::make_producer_start_state<MainloopPipeline>();
      int prologue_iterations = min(NumStages, num_iterations);
      // 预填充阶段
      for ( int i = 0; i < prologue_iterations; ++i) {
        // Can also specify stage to commit directly
        pipeline.producer_commit(smem_pipe_write);   // why? Prologue：先填满流水线前几级
        ++smem_pipe_write;
      }

      int mainloop_iterations = num_iterations - prologue_iterations;

      for ( ; mainloop_iterations > 0; --mainloop_iterations) {
        pipeline.producer_acquire(smem_pipe_write);
        pipeline.producer_commit(smem_pipe_write);
        ++smem_pipe_write;
      }
    }
    else {
      PipelineState smem_pipe_read;
      for (int iter=0 ; iter < num_iterations; ++iter) {
        pipeline.consumer_wait(smem_pipe_read);
        pipeline.consumer_release(smem_pipe_read);
        ++smem_pipe_read;
      }
    }
  }

  // To make sure remote SMEM doesn't get destroyed
  cute::cluster_arrive();  
  cute::cluster_wait();  
}
```



# 精读代码

文件路径为`include/cutlass/pipeline/pipeline.hpp`，实际路径是`include/cutlass/pipeline/sm90_pipeline.hpp`。所以只有sm_90的设备才支持使用cutlass pipeline

## 1、流水线状态管理`struct PipelineState`

用于跟踪流水线状态

### 1.1、数据成员

* static constexpr uint32_t Stages = Stages_； 流水线阶段数
* int index_ = 0; 当前阶段索引，0 ~ Stages-1
* uint32_t phase_ = 0; 相位标志，用于同步
* uint32_t count_ = 0; 总的推进次数



### 1.2、方法

* 重写了operator++

```c++
  CUTLASS_DEVICE
  void operator++() {
    if constexpr (Stages > 0) {
      ++index_;
      ++count_;
      if (index_ == Stages) {		// 环形回绕
        index_ = 0;
        phase_ ^= 1;		// 相位翻转
      }
    }
  }
```

* `PipelineState& advance(uint32_t num_iterations)`推进指定步数
* `PipelineState<Pipeline::Stages> make_producer_start_state()`

```c++
// 初始化
// 生产者从第0阶段开始工作（环形缓冲区的起始索引）
// 生产者初始相位设置为1（与消费者初始相位相反）
// 初始化流水线推进计数器为0
CUTLASS_DEVICE
PipelineState<Pipeline::Stages> make_producer_start_state() {
  // Producer starts with an opposite phase as the buffers are initially empty
  constexpr int InitialProducerStage = 0;
  constexpr uint32_t InitialProducerPhase = 1;
  constexpr uint32_t InitialProducerCount = 0;
  return {InitialProducerStage, InitialProducerPhase, InitialProducerCount};
}
```

## 2、TMA异步流水线 `PipelineTmaAsync<Stages_>`

**适用场景**：Tensor Memory Access (TMA) 加载的异步生产-消费模型

**数据成员**

```c++
  using FullBarrier = cutlass::arch::ClusterTransactionBarrier;
  using EmptyBarrier = cutlass::arch::ClusterBarrier;
  using ProducerBarrierType = FullBarrier::ValueType;
  using ConsumerBarrierType = EmptyBarrier::ValueType;
  static constexpr uint32_t Stages = Stages_;
  using PipelineState = cutlass::PipelineState<Stages>;

	// 线程角色
  enum class ThreadCategory {
    NonParticipant,
    Producer,
    Consumer,
    ProducerConsumer
  };

  struct Params {
    uint32_t transaction_bytes = 0;				// TMA拷贝的字节数
    ThreadCategory role = ThreadCategory::NonParticipant;	// 线程角色，生产者、消费者、都是
    uint32_t is_leader = 0;
    uint32_t num_consumers = 0; // Number of consumer threads
    uint32_t num_producers = 1; // Number of producer threads
  };

  uint32_t dst_blockid_ = 0;	// 目标块id,标识当前线程需要通知的目标Block在集群中的逻辑ID
  uint32_t is_signaling_thread_ = 0;	// 标记当前线程是否为信号发送责任线程
  FullBarrier *full_barrier_ptr_ = nullptr;	// 管理生产者→消费者的同步（数据就绪状态
  EmptyBarrier *empty_barrier_ptr_ = nullptr; // 管理消费者→生产者的同步（缓冲区空闲状态）
  Params params_;	// 流水线配置参数
```

**方法**

* init_barriers()

初始化同步屏障，类似于之前写的双缓冲区的生产者消费者同步方式

```c++
init_barriers(SharedStorage& storage, Params params, ClusterShape cluster_shape) {
int warp_idx = canonical_warp_idx_sync();
bool is_initializing_warp = (warp_idx == 0);	// 第一个warp执行
if (is_initializing_warp) {
  // Barrier FULL and EMPTY init
  uint32_t const producer_arv_cnt = params.num_producers;		// 生产者线程数量
  uint32_t const num_consumer_warpgroups_per_cluster = params.num_consumers / NumThreadsPerWarpGroup;
  uint32_t multicast_consumer_arrival_count = params.num_consumers; // If cluster_size is 1
  if (cute::size(cluster_shape) > 1) {
    multicast_consumer_arrival_count = (cute::size<0>(cluster_shape) + cute::size<1>(cluster_shape) - 1) *
          num_consumer_warpgroups_per_cluster;
  }

  cutlass::arch::detail::initialize_barrier_array_pair_aligned<decltype(storage.full_barrier_), decltype(storage.empty_barrier_), Stages>(
      storage.full_barrier_, storage.empty_barrier_, producer_arv_cnt, multicast_consumer_arrival_count);
}
cutlass::arch::fence_barrier_init();	// 通过fence_barrier_init()确保所有线程都能看到初始化完成的状态。
}
```

* 构造函数 不是太理解

```c++

  template<class ClusterShape, class InitBarriers, class InitMasks>
  CUTLASS_DEVICE
  PipelineTmaAsync(SharedStorage& storage, Params params, ClusterShape cluster_shape, InitBarriers = {}, InitMasks = {})
      : params_(params)
      , full_barrier_ptr_(&storage.full_barrier_[0])				// 共享内存barrier
      , empty_barrier_ptr_(&storage.empty_barrier_[0]) {

    int warp_idx = canonical_warp_idx_sync();
    int thread_idx = threadIdx.x;
    int lane_predicate = cute::elect_one_sync();

    static_assert(cute::is_same_v<InitBarriers, cute::true_type> || cute::is_same_v<InitBarriers, cute::false_type>);
    static_assert(cute::is_same_v<InitMasks, cute::true_type> || cute::is_same_v<InitMasks, cute::false_type>);
    if constexpr (cute::is_same_v<InitBarriers, cute::true_type>) {		// init barrier
      init_barriers(storage, params_, cluster_shape);
    }

     // 初始化 mask（决定哪个线程发送 empty arrive）
     // 分配 empty_arrive 任务给不同线程，防止所有线程都同时发信号，造成冗余/冲突。
    if constexpr (cute::is_same_v<InitMasks, cute::true_type>) {
      // Logic to optimally schedule Empty Arrives
      // Goal : To divide SYNCS Empty Arrival duty equally amongst the Warp-Group (128 threads)
      dim3 block_id = cute::block_id_in_cluster();
      auto cluster_size = cute::size(cluster_shape);

      if (cluster_size == 1) {		// 只有一个 block，直接本地完成同步信号
        is_signaling_thread_ = true;
        dst_blockid_ = 0;
      }
      else {
        // STEP 1 : Use Cute Layout function to generate an optimal dst block-id (0-15)
        if (params_.num_consumers % NumThreadsPerWarpGroup == 0) {	// 每个warp的第一个线程去发出信号
          auto [is_signaling_thread, dst_blockid] = detail::spread_arrivals_to_warpgroup(thread_idx % NumThreadsPerWarpGroup, warp_idx);
          is_signaling_thread_ = is_signaling_thread;
          dst_blockid_ = dst_blockid;
        }
        else if (params_.num_consumers == 32) {		// 不会跟上面重复吗？
          auto [is_signaling_thread, dst_blockid] = detail::spread_arrivals_to_warp(thread_idx % 32);
          is_signaling_thread_ = is_signaling_thread;
          dst_blockid_ = dst_blockid;
        }
        else {
          is_signaling_thread_ = 0;
          #ifndef NDEBUG
            asm volatile ("brkpt;\n" ::);
          #endif
        }

        // STEP 2: Find if this dst block-id needs an arrival for this problem
        is_signaling_thread_ &= dst_blockid_ < cluster_size;
        is_signaling_thread_ &= is_same_row_or_col(dst_blockid_, block_id, cluster_shape);
      }
    }
  }

```

下面是producer APIs

包括两对函数 `producer_try_acquire and producer_acquire,`以及`consumer_try_wait and consumer_wait`

**try和finalize的关系**：try是非阻塞的，尝试“等一小会”，看看barrier有没有翻转。不带try的是阻塞的，阻塞等待直到barrier翻转了才继续执行。

**理解这种设计**：GPU 是 SIMT 架构，大量线程同步执行，如果直接使用阻塞（blocking）等待，有些线程可能会白等。所以先用 `try_*` 方法“试探性等一下”，如果 barrier 翻了，直接继续，如果没翻，就把 “try 返回的 token” 传给 `acquire` 或 `wait`，再进行真正的等待。

例如

```c++
auto token = pipeline.producer_try_acquire(smem_pipe_write);  // 尝试获取空位
pipeline.producer_acquire(smem_pipe_write, token);  // 如果没获取到，等 barrier 翻；否则立刻通过
```

看具体的实现：

**非阻塞地尝试获取对某个流水线阶段的写入权限，如果不能立即获取，则返回一个“等待中”的 token**

```c++
  ProducerToken producer_try_acquire(PipelineState state, uint32_t skip_wait = false) {
    return producer_try_acquire(state.index(), state.phase(), skip_wait);
  }
  ProducerToken producer_try_acquire(uint32_t stage, uint32_t phase, uint32_t skip_wait) {
    detail::pipeline_check_is_producer(params_.role);
    if (skip_wait) {		// 在Prologue阶段，就是可以直接跳过的(一定是空的可写)
      return {BarrierStatus::WaitDone};
    }
      // 尝试等待 empty_barrier 的翻转（状态从 not empty → empty）非阻塞的
    bool barrier_status = empty_barrier_ptr_[stage].try_wait(phase);
      // 返回一个 ProducerToken，其内部状态是 WaitDone（可写）或 WaitNotDone（还不能写）
      // 取决于 barrier_status 的值。
      // 这个 token 会被传入 producer_acquire()，判断是否需要进一步阻塞等待
    return {static_cast<BarrierStatus>(barrier_status)};
  }
```

**producer_acquire**

```c++
    CUTLASS_DEVICE
    void producer_acquire(PipelineState state, ProducerToken barrier_token = {BarrierStatus::WaitAgain}) {
        producer_acquire(state.index(), state.phase(), barrier_token);
    }

  CUTLASS_DEVICE
  void producer_acquire(uint32_t stage, uint32_t phase, ProducerToken barrier_token) {
    detail::pipeline_check_is_producer(params_.role);		// check role
    if (barrier_token != BarrierStatus::WaitDone) {			
        // 如果前面try_acquire的结果是还没翻转，就在这里阻塞等待
      empty_barrier_ptr_[stage].wait(phase);
    }
	// 这个时候producer是可以写数据的， is_leader 表示当前线程是负责发起数据搬运的“领导线程”。
    // 调用 arrive_and_expect_tx() 表示：“我这边准备好要写数据到 shared memory 了，这里会产生 N 字节的数据流量”
    // 对TMA来说是必要的
    if (params_.is_leader) {
      full_barrier_ptr_[stage].arrive_and_expect_tx(params_.transaction_bytes);
    }
    #ifndef NDEBUG
    if (params_.role == ThreadCategory::Consumer || params_.role == ThreadCategory::NonParticipant) {
      asm volatile ("brkpt;\n" ::);
    }

    // Most likely you have elected more than one leader
    if (params_.is_leader && (threadIdx.x % 32 != 0)) {
      asm volatile ("brkpt;\n" ::);
    }
    #endif
  }

```

**producer_commit** 只在单元测试中使用

```c++
  CUTLASS_DEVICE
  void producer_commit(uint32_t stage, uint32_t bytes) {
    // Below code is used only for unit-testing (in the absence of TMA commit)
    #if CUTLASS_UNIT_TEST_PIPELINE
      if (params_.is_leader) {
        // STEP 1 : Commit to self
        // 相当于 arrive_and_expect_tx(bytes)
        full_barrier_ptr_[stage].complete_transaction(bytes); 

        // STEP 2 : Commit to other blocks in our cluster
        auto cluster_shape = cute::cluster_shape();
        Layout block_layout_in_cluster = make_layout(cluster_shape);
        dim3 local_block_id = cute::block_id_in_cluster();

        CUTLASS_PRAGMA_UNROLL
        for(int n = 0; n < size<1>(block_layout_in_cluster); ++n) {
          uint32_t dst_block_id = block_layout_in_cluster(local_block_id.x,n,Int<0>{});
          full_barrier_ptr_[stage].complete_transaction(dst_block_id, bytes, n!=local_block_id.y);
        }

        CUTLASS_PRAGMA_UNROLL
        for(int m = 0; m < size<0>(block_layout_in_cluster); ++m) {
          uint32_t dst_block_id = block_layout_in_cluster(m,local_block_id.y,Int<0>{});
          full_barrier_ptr_[stage].complete_transaction(dst_block_id, bytes, m!=local_block_id.x);
        }
      }
    #endif
  }
```

**consumer_try_wait**

类似的，只是尝试等待barrier为full

```c++
  CUTLASS_DEVICE
  ConsumerToken consumer_try_wait(uint32_t stage, uint32_t phase, uint32_t skip_wait) {
    detail::pipeline_check_is_consumer(params_.role);
    if (skip_wait) {
      return {BarrierStatus::WaitDone};
    }
    bool barrier_status = full_barrier_ptr_[stage].try_wait(phase);
    return {static_cast<BarrierStatus>(barrier_status)};
  }
```

**consumer_wait**

```c++
  // Wait for producer to commit transactions (done by TMA)
  CUTLASS_DEVICE
  void consumer_wait(uint32_t stage, uint32_t phase) {
    detail::pipeline_check_is_consumer(params_.role);
    full_barrier_ptr_[stage].wait(phase);		// 调用wait方法
  }

  // Wait for producer to commit transactions (done by TMA)
  CUTLASS_DEVICE
  void consumer_wait(uint32_t stage, uint32_t phase, ConsumerToken barrier_token) {
    detail::pipeline_check_is_consumer(params_.role);
    if (barrier_token == BarrierStatus::WaitAgain) {
      full_barrier_ptr_[stage].wait(phase);
    }
  }
```

**consumer_release**

```c++
  // Consumer signalling Producer of completion
  // Ensures all blocks in the Same Row and Column get notifed.
  CUTLASS_DEVICE
  void consumer_release(uint32_t stage, uint32_t skip = false) {
    detail::pipeline_check_is_consumer(params_.role);
      // 这里的arriva发出信号
      // 让“某个线程”去通知某个 stage 的 empty barrier：“该解锁啦，consumer 已经处理完了。”
    empty_barrier_ptr_[stage].arrive(dst_blockid_, is_signaling_thread_ & (!skip));
    #ifndef NDEBUG
    if (params_.role == ThreadCategory::Producer || params_.role == ThreadCategory::NonParticipant) {
      asm volatile ("brkpt;\n" ::);
    }
    #endif
  }
```

## 针对TMA设计的极简异步流水线

这个流水线是 **单向的、只有 Producer，没有 Consumer**，原因是：数据最终是直接写入 gmem，不需要软件 consumer。**TMA unit 硬件就是 consumer。**

`PipelineTmaStore` 是 CUTLASS 针对 TMA store 定制的 **单向 producer-only pipeline**，用来限制 store 并发深度，保证安全、高效的数据写入。不是太理解

```c++
template <
  int Stages_,
  // The number of committed TMA store batches that can be in flight upon return of producer acquire
  int UnacquiredStages_ = Stages_-1
>
class PipelineTmaStore {
public:
  static constexpr uint32_t Stages = Stages_;
  static_assert(Stages_ > 0);
  static_assert(UnacquiredStages_ >= 0);
  static constexpr uint32_t UnacquiredStages = static_cast<uint32_t>(UnacquiredStages_);
  using PipelineState = cutlass::PipelineState<Stages>;

  struct Params {
    bool always_wait = false;
  };

  CUTLASS_DEVICE
  PipelineTmaStore(Params params = {}) : params_(params) {}

  ////////////////////
  // Producer APIs
  ////////////////////
  // Wait for the least recently committed batch of TMA stores to complete
  CUTLASS_DEVICE
  void producer_acquire(PipelineState state) {
    producer_acquire(state.index(), state.count());
  }

  // Commit the most recently issued batch of TMA stores
  CUTLASS_DEVICE
  void producer_commit(PipelineState state) {
    producer_commit(state.index(), state.count());
  }

  // Wait for all TMA stores to complete
  CUTLASS_DEVICE
  void producer_tail([[maybe_unused]] PipelineState state) {
    tma_store_wait<0>();
  }

private:
  Params params_;

  // Wait for the least recently committed batch of TMA stores to complete
  // or until at most UnacquiredStages TMA store batches are in-flight (if specified)
  CUTLASS_DEVICE
  void producer_acquire([[maybe_unused]] uint32_t stage, uint32_t count) {
    // 如果已经有太多 store 在 flight 中（大于 UnacquiredStages），那就暂停，等待最早的那批写完。
    if (params_.always_wait || count > UnacquiredStages) {
      tma_store_wait<UnacquiredStages>();
    }
  }

  // Commit the most recently issued batch of TMA stores
  // 相当于发出一个 TMA store 批次请求，让 hardware TMA unit 开始工作
  CUTLASS_DEVICE
  void producer_commit([[maybe_unused]] uint32_t stage, [[maybe_unused]] uint32_t count) {
    tma_store_arrive();
  }
};
```

## 生产者-消费者 pipeline

高级流水线机制的实现，设计目的是协调 **Producer**（例如写入共享内存或 TMA 存储）和 **Consumer**（例如读取并执行计算）之间的异步协作

```
[Producer]      |        [Consumer]
----------------+------------------------------
try_acquire     |        
acquire         |        
expect_tx       |        
commit          | ->     try_wait / test_wait
                | ->     wait
                | <-     release
                |        
[Next stage]    |

```

```c++
template <int Stages_>
class PipelineTransactionAsync {
public:
  // ClusterTransactionBarrier 由 Producer 设置、Consumer 等待（即数据准备好通知 Consumer）
  using FullBarrier = cutlass::arch::ClusterTransactionBarrier;
  // ClusterBarrier 由 Consumer 设置、Producer 等待
  using EmptyBarrier = cutlass::arch::ClusterBarrier;
  using ProducerBarrierType = FullBarrier::ValueType;
  using ConsumerBarrierType = EmptyBarrier::ValueType;
  static constexpr uint32_t Stages = Stages_;
  using PipelineState = cutlass::PipelineState<Stages>;

  // barrier
  struct SharedStorage {
    cute::array<FullBarrier, Stages> full_barrier_;
    cute::array<EmptyBarrier, Stages> empty_barrier_;
  };

  // role
  enum class ThreadCategory {
    NonParticipant,
    Producer,
    Consumer,
    ProducerConsumer
  };
  // 
  struct Params {
      // 单次 producer transaction 要写入的字节数，用于 expect_transaction() 通知
    ThreadCategory role = ThreadCategory::NonParticipant;
    uint32_t transaction_bytes = 0;
    uint32_t producer_arv_count = 1;	// 表示一个阶段内有多少个 Producer 会 arrive()
    uint32_t consumer_arv_count = 1;
      // 当前线程块（block）在 cluster 中的逻辑 ID，用于跨块 barrier 协调
    uint32_t dst_blockid = cute::block_rank_in_cluster();
  };

    // 初始化同步barrier数组
  static
  CUTLASS_DEVICE
  void
  init_barriers(SharedStorage& storage, Params const& params) {
      // 从smem中获取barrier数组指针
      // 每个 pipeline stage 有一对 barrier：full（写完）、empty（读完）
      // 缓冲区数量应该就对应的阶段数
    FullBarrier *full_barrier_ptr = storage.full_barrier_.data();
    EmptyBarrier *empty_barrier_ptr = storage.empty_barrier_.data();
    int warp_idx = canonical_warp_idx_sync();
    bool is_initializing_warp = (warp_idx == 0);
      // 第一个warp负责初始化
    if (is_initializing_warp) {
      // Barrier FULL and EMPTY init， 做的事情包括：配置producer comsumer数量，内存对齐
                  								cutlass::arch::detail::initialize_barrier_array_pair_aligned<decltype(full_barrier_ptr), decltype(empty_barrier_ptr), Stages>(
          full_barrier_ptr, empty_barrier_ptr, params.producer_arv_count, params.consumer_arv_count);
    }
    cutlass::arch::fence_barrier_init();
  }

  // Constructor
  template<class InitBarriers>
  CUTLASS_DEVICE
  PipelineTransactionAsync(SharedStorage& storage, Params const& params, InitBarriers = cute::true_type{})
    : params_(params)
    , full_barrier_ptr_(storage.full_barrier_.data())
    , empty_barrier_ptr_(storage.empty_barrier_.data()) {

    int warp_idx = canonical_warp_idx_sync();
    int lane_predicate = cute::elect_one_sync();

    static_assert(cute::is_same_v<InitBarriers, cute::true_type> || cute::is_same_v<InitBarriers, cute::false_type>);

    if constexpr (cute::is_same_v<InitBarriers, cute::true_type>) {
      init_barriers(storage, params);
    }

  }

  // Constructor
  CUTLASS_DEVICE
  PipelineTransactionAsync(SharedStorage& storage, Params const& params) :
    PipelineTransactionAsync(storage, params, cute::true_type{}) { }

  CUTLASS_DEVICE
  ProducerToken producer_try_acquire(PipelineState state, uint32_t skip_wait = false) {
    return producer_try_acquire(state.index(), state.phase(), skip_wait);
  }

  CUTLASS_DEVICE
  void producer_acquire(PipelineState state, ProducerToken barrier_token = {BarrierStatus::WaitAgain}) {
    producer_acquire(state.index(), state.phase(), barrier_token);
  }

  // Perform an expect-tx operation on the stage's full barrier. Must be called by 1 thread
  CUTLASS_DEVICE
  void producer_expect_transaction(PipelineState state) {
    producer_expect_transaction(state.index());
  }

  CUTLASS_DEVICE
  void producer_commit(PipelineState state) {
    producer_commit(state.index());
  }

  // Prevents early exit of producer blocks in Cluster.
  // This should be called once before kernel exits.
  // 等待该阶段的 数据被 consumer 消费完（即 barrier 被 consumer release）
  // 假如block 退出，smem被销毁了，这样TMA没完成就会发生数据错乱
  CUTLASS_DEVICE
  void producer_tail(PipelineState state) {
    for (int count = 0; count < Stages; ++count) {
      producer_acquire(state);		// 等待消费者完成，确保退出的时候消费者已经用完数据
      ++state;
    }
  }

  CUTLASS_DEVICE
  ProducerBarrierType* producer_get_barrier(PipelineState state) {
    return producer_get_barrier(state.index());
  }

  ////////////////////
  // Consumer APIs
  ////////////////////
  CUTLASS_DEVICE
  ConsumerToken consumer_try_wait(PipelineState state, uint32_t skip_wait = false) {
    return consumer_try_wait(state.index(), state.phase(), skip_wait);
  }

  CUTLASS_DEVICE
  ConsumerToken consumer_test_wait(PipelineState state, uint32_t skip_wait = false) {
    return consumer_test_wait(state.index(), state.phase(), skip_wait);
  }

  CUTLASS_DEVICE
  void consumer_wait(PipelineState state, ConsumerToken barrier_token = {BarrierStatus::WaitAgain}) {
    consumer_wait(state.index(), state.phase(), barrier_token);
  }

  CUTLASS_DEVICE
  void consumer_release(PipelineState state) {
    consumer_release(state.index());
  }

private:
  FullBarrier *full_barrier_ptr_ = nullptr;
  EmptyBarrier *empty_barrier_ptr_ = nullptr;
  Params params_;

  CUTLASS_DEVICE
  ProducerToken producer_try_acquire(uint32_t stage, uint32_t phase, uint32_t skip_wait) {
    detail::pipeline_check_is_producer(params_.role);		// check role
    if (skip_wait) {
      return {BarrierStatus::WaitDone};				
    }
    bool barrier_status = empty_barrier_ptr_[stage].try_wait(phase);
    return {static_cast<BarrierStatus>(barrier_status)};
  }

  CUTLASS_DEVICE
  void producer_acquire(uint32_t stage, uint32_t phase, ProducerToken barrier_token) {
    detail::pipeline_check_is_producer(params_.role);
    if (barrier_token == BarrierStatus::WaitAgain) {
      empty_barrier_ptr_[stage].wait(phase);	// wait untill producer commit
    }
  }

  // Perform an expect-tx operation on the stage's full barrier. Must be called by 1 thread
  // expect_transaction(bytes) —— 表示 “我写了 X 个字节的东西，请准备接收”
  // arrive() —— 表示 “我写完了，你可以来消费了”
  /*
    pipeline.producer_expect_transaction(state); // 标记我要写
    // 写共享内存
    pipeline.producer_commit(state);            // 通知 TMA：可以读了
    假如漏了 expect_transaction()，那么即使你调用 arrive()，TMA 也不知道你写了多少数据
  */
  CUTLASS_DEVICE
  void producer_expect_transaction(uint32_t stage) {
    detail::pipeline_check_is_producer(params_.role);
    full_barrier_ptr_[stage].expect_transaction(params_.transaction_bytes);
  }

  CUTLASS_DEVICE
  void producer_commit(uint32_t stage) {
    detail::pipeline_check_is_producer(params_.role);
    full_barrier_ptr_[stage].arrive(params_.dst_blockid);		// 这里调用arrive
  }

  CUTLASS_DEVICE
  ProducerBarrierType* producer_get_barrier(uint32_t stage) {
    return reinterpret_cast<ProducerBarrierType*>(&full_barrier_ptr_[stage]);
  }

  CUTLASS_DEVICE
  ConsumerToken consumer_try_wait(uint32_t stage, uint32_t phase, uint32_t skip_wait) {
    detail::pipeline_check_is_consumer(params_.role);
    if (skip_wait) {
      return {BarrierStatus::WaitDone};
    }
    bool barrier_status = full_barrier_ptr_[stage].try_wait(phase);
    return {static_cast<BarrierStatus>(barrier_status)};
  }

  CUTLASS_DEVICE
  ConsumerToken consumer_test_wait(uint32_t stage, uint32_t phase, uint32_t skip_wait) {
    detail::pipeline_check_is_consumer(params_.role);
    if (skip_wait) {
      return {BarrierStatus::WaitDone};
    }
    bool barrier_status = full_barrier_ptr_[stage].test_wait(phase);
    return {static_cast<BarrierStatus>(barrier_status)};
  }

  CUTLASS_DEVICE
  void consumer_wait(uint32_t stage, uint32_t phase, ConsumerToken barrier_token) {
    detail::pipeline_check_is_consumer(params_.role);
    if (barrier_token == BarrierStatus::WaitAgain) {
      full_barrier_ptr_[stage].wait(phase);
    }
  }

  CUTLASS_DEVICE
  void consumer_release(uint32_t stage, uint32_t skip = false) {
    detail::pipeline_check_is_consumer(params_.role);
    empty_barrier_ptr_[stage].arrive(params_.dst_blockid, (not skip));
  }
};
```

## PipelineAsync通用封装

```c++
namespace PipelineDetail {
  template<int Stages>
  using PipelineAsyncPipelineState = cutlass::PipelineState<Stages>;

  template<int Stages>
  struct PipelineAsyncSharedStorage {
    using FullBarrier = cutlass::arch::ClusterBarrier;
    using EmptyBarrier = cutlass::arch::ClusterBarrier;
    FullBarrier full_barrier_[Stages];		// barrier，每个阶段都有full and empty barrier
    EmptyBarrier empty_barrier_[Stages];
  };
};


template <int Stages_>
class PipelineAsync {
public:
  static constexpr uint32_t Stages = Stages_;
  using SharedStorage = PipelineDetail::PipelineAsyncSharedStorage<Stages>;
  using FullBarrier = typename SharedStorage::FullBarrier;
  using EmptyBarrier = typename SharedStorage::EmptyBarrier;
  using ProducerBarrierType = typename FullBarrier::ValueType;
  using ConsumerBarrierType = typename EmptyBarrier::ValueType;
  using PipelineState = PipelineDetail::PipelineAsyncPipelineState<Stages>;

  // role
  enum class ThreadCategory {
    NonParticipant,
    Producer,
    Consumer,
    ProducerConsumer
  };
 // 
  struct Params {
    ThreadCategory role = ThreadCategory::NonParticipant;
    uint32_t producer_arv_count = 1; // 这里配合上面barrier，需要发出full barrier的warp数量
    uint32_t consumer_arv_count = 1;
    uint32_t dst_blockid = cute::block_rank_in_cluster();
  };

  static
  CUTLASS_DEVICE
  void
  init_barriers(SharedStorage& storage, Params params) {
    int warp_idx = canonical_warp_idx_sync();
    bool is_initializing_warp = (warp_idx == 0);
    if (is_initializing_warp) {
      // Barrier FULL and EMPTY init
      cutlass::arch::detail::initialize_barrier_array_pair_aligned<decltype(storage.full_barrier_), decltype(storage.empty_barrier_), Stages>(
          storage.full_barrier_, storage.empty_barrier_, params.producer_arv_count, params.consumer_arv_count);
    }
    cutlass::arch::fence_barrier_init();
  }

  // construct
  template<class InitBarriers>
  CUTLASS_DEVICE
  PipelineAsync(
    SharedStorage& storage,
    Params const& params,
    InitBarriers = {}) :
      params_(params),
      full_barrier_ptr_(&storage.full_barrier_[0]),
      empty_barrier_ptr_(&storage.empty_barrier_[0]) {

    static_assert(cute::is_same_v<InitBarriers, cute::true_type> || cute::is_same_v<InitBarriers, cute::false_type>);
    if constexpr (cute::is_same_v<InitBarriers, cute::true_type>) {
      init_barriers(storage, params_);
    }
  }

  CUTLASS_DEVICE
  PipelineAsync(
    SharedStorage& storage,
    Params const& params) :
      PipelineAsync(storage, params, cute::true_type{}) { }

  // Default assumption when only storage is passed is :
  // => single producer, single consumer & they are in the same block (within the Cluster)
  CUTLASS_DEVICE
  PipelineAsync(SharedStorage& storage)
    : PipelineAsync(storage, {}, cute::true_type{}) {}

  ////////////////////
  // Producer APIs
  ////////////////////
  // Four member functions are always used in pairs:
  //
  // * producer_try_acquire and producer_acquire, and
  // * consumer_try_wait and consumer_wait.
  //
  // The two functions with "try" in their names are called "try" functions,
  // and the other two are conceptually "finalize" functions.
  // The "try" function in each pair starts the process of waiting on the barrier to flip.
  // It opportunistically waits for an implementation-dependent timeout.
  // Whether or not the barrier has flipped yet, the try function will return a token.
  // If the token indicates that the barrier has not flipped,
  // then the token must be passed into the corresponding "finalize" function.
  // The finalize function will then block until the barrier has flipped.
  // If the token indicates that the barrier _has_ flipped,
  // then it is still correct to pass it into the finalize function.
  // The finalize function will return immediately in that case.
  CUTLASS_DEVICE
  ProducerToken producer_try_acquire(PipelineState state, uint32_t skip_wait = false) {
    return producer_try_acquire(state.index(), state.phase(), skip_wait);
  }

  CUTLASS_DEVICE
  void producer_acquire(PipelineState state, ProducerToken barrier_token = {BarrierStatus::WaitAgain}) {
    producer_acquire(state.index(), state.phase(), barrier_token);
  }

  CUTLASS_DEVICE
  void producer_commit(PipelineState state) {
    producer_commit(state.index());
  }

  template<class UserDefinedArriveOp>
  CUTLASS_DEVICE
  void producer_commit(PipelineState state, UserDefinedArriveOp&& user_defined_arrive_op) {
    cute::forward<UserDefinedArriveOp>(user_defined_arrive_op)(producer_get_barrier(state.index()));
    producer_commit(state);
  }

  // Prevents early exit of producer blocks in Cluster.
  // This should be called once before kernel exits.
  CUTLASS_DEVICE
  void producer_tail(PipelineState state) {
    for (int count = 0; count < Stages; ++count) {
      producer_acquire(state);
      ++state;
    }
  }

  CUTLASS_DEVICE
  ProducerBarrierType* producer_get_barrier(PipelineState state) {
    return producer_get_barrier(state.index());
  }

  ////////////////////
  // Consumer APIs
  ////////////////////
  CUTLASS_DEVICE
  ConsumerToken consumer_try_wait(PipelineState state, uint32_t skip_wait = false) {
    return consumer_try_wait(state.index(), state.phase(), skip_wait);
  }

  CUTLASS_DEVICE
  ConsumerToken consumer_test_wait(PipelineState state, uint32_t skip_wait = false) {
    return consumer_test_wait(state.index(), state.phase(), skip_wait);
  }

  CUTLASS_DEVICE
  void consumer_wait(PipelineState state, ConsumerToken barrier_token = {BarrierStatus::WaitAgain}) {
    consumer_wait(state.index(), state.phase(), barrier_token);
  }

  CUTLASS_DEVICE
  void consumer_release(PipelineState state) {
    consumer_release(state.index());
  }

  CUTLASS_DEVICE
  ProducerBarrierType* producer_get_barrier(uint32_t stage) {
    return reinterpret_cast<ProducerBarrierType*>(&full_barrier_ptr_[stage]);
  }

private:
  Params params_;
  FullBarrier *full_barrier_ptr_;
  EmptyBarrier *empty_barrier_ptr_;

  CUTLASS_DEVICE
  ProducerToken producer_try_acquire(uint32_t stage, uint32_t phase, uint32_t skip_wait) {
    detail::pipeline_check_is_producer(params_.role);
    if (skip_wait) {
      return {BarrierStatus::WaitDone};
    }
    bool barrier_status = empty_barrier_ptr_[stage].try_wait(phase);
    return {static_cast<BarrierStatus>(barrier_status)};
  }

  CUTLASS_DEVICE
  void producer_acquire(uint32_t stage, uint32_t phase, ProducerToken barrier_token) {
    detail::pipeline_check_is_producer(params_.role);
    if (barrier_token == BarrierStatus::WaitAgain) {
      empty_barrier_ptr_[stage].wait(phase);
    }
  }

  CUTLASS_DEVICE
  void producer_commit(uint32_t stage) {
    detail::pipeline_check_is_producer(params_.role);
    full_barrier_ptr_[stage].arrive();
  }

  CUTLASS_DEVICE
  ConsumerToken consumer_try_wait(uint32_t stage, uint32_t phase, uint32_t skip_wait) {
    detail::pipeline_check_is_consumer(params_.role);
    if (skip_wait) {
      return {BarrierStatus::WaitDone};
    }
    bool barrier_status = full_barrier_ptr_[stage].try_wait(phase);
    return {static_cast<BarrierStatus>(barrier_status)};
  }

  CUTLASS_DEVICE
  ConsumerToken consumer_test_wait(uint32_t stage, uint32_t phase, uint32_t skip_wait) {
    detail::pipeline_check_is_consumer(params_.role);
    if (skip_wait) {
      return {BarrierStatus::WaitDone};
    }
    bool barrier_status = full_barrier_ptr_[stage].test_wait(phase);
    return {static_cast<BarrierStatus>(barrier_status)};
  }

  CUTLASS_DEVICE
  void consumer_wait(uint32_t stage, uint32_t phase) {
    detail::pipeline_check_is_consumer(params_.role);
    bool done = full_barrier_ptr_[stage].test_wait(phase);
    if (!done) {
      full_barrier_ptr_[stage].wait(phase);
    }
  }

  CUTLASS_DEVICE
  void consumer_wait(uint32_t stage, uint32_t phase, ConsumerToken barrier_token) {
    detail::pipeline_check_is_consumer(params_.role);
    if (barrier_token == BarrierStatus::WaitAgain) {
      full_barrier_ptr_[stage].wait(phase);
    }
  }

  CUTLASS_DEVICE
  void consumer_release(uint32_t stage) {
    detail::pipeline_check_is_consumer(params_.role);
    empty_barrier_ptr_[stage].arrive(params_.dst_blockid);
  }
};
```

一定会好奇`initialize_barrier_array_pair_aligned`做了什么吧，下面做的事情是，阅读barrier.h代码，路径为

include/cutlass/arch/barrier.h

