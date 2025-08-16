include/cutlass/arch/barrier.h

这个代码主要介绍了Hopper架构同步与通信机制(barrier)的实现，实现了不同类型的屏障，包括**Named Barrier**、**Cluster Barrier**、**Cluster Transaction Barrier** 等，用于线程组之间、CTA 之间或 Cluster 内线程块之间的同步。

# 一些通用的指令

* `fence_view_async_shared()`

发出一个 CUDA SM90 上 Hopper 架构的共享内存异步可见性屏障，用于确保已发起的 `cp.async` 操作（异步共享内存拷贝）对其他线程是 **可见的**

**注意**：它不会阻塞线程或等待拷贝完成，仅用于**确保数据在共享内存中可见**，需要配合其他的同步语句才能实现同步。

Q：为什么需要这个？

A：原因在于发出的异步拷贝指令不是立即执行的，而是排队处理的，假如线程A发出了异步拷贝指令，B使用同步语句类似__syncthreads()，假如上面的拷贝在同步之后才执行呢？这就会出现数据错乱的问题。fence_view_async_shared()的作用就是，告诉其他线程这个指令的发出，那么后续的同步一定会等到异步拷贝指令完成

```c++
CUTLASS_DEVICE void fence_view_async_shared();
CUTLASS_DEVICE
void fence_view_async_shared() {
#if CUDA_BARRIER_ENABLED
    cutlass::arch::synclog_emit_fence_view_async_shared(__LINE__);
    // 确保已发出的异步共享内存访问在 fence 之后对 CTA 中其他线程可见
    asm volatile (
        "{\n\t"
        "fence.proxy.async.shared::cta; \n"
        "}"
        ::);
#elif defined(__CUDA_ARCH__)
  asm volatile ("brkpt;\n" ::);
#endif
}
```

* `initialize_barrier_array`

初始化barrier数组，里面的arv_cnt，表示 **barrier 的“到达计数”（arrival count）**，也就是参与这个同步点的线程或 warp 数，在这里应该是warp数量

```c++
// Single threaded versions that need to be called in an elect_one region
template<typename T, uint32_t Stages>
CUTLASS_DEVICE
void initialize_barrier_array(T ptr, int arv_cnt) {
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < Stages; i++) {
    ptr[i].init(arv_cnt);
  }
}

template<typename T, uint32_t Stages>
CUTLASS_DEVICE
void initialize_barrier_array(uint64_t *ptr, int arv_cnt) {
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < Stages; i++) {
    T::init(&ptr[i], arv_cnt);
  }
}

// 同时初始化full和empty数组
template<typename FullBarrier, typename EmptyBarrier, uint32_t Stages>
CUTLASS_DEVICE
void initialize_barrier_array_pair(FullBarrier full_barriers, EmptyBarrier empty_barriers, int full_barrier_arv_cnt, int empty_barrier_arv_cnt) {
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < Stages; i++) {
    full_barriers[i].init(full_barrier_arv_cnt);
    empty_barriers[i].init(empty_barrier_arv_cnt);
  }
}

template<typename FullBarrier, typename EmptyBarrier, uint32_t Stages>
CUTLASS_DEVICE
void initialize_barrier_array_pair(uint64_t *full_barriers_ptr, uint64_t *empty_barriers_ptr, int full_barrier_arv_cnt, int empty_barrier_arv_cnt) {
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < Stages; i++) {
    FullBarrier::init(&full_barriers_ptr[i], full_barrier_arv_cnt);
    EmptyBarrier::init(&empty_barriers_ptr[i], empty_barrier_arv_cnt);
  }
}

// Aligned versions that need to be call warp wide
// 选举一个线程完成初始化，
// 多线程对同一个 barrier 执行 init() 会产生冲突，可能导致同步失败、死锁或 undefined behavior
template<typename T, uint32_t Stages>
CUTLASS_DEVICE
void initialize_barrier_array_aligned(T ptr, int arv_cnt) {
  if(cute::elect_one_sync()) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Stages; i++) {
      ptr[i].init(arv_cnt);
    }
  }
}

template<typename T, uint32_t Stages>
CUTLASS_DEVICE
void initialize_barrier_array_aligned(uint64_t *ptr, int arv_cnt) {
  if(cute::elect_one_sync()) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Stages; i++) {
      T::init(&ptr[i], arv_cnt);
    }
  }
}

// 上面的组合，也是pipeline里面用到的
template<typename FullBarrier, typename EmptyBarrier, uint32_t Stages>
CUTLASS_DEVICE
void initialize_barrier_array_pair_aligned(FullBarrier full_barriers, EmptyBarrier empty_barriers, int full_barrier_arv_cnt, int empty_barrier_arv_cnt) {
  if(cute::elect_one_sync()) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Stages; i++) {
      full_barriers[i].init(full_barrier_arv_cnt);
      empty_barriers[i].init(empty_barrier_arv_cnt);
    }
  }
}

template<typename FullBarrier, typename EmptyBarrier, uint32_t Stages>
CUTLASS_DEVICE
void initialize_barrier_array_pair_aligned(uint64_t *full_barriers_ptr, uint64_t *empty_barriers_ptr, int full_barrier_arv_cnt, int empty_barrier_arv_cnt) {
  if(cute::elect_one_sync()) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Stages; i++) {
      FullBarrier::init(&full_barriers_ptr[i], full_barrier_arv_cnt);
      EmptyBarrier::init(&empty_barriers_ptr[i], empty_barrier_arv_cnt);
    }
  }
}
```

* ReservedNamedBarriers

**Hopper 架构**中，引入了 **`mbarrier::named_barrier`**，允许开发者为同步操作指定一个“名字”（即编号），从而更灵活地控制不同 warp、线程组间的同步.这些编号（name）必须是 **唯一且全局可控的整数值（`0 ~ 15`）**。如果多个模板或者kernel使用了同一个barrier编号，很有可能会发生冲突。下面的代码是cutlass保留的barrier编号，用于防止冲突

```c++
enum class ReservedNamedBarriers { 
  EpilogueBarrier = 1,
  TransposeBarrier = 2,
  TransformBarrier = 3,
  StreamkBarrier0 = 4,
  StreamkBarrier1 = 5
  , FirstUserBarrier = StreamkBarrier1 + 1
};
```



# class NamedBarrier

数据成员

* uint32_t const num_threads_;   

  barrier的到达计数，范围是[1, num_threads_CTA]

* uint32_t const id_; 

​	barrier id，[0，15]

方法：

* 构造函数

```c++
  CUTLASS_DEVICE
  NamedBarrier(uint32_t num_threads, ReservedNamedBarriers reserved_named_barriers)
      : num_threads_(num_threads), id_(static_cast<uint32_t>(reserved_named_barriers)) {}

  // Constructor for CUTLASS users:
  // effective barrier ID starts from ReservedNamedBarrierCount
  CUTLASS_DEVICE
  NamedBarrier(uint32_t num_threads, uint32_t id = 0)
      : num_threads_(num_threads), id_(id + ReservedNamedBarrierCount) {
    CUTLASS_ASSERT(id + ReservedNamedBarrierCount <= HardwareMaxNumNamedBarriers && "Effective barrier_id should not exceed 16.");
  }
```

private 方法：

* arrive_and_wait_internal

  当前线程 **到达 barrier 并阻塞**，直到达到指定数量的线程也都到达为止，会发现就是之前用的指令

  下面两个方法的区别在于，对齐版本要求等待线程是整的warp，性能更好。非对齐版本更灵活，但是略慢

```c++
  CUTLASS_DEVICE
  static void arrive_and_wait_internal(uint32_t num_threads, uint32_t barrier_id) {
#if CUDA_BARRIER_ENABLED
    asm volatile("bar.sync %0, %1;" : : "r"(barrier_id), "r"(num_threads));
    cutlass::arch::synclog_emit_named_barrier_arrive_and_wait(__LINE__, num_threads, barrier_id);
#elif defined(__CUDA_ARCH__)
    asm volatile ("brkpt;\n" ::);
#endif
  }

// 非warp对齐版本
  CUTLASS_DEVICE
  static void arrive_internal(uint32_t num_threads, uint32_t barrier_id) {
#if CUDA_BARRIER_ENABLED
    cutlass::arch::synclog_emit_named_barrier_arrive(__LINE__, num_threads, barrier_id);
    asm volatile("bar.arrive %0, %1;" : : "r"(barrier_id), "r"(num_threads));
#elif defined(__CUDA_ARCH__)
    asm volatile ("brkpt;\n" ::);
#endif
  }


```

* arrive_internal

这两个函数都表示 **“我到了 named barrier，但我不等待，直接走人”**。

同步操作被解耦成两个阶段，arrive只是通知线程到达不阻塞，wait则会在barrier上等待足够的线程到达。

```c++
  CUTLASS_DEVICE
  static void arrive_internal(uint32_t num_threads, uint32_t barrier_id) {
#if CUDA_BARRIER_ENABLED
    cutlass::arch::synclog_emit_named_barrier_arrive(__LINE__, num_threads, barrier_id);
    asm volatile("bar.arrive %0, %1;" : : "r"(barrier_id), "r"(num_threads));
#elif defined(__CUDA_ARCH__)
    asm volatile ("brkpt;\n" ::);
#endif
  }

  CUTLASS_DEVICE
  static void arrive_internal_unaligned(uint32_t num_threads, uint32_t barrier_id) {
#if CUDA_BARRIER_ENABLED
    cutlass::arch::synclog_emit_named_barrier_arrive(__LINE__, num_threads, barrier_id);
    asm volatile("barrier.arrive %0, %1;" : : "r"(barrier_id), "r"(num_threads));
#elif defined(__CUDA_ARCH__)
    asm volatile ("brkpt;\n" ::);
#endif
  }

  CUTLASS_DEVICE
  static void sync_internal(uint32_t num_threads, uint32_t barrier_id) {
    NamedBarrier::arrive_and_wait_internal(num_threads, barrier_id);
  }
```

public方法

* arrive_and_wait
* arrive_and_wait_unaligned
* arrive
* arrive_unaligned
* sync：就是arrive_and_wait()



# struct ClusterBarrier

是 CUDA Hopper 引入的 **共享内存（SMEM）上构建的跨 CTA 的 barrier 同步机制**，主要用于 **多个 CTA 之间的数据交换、协调计算阶段等复杂任务的同步**

数据成员：

```c++
using ValueType = uint64_t;
protected:
	ValueType barrier_;
```

成员方法

* init

初始化一个共享内存地址处的 barrier 结构，用于支持后续的跨 CTA 同步

`mbarrier.init.shared::cta.b64` 是 CUDA Hopper 新指令，用于初始化一个在 CTA 局部共享内存中的 `mbarrier` 对象

```c++
  CUTLASS_DEVICE
  void init(uint32_t arrive_count) const {
    ClusterBarrier::init(&this->barrier_, arrive_count);
  }

  CUTLASS_DEVICE
  static void init(ValueType const* smem_ptr, uint32_t arrive_count) {
#if CUDA_BARRIER_ENABLED
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "{\n\t"
        "mbarrier.init.shared::cta.b64 [%1], %0; \n"
        "}"
        :
        : "r"(arrive_count), "r"(smem_addr));
    cutlass::arch::synclog_emit_cluster_barrier_init(__LINE__, smem_addr, arrive_count);
#elif defined(__CUDA_ARCH__)
    asm volatile ("brkpt;\n" ::);
#endif
  }
```

* wait

传入的参数是指向共享内存中的 barrier 对象和要等待的phase值

采用轮询的方式等待直到phase到达phase，设置了一个粗略的超时时间，主要防止一直卡住

在 CUTLASS/Hopper 的 `ClusterBarrier` 中，**`phase` 表示当前同步的轮次**。每次所有参与的 CTA 都 `arrive()` 之后，barrier 进入下一个 `phase`。这使得同一个 barrier 可以复用多次，不同轮次使用不同 `phase` 值来判断是否满足

```c++
  CUTLASS_DEVICE
  void wait(uint32_t phase) const {
    ClusterBarrier::wait(&this->barrier_, phase);
  }
  // Static version of wait - in case we don't want to burn a register
  CUTLASS_DEVICE
  static void wait(ValueType const* smem_ptr, uint32_t phase) {
#if CUDA_BARRIER_ENABLED
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    cutlass::arch::synclog_emit_cluster_barrier_wait(__LINE__, smem_addr, phase);
    // Arbitrarily large timer value after which try-wait expires and re-tries.
      
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred       P1; \n\t"
        "LAB_WAIT: \n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2; \n\t"
        "@P1 bra DONE; \n\t"
        "bra     LAB_WAIT; \n\t"
        "DONE: \n\t"
        "}"
        :
        : "r"(smem_addr), "r"(phase), "r"(ticks));

#elif defined(__CUDA_ARCH__)
    asm volatile ("brkpt;\n" ::);
#endif
  }

```

* try_wait

跟上面的区别在于，不会轮询阻塞，而是直接返回检查结果

```c++
  CUTLASS_DEVICE
  bool try_wait(uint32_t phase) const {
    return ClusterBarrier::try_wait(&this->barrier_, phase);
  }

  CUTLASS_DEVICE
  static bool try_wait(ValueType const* smem_ptr, uint32_t phase) {
#if CUDA_BARRIER_ENABLED
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    cutlass::arch::synclog_emit_cluster_barrier_try_wait(__LINE__, smem_addr, phase);
    uint32_t waitComplete;

    asm volatile(
        "{\n\t"
        ".reg .pred P1; \n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2; \n\t"
        "selp.b32 %0, 1, 0, P1; \n\t"
        "}"
        : "=r"(waitComplete)
        : "r"(smem_addr), "r"(phase));

    return static_cast<bool>(waitComplete);
#elif defined(__CUDA_ARCH__)
    asm volatile ("brkpt;\n" ::);
#endif
    return 0;
  }
```

* arrive

是 **Hopper 架构中 cluster-wide barrier 同步机制** 的关键实现，它让一个 CTA（线程块）可以向同一个 cluster 中的**另一个 CTA 发出 barrier 到达（arrive）信号**

传入参数是CTA在cluster中的id cta_id，一个bool值，表示是否执行这个 `arrive` 操作

mapa.shared::cluster.u32  remAddr32, %0, %1;

**作用**：给定一个共享内存地址 `smem_addr`（在本 CTA 内），和目标 CTA 的 cluster-local ID `cta_id`，计算出该目标 CTA 上的 **相同位置（offset）** 的共享内存地址。也就是把当前 CTA 的 `&barrier_` 映射到 cluster 内第 `cta_id` 个 CTA 的 `&barrier_`

mbarrier.arrive.shared::cluster.b64 [remAddr32]

 **跨 CTA 的 barrier arrive 指令**，会向 `remAddr32` 指定地址（也就是目标 CTA 的 barrier）发送一个 arrive

```c++
  CUTLASS_DEVICE
  void arrive(uint32_t cta_id, uint32_t pred = true ) const {
    ClusterBarrier::arrive(&this->barrier_, cta_id, pred);
  }

  CUTLASS_DEVICE
  static void arrive(ValueType const* smem_ptr, uint32_t cta_id, uint32_t pred) {
#if CUDA_BARRIER_ENABLED
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    if (pred) {
      asm volatile(
          "{\n\t"
          ".reg .b32 remAddr32;\n\t"
          "mapa.shared::cluster.u32  remAddr32, %0, %1;\n\t"
          "mbarrier.arrive.shared::cluster.b64  _, [remAddr32];\n\t"
          "}"
          :
          : "r"(smem_addr), "r"(cta_id));
    }

    cutlass::arch::synclog_emit_cluster_barrier_arrive_cluster(__LINE__, smem_addr, cta_id, pred);
#elif defined(__CUDA_ARCH__)
    asm volatile ("brkpt;\n" ::);
#endif
  }

// local smem arrive
  CUTLASS_DEVICE
  static void arrive(ValueType const* smem_ptr) {
#if CUDA_BARRIER_ENABLED
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "{\n\t"
        "mbarrier.arrive.shared::cta.b64 _, [%0];\n\t"
        "}"
        :
        : "r"(smem_addr));
    cutlass::arch::synclog_emit_cluster_barrier_arrive(__LINE__, smem_addr);
#elif defined(__CUDA_ARCH__)
    asm volatile ("brkpt;\n" ::);
#endif
  }
```

* invalidate

 **Hopper 架构的 `mbarrier` 同步机制** 提供的一个“清除/重置”操作

```c++
  CUTLASS_DEVICE
  static void invalidate(ValueType const* smem_ptr) {
#if CUDA_BARRIER_ENABLED
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "{\n\t"
        "mbarrier.inval.shared::cta.b64 [%0]; \n\t"
        "}"
        :
        : "r"(smem_addr));
#elif defined(__CUDA_ARCH__)
    asm volatile ("brkpt;\n" ::);
#endif
  }
```

# struct ClusterTransactionBarrier

传统的 barrier 只能控制线程/CTA 何时到达，但无法感知内存事务、DMA、复制等具体的数据量完成情况。

SM90 引入了**transaction-aware barrier**，让我们可以1)控制线程何时同步，2)控制 **多少字节的数据流** 完成之后再继续，3)支持 finer-grained async copy、pipeline 等使用场景

方法：

* arrive_and_expect_tx

语义是线程到达，但是还在等待transaction_bytes这么多数据

等待其他线程发起barrier.complete_transaction(bytes);标记数据就绪

相较于fence_view_async_shared，优势在于可以跨CTA控制

```c++
// local CTA版本  
CUTLASS_DEVICE
  void arrive_and_expect_tx(uint32_t transaction_bytes) const {
    ClusterTransactionBarrier::arrive_and_expect_tx(&this->barrier_, transaction_bytes);
  }

  CUTLASS_DEVICE
  static void arrive_and_expect_tx(ValueType const* smem_ptr, uint32_t transaction_bytes) {
#if CUDA_BARRIER_ENABLED
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "{\n\t"
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%1], %0; \n\t"
        "}"
        :
        : "r"(transaction_bytes), "r"(smem_addr));
    cutlass::arch::synclog_emit_cluster_transaction_barrier_arrive_and_expect_tx(__LINE__, smem_addr, transaction_bytes);
#elif defined(__CUDA_ARCH__)
    asm volatile ("brkpt;\n" ::);
#endif
  }

// 远程CTA版本
  CUTLASS_DEVICE
  void arrive_and_expect_tx(uint32_t transaction_bytes, uint32_t cta_id, uint32_t pred = 1u) const {
    ClusterTransactionBarrier::arrive_and_expect_tx(&this->barrier_, transaction_bytes , cta_id, pred);
  }

  CUTLASS_DEVICE
  static void arrive_and_expect_tx(
      ValueType const* smem_ptr, uint32_t transaction_bytes, uint32_t cta_id, uint32_t pred) {
#if CUDA_BARRIER_ENABLED
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        ".reg .b32 remAddr32;\n\t"
        "setp.eq.u32 p, %2, 1;\n\t"
        "@p mapa.shared::cluster.u32  remAddr32, %0, %1;\n\t"
        "@p mbarrier.arrive.expect_tx.shared::cluster.b64  _, [remAddr32], %3;\n\t"
        "}"
        :
        : "r"(smem_addr), "r"(cta_id), "r"(pred), "r"(transaction_bytes));
#elif defined(__CUDA_ARCH__)
    asm volatile ("brkpt;\n" ::);
#endif
  }
```

* expect_transaction

用于在 **不触发“arrive”操作** 的情况下，**仅设置期望的事务大小**,告诉 barrier：

> “我还没到（arrive），但我希望将来有 **transaction_bytes 字节** 的数据传输完成之后再唤醒我。”

```c++
CUTLASS_DEVICE
  void expect_transaction(uint32_t transaction_bytes) const {
    ClusterTransactionBarrier::expect_transaction(&this->barrier_, transaction_bytes);
  }

  CUTLASS_DEVICE
  static void expect_transaction(ValueType const* smem_ptr, uint32_t transaction_bytes) {
#if CUDA_BARRIER_ENABLED
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "{\n\t"
        "mbarrier.expect_tx.shared::cta.b64 [%1], %0; \n\t"
        "}"
        :
        : "r"(transaction_bytes), "r"(smem_addr));
    cutlass::arch::synclog_emit_cluster_transaction_barrier_expect_transaction(__LINE__, smem_addr, transaction_bytes);
#elif defined(__CUDA_ARCH__)
    asm volatile ("brkpt;\n" ::);
#endif
  }
```

* complete_transaction

```c++
  CUTLASS_DEVICE
  void complete_transaction(uint32_t transaction_bytes, uint32_t pred = 1) const {
    uint32_t cta_rank = cute::block_rank_in_cluster();
    ClusterTransactionBarrier::complete_transaction(&this->barrier_, cta_rank, transaction_bytes, pred);
  }


  CUTLASS_DEVICE
  void complete_transaction(uint32_t dst_cta_id, uint32_t transaction_bytes, uint32_t pred) const {
    ClusterTransactionBarrier::complete_transaction(&this->barrier_, dst_cta_id, transaction_bytes, pred);
  }

  CUTLASS_DEVICE
  static void complete_transaction(
      ValueType const* smem_ptr, uint32_t dst_cta_id, uint32_t transaction_bytes, uint32_t pred = 1) {
#if CUDA_BARRIER_ENABLED
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    smem_addr = cute::set_block_rank(smem_addr, dst_cta_id);
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.eq.u32 p, %2, 1;\n\t"
        "@p mbarrier.complete_tx.shared::cluster.relaxed.cluster.b64   [%1], %0;"
        "}"
        :
        : "r"(transaction_bytes), "r"(smem_addr), "r"(pred));
    cutlass::arch::synclog_emit_cluster_transaction_barrier_complete_transaction(__LINE__, smem_addr, dst_cta_id, transaction_bytes, pred);
#elif defined(__CUDA_ARCH__)
    asm volatile ("brkpt;\n" ::);
#endif
  }
```



