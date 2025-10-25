带着问题去学习：

* 理解TensorRT-LLM和tensorRT的关系？

可以简单理解为TensorRT LLM = TensorRT + 一堆LLM优化 + python易用接口



# Attention/KV Cache 核心机制

* FlashAttention怎么实现的？和传统的attention kernel区别是什么？
* 什么是Paged KV Cache？具体什么结构？
* 执行一次decode，kv cache是怎么复用、更新和拷贝的？
* GQA、MQA kernel代码，怎么减少内存访问的？
*  Kernel launch 参数（block/thread/grid）是如何决定的？能否进一步优化？
* 自己修改KV cache管理方式？

# Engine 构建流程

* TensorRT-LLM 的 Engine 是如何由模型 graph 转换成 TensorRT Engine 的？
* 每一层 Transformer 是如何注册到 builder 里的？
* Plugin 是怎么嵌入的？自定义算子完整流程，比如自己写一个rmsnorm，跟踪layernormKernels怎么到python的

# Plugin与自定义算子

* 哪些核心算子是 TensorRT 原生不支持的？
* 每个 Plugin 是如何注册的？生命周期是什么？
* 怎么开发自己的 Plugin（例如专门为召回 embedding 计算写个 fused kernel）？

# 量化机制

tensorrt_llm/quantization

* TensorRT-LLM 的 INT8/FP8 量化是怎么实现的？
* 权重量化与激活量化在 kernel 层面如何结合？
* PTQ/QAT 的标定流程是怎样的？

# Runtime 执行与调度

tensorrt_llm/runtime

* 一次推理的生命周期是怎样的？（输入 → Engine → CUDA → 输出）
* decode 流式推理是怎么管理多个 stream 的？
* batch scheduling（动态批处理）是如何在 TensorRT Engine 上实现的？

# GPU并行

* maybe

# Triton Backend

triton_backend

* TensorRT-LLM Engine 是如何嵌入 Triton Server 的？
* Triton 的动态 batching 是怎么和 KV cache 对齐的？