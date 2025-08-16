TensorCore 是从Nvidia Volta 架构GPU开始支持的重要特性，使CUDA开发者能够使用混合精度来获得更高的吞吐量，而不牺牲精度。

参考：https://zhuanlan.zhihu.com/p/714517817

# 一、Tensor Core简介

## 什么是混合精度？

* 混合精度是指在底层硬件算子层面，使用半精度（FP16）作为输入和输出，使用全精度（FP32）进行中间结果计算从而不损失过多精度的技术
* 这个底层硬件层面其实指的就是Tensor Core，所以GPU上有Tensor Core是使用混合精度训练加速的必要条件

![img](https://pic1.zhimg.com/v2-c5ccec58ba0a985aa7303b7c8d7f14e8_r.jpg)

## 第一代Tensor Core

Volta架构引入了第一代Tensor Core

* 每个Tensor Core每个时钟执行64个FP32 FMA混合精度运算，SM中8个Tensor Core，每个时钟周期内总共执行512个浮点运算
* 