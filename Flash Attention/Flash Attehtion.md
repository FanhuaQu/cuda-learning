参考文档是https://zhuanlan.zhihu.com/p/668888063

flash-attention 项目地址https://github.com/Dao-AILab/flash-attention

论文地址

FA1：https://arxiv.org/pdf/2205.14135

FA2: https://arxiv.org/pdf/2307.08691

FA3：https://arxiv.org/pdf/2407.08608

先弄懂原理，再精度论文，然后实现一遍



# 0x01 Standard Self-Attention

标准的Self-Attention的计算公式如下
$$
O = \operatorname{softmax}\left(Q K^{T}\right) V
$$
其中Q、K、V、O都是2D矩阵，shape是(N, d)，N是seqlen，d是headdim。由于多头注意力每个head的计算逻辑是一样的，这里只描述单个Head的情况。将上面公式展开得到下面的3-pass算法
$$
\begin{aligned}
S &= Q K^{T}, (N \times N) \\
P &= \operatorname{softmax}(S), (N \times N) \\
O &= P V, (N \times d)
\end{aligned}
$$
$Q K^{T} $获得每个query相对于所有key的点积，由于Q、K、V都是经过laynorm的数值，在直观上，点积越大，某个Q行和$K^{T}$的相关性就大。分析上面的三个公式，1和2产生的中间矩阵S和P的内存需求是$O(N^{2})$。总的HBM IO Access需求就是$O(Nd+N^{2})$。在seqlen很大的场景，会爆显存。同时HBM的访存压力也会极具变大。



Attention是Transformer中的标准组件，常见的包括Multi-Head Attention（MHA）、Mask Multi-Head Attention、Cross Attention、MQA和GQA等等。目前大部分LLM大模型以及Stable Diffusion中的基础模型，都是Transformer-Based，因此也出现很多针对Transformer进行训推性能优化的方法，这其中，优化Attention的计算效率和访存效率，可以说是重中之重。

FlashAttention不需要保留中的S和P矩阵，而是整个Attention计算融合到单个CUDA Kernel中。FlashAttention利用了Tiling(forward)+ Recompute(backward)对Attention计算进行融合，特别是对于forward阶段的tiling，可以看做是对online-softmax技术的一种延伸。

由于softmax计算需要依赖于一个全局的分母项。FlashAttention和online softmax想解决的核心问题，正是如何将算法本身从这个全局的依赖中解耦，从而可以使用Tiling进行快速的片上计算。从结果上来看，相对于原始的3-pass算法，online-softmax是2-pass算法，而FlashAttention是1-pass算法

# 0x02 Safe Softmax：3-pass

先来回顾一下safe softmax，相较于原生的softmax，减去了一个max值，以确保计算过程不会产生数值溢出。
$$
\
\operatorname{softmax}\left(\left\{x_{1},\ldots, x_{N}\right\}\right)=\left\{\frac{e^{x_{i}}}{\sum_{j=1}^{N}e^{x_{j}}}\right\}_{i=1}^{N}
$$
safe-softmax计算公式如下，由于可以保证$x_{i}-m \le 0$，因此可以确保softmax计算不会溢出
$$
\operatorname{safe-softmax} = \frac{e^{x_{i}}}{\sum_{j=1}^{N} e^{x_{j}}} = \frac{e^{x_{i}-m}}{\sum_{j=1}^{N} e^{x_{j}-m}}, \quad m = \max_{j=1}^{N} \left( x_{j} \right)
$$

* **3-pass softmax**

对于safe softmax，工程上的实现如下

1. 先遍历$x_{i}$得到最大值m
2. 遍历$x_{i}$计算上面的分母部分$d_{N}$
3. 遍历$x_{i}$得到输出$a_{i} = {e^{x_{i}-m}/d_{N}}$

这个算法要求对输入序列遍历3次。在Transformer中self-Atttention的背景下，$x$是$QK^{T}$计算的pre-softmax logits。这意味着，如果我们没有足够的SRAM保存pre-softmax logits(O(N^2))，就需要访问Q和K三次，并实时重新计算$x$，访存十分低效



# 0x03 Online Softmax：2-pass

将上面的3-pass softmax的三个操作进行融合，