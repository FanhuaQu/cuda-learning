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

想办法将上面的3-pass softmax的三个操作进行融合，由于计算分母部分依赖于第一步的最大值m，有没有办法把这个依赖关系去掉呢？我们可以简单得到下面依赖关系
$$
d_{i}^{\prime} \leftarrow d_{i-1}^{\prime} e^{m_{i-1}-m_{i}} + e^{x_{i}-m_{i}}
$$
这样是不是可以把前面的1和2融入到一个loop里面呢？

可以发现$d_{i}$和$d_{i-1}$之间不存在依赖$m_{N}$的递归关系，先假设$d_{i}^{\prime}$如下
$$
d_{i}^{\prime}:=\sum_{j=1}^{i}e^{x_{j}-m_{i}}
$$
对于序列[1:N]，当i=N时，恰好有
$$
d_{N}=d_{N}^{\prime}:=\sum_{j=1}^{N}e^{x_{j}-m_{N}}
$$
来推导一下$d_{i}^{\prime}$和$d_{i-1}^{\prime}$之间的递归关系
$$
d_{i}^{\prime}=\sum_{j=1}^{i}e^{x_{j}-m_{i}}\\
=(\sum_{j=1}^{i-1}e^{x_{j}-m_{i}})+e^{x_{i}-m_{i}}\\
=(\sum_{j=1}^{i-1}e^{x_{j}-m_{i-1}})e^{m_{i-1}-m_{i}} +e^{x_{i}-m_{i}}\\
=d_{i-1}^{\prime} e^{m_{i-1}-m_{i}} + e^{x_{i}-m_{i}}
$$
会发现，$d_{i}^{\prime}$和$d_{i-1}^{\prime}$之间递归关系只依赖于$m_{i-1}$和$m_{i}$，所以我们可以把$d_{i}^{\prime}$和$m_{i}$的计算放到同一个循环里，当循环计算到$i=N$的时候，就得到了$d_{N}^{\prime}$，这样看的话，就是将前面的1和2合并了，减少了一次遍历。

* **Algorithm 2-pass online softmax**

经过前面的推导，可以得到2-pass的online-softmax算法

1. 

$$
m_{i} \leftarrow max(m_{i-1}, x_{i})\\
d_{i}^{\prime}\leftarrow d_{i-1}^{\prime}e^{m_{i-1}-m{i}}+e^{x_{i}-m_{i}}
$$

2. 

$$
a_{i}\leftarrow \frac{e^{x_{i-1}-m_{N}}}{d_{N}^{\prime}}
$$

在计算量上，2-pass并没有减少，甚至多了一点，也就是需要计算$d_{i-1}^{\prime}e^{m_{i-1}-m_{i}}$

前面介绍过，softmax的输入是$x=QK^T$，维度是`N*N`，显存可能是放不下的因此

1. 要么提前计算好完整的x，全部存在显存里面，但是容易爆显存
2. 要么在算法中online计算，每次循环load一部分Q、K到片上内存，计算得到x

Attention优化的目标是避开上面的1，尽量节省显存。而对于2，我们不需要保存中间结果，因为Q、K乘完之后直接就是计算softmax，节约了显存。但是增加了计算和HBM IO Accesses(不断地Load Q、K，不过这部分可以和计算overlap)，2-pass相较于3-pass，可以减少一次整体的Load Q,K以及对$x_{i}$的online recompute。2-pass对应attention中的应用是Memory Efficient Attention（注意还没到FlashAttention）

# 0x04 FlashAttention V1

单纯就safe softmax而言，并不存在1-pass算法，但是Attention的目标并不是求softmax，而是
$$
O=softmax(QK^T)V
$$
FlashAttention就是Attention的1-pass算法

## **Algorithm Multi-pass Self-Attention**

我们先看原始的self-Attention的工程实现

在第一个循环中，使用了前面推导的$d_{i}^{\prime}\leftarrow d_{i-1}^{\prime}e^{m_{i-1}-m{i}}+e^{x_{i}-m_{i}}$。第一个循环和前面的2-pass online softmax完全一致，只是增加了$x_{i}$的计算
$$
\begin{aligned}
x_{i} &\leftarrow Q[k,:] K^{T}[:,i] \\
m_{i} &\leftarrow \max \left( m_{i-1}, x_{i}\right) \\
d_{i}^{\prime} &\leftarrow d_{i-1}^{\prime} e^{m_{i-1}-m_{i}} + e^{x_{i}-m_{i}}
\end{aligned}
$$
在2-pass FalshAttention的第二个循环中，计算了概率值，以及当前步迭代得到的$o_{i}$：
$$
\begin{aligned}
a_{i} &\leftarrow \frac{e^{x_{i}-m_{N}}}{d_{N}^{\prime}} \\
\boldsymbol{o}_{i} &\leftarrow \boldsymbol{o}_{i-1} + a_{i} V[i,:]
\end{aligned}
$$
上面式子2不太理解，合并之后是
$$
o_{i} \leftarrow {o}_{i-1} + \frac{e^{x_{i}-m_{N}}}{d_{N}^{\prime}}  V[i,:]
$$
会发现$o_{i}$和$o_{i-1}$之间的递归关系依赖于$m_{N}$，所以会希望和前面2-pass一样将这个依赖关系消除

## **Algorithm 1-pass FlashAttention**

借助前面online-softmax的推导思路，推导1-pass版本的Flash-Attention，首先定义$o_{i}^{\prime}$为
$$
o_{i}^{\prime}:=\sum_{j=1}^{i}(\frac{e^{x_{j}-m_{i}}}{d_{i}^{\prime}}V[j,:])
$$
$o_{i}^{\prime}$具备的特性是，对于[1,N]，当i=N时，满足
$$
o_{N}^{\prime} = o_{N}:=\sum_{j=1}^{N}(\frac{e^{x_{j}-m_{N}}}{d_{N}^{\prime}}V[j,:])
$$
下面推导一下$o_{i}^{\prime}$和$o_{i-1}^{\prime}$之间的关系
$$
&o_{i}^{\prime}=\sum_{j=1}^{i}(\frac{e^{x_{j}-m_{i}}}{d_{i}^{\prime}}V[j,:])\\
&=(\sum_{j=1}^{i-1}(\frac{e^{x_{j}-m_{i}}}{d_{i}^{\prime}})V[j,:])+\frac{e^{x_{i}-m_{i}}}{d_{i}^{\prime}}V[i,:]\\
&=(\sum_{j=1}^{i-1}(\frac{e^{x_{j}-m_{i-1}}}{d_{i-1}^{\prime}}\frac{e^{x_{j}-m_{i}}}{e^{x_{j}-m_{i-1}}}\frac{d_{i-1}^{\prime}}{d_{i}^{\prime}})V[j,:])+\frac{e^{x_{i}-m_{i}}}{d_{i}^{\prime}}V[i,:]\\
&=(\sum_{j=1}^{i-1}(\frac{e^{x_{j}-m_{i-1}}}{d_{i-1}^{\prime}}V[j,:])\frac{d_{i-1}^{\prime}}{d_{i}^{\prime}}e^{m_{i-1}-m_{i}})+\frac{e^{x_{i}-m_{i}}}{d_{i}^{\prime}}V[i,:]\\
&=o_{i-1}^{\prime}\frac{d_{i-1}^{\prime}}{d_{i}^{\prime}}e^{m_{i-1}-m_{i}}+\frac{e^{x_{i}-m_{i}}}{d_{i}^{\prime}}V[i,:]\\
$$
这样就得到了$o_{i}^{\prime}$和$o_{i-1}^{\prime}$之间的递归关系，并不依赖于$m_{N}$，这样的话，我们就可以把第二个循环的计算，完全合并到第一个循环里面去，这就是1-pass Flash-Attention的算法
$$
\begin{aligned}
x_{i} &\leftarrow Q[k,:] K^{T}[:,i] \\
m_{i} &\leftarrow \max \left( m_{i-1}, x_{i}\right) \\
d_{i}^{\prime} &\leftarrow d_{i-1}^{\prime} e^{m_{i-1}-m_{i}} + e^{x_{i}-m_{i}}\\
o_{i}^{\prime} &= o_{i-1}^{\prime}\frac{d_{i-1}^{\prime}}{d_{i}^{\prime}}e^{m_{i-1}-m_{i}}+\frac{e^{x_{i}-m_{i}}}{d_{i}^{\prime}}V[i,:]\\
O[k,:] &\leftarrow o_{N}^{\prime}\\
\end{aligned}
$$
进一步的，前面的$QK^T$矩阵乘法可以tiling，就得到了分块Tiling版本的FlashAttention

![image-20250830214504504](C:\Users\Qufanhua\AppData\Roaming\Typora\typora-user-images\image-20250830214504504.png)

1-pass版本的Flash-Attention只需要load Q、K、V矩阵一次，减少了S和P矩阵的显存，同时减少了Q、K的IO操作















