最近工作接触到了GQA和MHA，借此机会总结一下MHA、GQA、MHA的原理，和flops计算

# 1、Self-Attention

首先来复习一下self-attention，公式如下
$$
Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
$$
<img src="assets\image-20251024233044396.png" alt="image-20251024233044396" style="zoom:50%;" />

这里的Q、K、V是由一个统一的输入矩阵X经过投影得到的
$$
Q=XW^Q\\
K=XW^K\\
V=XW^V\\
$$
注意这里假设了三个投影矩阵的形状相同，实际上并不是必须的，可以如下
$$
X:(seq\_len, embed\_dim)\\
W_Q:(embed\_dim, d_k)\\
W_K:(embed\_dim, d_k)\\
W_V:(embed\_dim, d_v)\\
Q:(seq_len, d_k)\\
K:(seq_len, d_k)\\
V:(seq_len, d_v)\\
output:(seq_len, dv)\\
$$
代码实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

Class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.head_dim = embed_dim
        
        # Q、K、V投影
        self.Wq = nn.Linear(embed_dim, self.head_dim)
        self.Wk = nn.Linear(embed_dim, self.head_dim)
        self.Wv = nn.Linear(embed_dim, self.hrad_dim)
    
    def forward(self, inputs):	# inputs shape: (batch_size, seq_len, embed_dim)
        batch_size, seq_len, _ = inputs.shape
        
        # 计算Q、K、V投影
        Q = self.Wq(inputs)
        K = self.Wk(inputs)
        V = self.Wv(inputs)
        
        # 计算相似度矩阵: (batch_size, seq_len, seq_len)
        attention_scores = F.matmul(Q, K.transpose(-1,-2)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        # 计算注意力权重
        attention_weight = F.softmax(attention_scores, dim=-1)
        # 计算输出
        outputs = torch.mutmul(attention_weights, V)
        return outputs
        
```

## 1.1 self-Attention计算量分析

看下fwd的计算量：
$$
\begin{align}
Q投影&: 2*batch\_size*seq\_len*embed\_dim*d\_k\\
K投影&: 2*batch\_size*seq\_len*embed\_dim*d\_k\\
V投影&: 2*batch\_size*seq\_len*embed\_dim*d\_k\\
QK^T以及V&: 4*batch\_size*seq\_len^2*d\_k\\
softmax的计算量是&:c_{soft}*batch_size*seq\_len^2  (是否要考虑？)\\
总的计算量是&: 6*batch\_size*seq\_len*embed\_dim*d\_k + 4*batch\_size*seq\_len^2*d\_k + c_{soft}*batch_size*seq\_len^2\\
后面可能的输出投影&:2*batch\_size*seq\_len*embed\_dim^2
\end{align}
$$
bwd计算量近似为fwd的两倍

# 2、Multi-Head Attention（MHA）

多头注意力是将输入映射到多个子空间(head)，在每个子空间分别做注意力计算，然后再把结果拼接回去。目的是让模型在不同的表示空间上并行捕捉不同的模式，增强表达能力。

前面提到，对于普通的self-attention来说，一般会取
$$
d_k = d_v = embed\_dim
$$
对于多头注意力机制而言，假设头数为h，通常选取
$$
head\_dim=d_k = d_v = \frac{embed\_dim}{h}
$$
延续前面的推导，每个头的输出为：
$$
output_i:(seq\_len, head\_dim)
$$
然后将每个头的输出进行拼接，然后经过输出投影，这样最终输出的形状就和单头注意力机制一样了。
$$
MultiHead(Q,K,V)=Concat(output_1, ..., output_h)W^o
$$
对于TP拓扑来说，MHA刚好可以按照头进行划分，因为各个头之间是没有相互依赖的，中间也不需要进行通信。

代码实现如下，实际算的时候QKV投影还是在一起算的，算完之后在切分，实际上独立地h个投影矩阵也是一样的。

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.wq = nn.Linear(embed_dim, embed_dim)
        self.wk = nn.Linear(embed_dim, embed_dim)
        self.wv = nn.Linear(embed_dim, embed_dim)
        self.wo = nn.Linear(embed_dim, embed_dim)
       
    def mh_split(self, hidden):
        batch_size = hidden.shape[0]
        x = hidden.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        return x
    
    def forward(self, hidden_states, mask=None):
        batch_size = hidden_states.size(0)
		
        # Q、K、V投影，(batch_size, seq_len, embed_dim)
        q, k, v = self.wq(hidden_states), self.wk(hidden_states), self.wv(hidden_states)
        
        # 多头切分(batch_size, seq_len, num_heads, embed_dim / num_heads)
        q, k, v = self.mh_split(q), self.mh_split(k), self.mh_split(v)
        
        # Attention Score
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
		if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention, v)
        
        # 拼接多头
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)

        # 线性变换
        output = self.wo(output)

        return output
        
```

可以看到和传统的self-Attention区别在于QKV投影之后分成了h个头进行计算，最后拼接之后还需要经过一次输出投影。增加头的数量可以提高表达能力，但是每个头的维度不应该太小(会限制信息容量)

## 2.1 MHA计算量分析

$$
\begin{align}
Q投影&: 2*batch\_size*seq\_len*embed\_dim*d\_k\\
K投影&: 2*batch\_size*seq\_len*embed\_dim*d\_k\\
V投影&: 2*batch\_size*seq\_len*embed\_dim*d\_k\\
QK^T以及V&: 4*batch\_size*seq\_len^2*head\_dim*h\\
softmax的计算量是&:c_{soft}*batch_size*seq\_len^2  (是否要考虑？)\\
总的计算量是&: 6*batch\_size*seq\_len*embed\_dim*d\_k + 4*batch\_size*seq\_len^2*d\_k + c_{soft}*batch_size*seq\_len^2\\
后面可能的输出投影&:2*batch\_size*seq\_len*embed\_dim^2
\end{align}
$$

会发现，MHA的计算量相较于普通的self-Attention没有变化，但是表达能力提高了。



# 3、Mutil-Query Attention（MQA）

MQA是MHA的高效变体，可以在保持模型表达能力的前提下显著降低计算和内存消耗，尤其是在推理阶段。与MHA 不同的是，**MQA 让所有的Head之间共享同样的一份 K 和 V 矩阵（意味K和V的计算唯一），只让 Q 保留了原始多头的性质**（每个Head存在不同的转换），从而大大减少 K 和 V 矩阵的参数量以及KV Cache的显存占用，以此来达到提升推理速度，但是会带来精度上的损失。

在MHA中，每个head都有自己的QKV权重，h组head就有3h个投影矩阵

在推理阶段，每个token的kv都需要缓存到kv cache中，h和head就需要缓存h份k/v张量

例如对于emded_dim = 4096, h=32, dk=dv=128, seq_len=2048

计算kv cache的大小为
$$
2 * h *  seq_len*d_k = 2 * 32 * 2048*4096=16.8\ million \ floats
$$
占用的显存不容忽视

MQA的思想是，每个头拥有独立的Q投影，但是所有头共享同一个K和V投影

计算公式如下：
$$
\begin{align}
Q_i &= XW_i^Q\\
K &= XW^K\\
V &= XW^V\\
\end{align}
$$
计算过程:
$$
\begin{align}
head_i     &= softmax(\frac{Q_i K^T}{\sqrt{d_k}})V \\
MQA(Q,K,V) &= Concat(head_1, ..., head_h)W^o 
\end{align}
$$
**如何将现有的预训练多头注意力模型转换为多查询注意力模型 (MQA)？**从现有的多头模型创建多查询注意力模型涉及两个步骤：模型结构的转换和随后的预训练

* **模型结构的转换**：此步骤将多头模型的结构转换为多查询模型。它是通过将原始模型的多个头的键和值的投影矩阵（线性层）合并(均值池化)为键和值的单个投影矩阵来实现的。这种均值池化方法被发现比选择现有键和值头之一或从头开始初始化新的键和值头更有效。生成的结构具有合并的键和值投影，这是多查询模型的特征
* **对转换后的模型进行预训练**：结构转换后，模型将接受额外的训练。此训练不像原始模型训练那样广泛；它只是原始模型训练步骤的一小部分（表示为 α）。此预训练阶段的目的是让模型根据其新的简化注意力机制调整和优化其性能。训练遵循与原始相同的方法，确保学习动态的一致性。

代码实现

```Python
import torch
import torch.nn as nn


class MultiQuerySelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiQuerySelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.wq = nn.Linear(embed_dim, embed_dim)

        # MQA
        self.wk = nn.Linear(embed_dim, self.head_dim)
        self.wv = nn.Linear(embed_dim, self.head_dim)
        self.wo = nn.Linear(embed_dim, embed_dim)

    def q_h_split(self, hidden, head_num=None):
        batch_size, seq_len = hidden.size()[:2]
        # q拆分多头
        if head_num == None:
            x = hidden.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            return x
        else:
            # 这是MQA: 需要拆分k和v,这里面的head_num =1 的
            # 最终返回维度(batch_size, 1, seq_len, head_dim)
            return hidden.view(batch_size, seq_len, head_num, self.head_dim).transpose(1, 2)

    def forward(self, hidden_states, mask=None):
        batch_size = hidden_states.size(0)

        # 线性变换
        q, k, v = self.wq(hidden_states), self.wk(hidden_states), self.wv(hidden_states)

        # 多头切分
        # 这是MHA的
        # q, k ,v  = self.split(q), self.split(k), self.split(v)
        # 这是MQA的
        q, k, v = self.q_h_split(q), self.q_h_split(k, 1), self.q_h_split(v, 1)

        # 注意力计算
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        print("scores:", scores.shape)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention, v)

        # 多头合并
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        # 线性变换
        output = self.wo(output)
        return output


```



## 3.1 MQA计算量分析

$$
\begin{align}
Q投影&: 2*batch\_size*seq\_len*embed\_dim*d\_k\\
K投影&: 2*batch\_size*seq\_len*embed\_dim*\frac{d\_k}{h}\\
V投影&: 2*batch\_size*seq\_len*embed\_dim*\frac{d\_k}{h}\\
QK^T以及V&: 4*batch\_size*seq\_len^2*head\_dim*h\\
总的计算量是&: 2*batch\_size*seq\_len*embed\_dim*d\_k + 4*batch\_size*seq\_len*embed\_dim*\frac{d\_k}{h} + 4*batch\_size*seq\_len^2*d\_k\\
加上后面的输出投影&:2*batch\_size*seq\_len*embed\_dim^2
\end{align}
$$

总的计算量比前面的MHA稍微小一点，(K/V投影部分减少成了1/h)

推理部分显著减小了内存，将kv cache节省到了1/h



# 4、Group Query Attention(GQA)

GQA是MHA和MQA的折中，虽然MQA方式大幅减小了参数数量，但是，带来推理加速的同时会造成模型性能损失，且在训练过程使得模型变得不稳定（**复杂度的降低可能会导致质量下降和训练不稳定**），因此在此基础上提出了GQA，它将Query进行分组，每个组内共享一组Key、Value。

![image-20251025205249713](assets\image-20251025205249713.png)

当G数量为1时，就变成了MQA，所有的Q头只有一个KV头

当G的数量为h时，就变成了MHA，每个Q头有对应的一个KV头

通过**利用 GQA，该模型在 MHA 质量和 MQA 速度之间保持平衡**。由于键值对较少，内存带宽和数据加载需求被最小化。G 的选择代表了一种权衡：更多的组（更接近 MHA）可带来更高的质量但性能较慢，而更少的组（接近 MQA）可提高速度但有牺牲质量的风险。此外，随着模型规模的扩大，GQA 允许内存带宽和模型容量按比例减少，与模型规模相对应。相比之下，对于更大的模型，在 MQA 中减少到单个键和值头可能会过于严重。

```python
import torch
import torch.nn as nn


class GroupedQueryAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(GroupedQueryAttention, self).__init__()
        # 这里的num_heads是指的Q头个数
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.wq = nn.Linear(embed_dim, embed_dim)

        # 这是MHA的
        # self.wk = nn.Linear(embed_dim, embed_dim)
        # self.wv = nn.Linear(embed_dim, embed_dim)

        # 这是MQA的
        # self.wk = nn.Linear(embed_dim, self.head_dim)
        # self.wv = nn.Linear(embed_dim, self.head_dim)

        # 这是GQA的
        self.group_num = 4
        # 每个头的长度是head_dim
        self.wk = nn.Linear(embed_dim, self.group_num * self.head_dim)
        self.wv = nn.Linear(embed_dim, self.group_num * self.head_dim)

        self.wo = nn.Linear(embed_dim, embed_dim)

    def split(self, hidden, group_num=None):
        batch_size, seq_len = hidden.size()[:2]
        # q需要拆分多头
        if group_num == None:
            x = hidden.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            return x
        else:
            # 这是kv需要拆分的多头
            x = hidden.view(batch_size, seq_len, group_num, self.head_dim).transpose(1, 2)
            x = x[:, :, None, :, :].expand(batch_size, group_num, self.num_heads // group_num, seq_len,
                                           self.head_dim).reshape(batch_size, self.num_heads, seq_len, self.head_dim)
            return x

    def forward(self, hidden_states, mask=None):
        batch_size = hidden_states.size(0)

        # 线性变换
        q, k, v = self.wq(hidden_states), self.wk(hidden_states), self.wv(hidden_states)

        # 多头切分
        # 这是MHA的
        # q, k ,v  = self.split(q), self.split(k), self.split(v)
        # 这是MQA的
        # q, k ,v  = self.split(q), self.split(k, 1), self.split(v, 1)
        # 这是GQA的
        q, k, v = self.split(q), self.split(k, self.group_num), self.split(v, self.group_num)

        # 注意力计算
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        print("scores:", scores.shape)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention, v)

        # 合并多头
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)

        # 线性变换
        output = self.wo(output)

        return output


```



## 4.1 GQA计算量分析

这里g表示group数量，也就是kv头的个数
$$
\begin{align}
kv_dim &= embed\_dim / h / g\\
Q投影&: 2*batch\_size*seq\_len*embed\_dim*d\_k\\
K投影&: 2*batch\_size*seq\_len*embed\_dim*\frac{d\_k}{h}*g\\
	&= 2*batch\_size*seq\_len*embed\_dim*kv\_dim\\
V投影&: 2*batch\_size*seq\_len*embed\_dim*\frac{d\_k}{h}\\
	&= 2*batch\_size*seq\_len*embed\_dim*kv\_dim\\
QK^T以及V&: 4*batch\_size*seq\_len^2*head\_dim*h\\
总的计算量是&: 2*batch\_size*seq\_len*embed\_dim*d\_k + 4*batch\_size*seq\_len*embed\_dim*kv\_dim + 4*batch\_size*seq\_len^2*d\_k\\
还要加上后面的输出投影&:2*batch\_size*seq\_len*embed\_dim^2
\end{align}
$$
