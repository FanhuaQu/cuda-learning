Attention的意思是注意力，例如拿到一张图片我们会第一眼关注到一些部分，而忽略一些不起眼的部分。

Q、K、V的概念：Q就是Query，就是一个查询或请求。K和V分别是Key和Value。以搜索场景为例，我们搜索的内容就是query，搜索引擎根据query去数据库中查询相关的标签key，然后展示key对应的结果value。

怎么理解self-attention呢？attention可以看成是从source到target的映射，例如从中文到英文。而self-attention是source内部元素之间发生的attention机制。

计算过程：

首先input尺寸是[seq_len, hidden_dim]，进行线性投影分别得到Q、K、V，投影权重的尺寸是[hidden_dim, hidden_dim]。Q、K、V的尺寸依然是[seq_len, hidden_dim]。
$$
Q = X @ W_Q\\
K = X@W_K\\
V = X@W_V
$$


然后是计算attention scores，逻辑是计算Q的每一行对V的转置作点积。
$$
atten\ scores = QK^T
$$
输出结果的尺寸是[seq_len, seq_len]，这就是权重注意力矩阵
然后下一步是对attention scores作softmax，作用是进行归一化
$$
attten = softmax(QK^T)/sqrt(hedden_dim)
$$
然后是将归一化之后的attention scores与value相乘。输出尺寸是[seq_len, hidden].
$$
Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
$$




















参考文档：

https://zhuanlan.zhihu.com/p/619154409