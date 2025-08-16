这个文档清晰很多

https://zhuanlan.zhihu.com/p/662089556

CuTe 提供了一个“` 布局`代数”，以支持以不同的方式组合布局，包括下列运算

* Layout functional
* 布局 `“产品 `”的概念，用于根据另一种布局重现一种布局
* `布局 `“划分”的概念，用于根据另一种布局拆分一个布局



# 合并(Coalesce)

`Layouts are functions from integers to integers.`

合并之前布局中的一些模式，例如

```c++
auto layout = Layout<Shape <_2,Shape <_1,_6>>,
                     Stride<_1,Stride<_6,_2>>>{};    // (_2,(_1,_6)):(_1,(_6,_2))
auto result = coalesce(layout);    // _12:_1
```

目的是什么？

这小节看得不明白，**先挖个坑**



## 按模式合并(By-mode Coalesce)

例如当有一个2D Layout，希望合并结果保持2D

```c++
// Apply coalesce at the terminals of trg_profile
Layout coalesce(Layout const& layout, IntTuple const& trg_profile)
// example
auto a = Layout<Shape <_2,Shape <_1,_6>>,
                Stride<_1,Stride<_6,_2>>>{};
auto result = coalesce(a, Step<_1,_1>{});   // (_2,_6):(_1,_2)
// Identical to
auto same_r = make_layout(coalesce(layout<0>(a)),
                          coalesce(layout<1>(a)));

```



# Composition

没太看明白，先跳过

## Computing Composition



# 补集（complement）

Layout的本质是函数，函数的本质是集合，Layout定义了从domain到codomain的投影，当codomain存在不连续时，则存在空洞的位置，如图4所示，这时候我们可以构造一个Layout2能够填充上codomain的空洞位置，此时我们构造的Layout则为原Layout的补集，同时为了表示的简洁性，补集会被压缩为最小表示，周期性重复的部分会被约掉

![img](https://picx.zhimg.com/v2-d9bac5fef72c489238bf31bd1a660d67_r.jpg)

# 乘法（product）

重复某个Tensor若干次。由于Tensor的表示是有高维数据，所以其上的乘法在实现上也有多个，但本质不变。cute中定义的乘法包含如下5个
[logical_product](https://zhida.zhihu.com/search?content_id=235259605&content_type=Article&match_order=1&q=logical_product&zhida_source=entity)，[tiled_product](https://zhida.zhihu.com/search?content_id=235259605&content_type=Article&match_order=1&q=tiled_product&zhida_source=entity)，[zipped_product](https://zhida.zhihu.com/search?content_id=235259605&content_type=Article&match_order=1&q=zipped_product&zhida_source=entity)，[blocked_product](https://zhida.zhihu.com/search?content_id=235259605&content_type=Article&match_order=1&q=blocked_product&zhida_source=entity)，raked_product

两个Layout对其进行相乘，其中第一个shape:(x, y)，第二个shape: (z, w), 则其乘积的shape: (x, y, z, w)，

| 乘法模式 | 乘积的shape      |
| -------- | ---------------- |
| logical  | ((x, y), (z, w)) |
| zipped   | ((x, y), (z, w)) |
| tiled    | ((x, y), z, w)   |
| blocked  | ((x, z), (y, w)) |
| raked    | ((z, x), (w, y)) |

在Layout x Layout的计算中，首先将Layout1按照Layout2的顺序进行重复，原来Layout2中的位置被Layout1所占据，然后将其中的内层数据数据Layout1按照列优先的顺序排成一列，外层Layout按照列优先排列后表示为最终矩阵的列

![img](https://pica.zhimg.com/v2-239bd71fcafce541b9109c1a4bc51a00_r.jpg)



# 除法（divide）

除法是乘法逆运算，实数域上的除法表示被除数能被除数分多少份。如 10÷5=2，Layout上的除法也有类似的逻辑，但Layout除法和实数域不同的是Layout除法的结果是一种划分层次，但不是被划分的结果。如果用实数域的公式来表达则为 10÷5=(5,2)。其中作为结果的括号中的5表示被分解的块的大小，2表示可以被分解多少次。

![img](https://picx.zhimg.com/v2-16ba096faf1b1b08391a4c11c3e0d99d_r.jpg)



看的迷迷糊糊的，快进到Tensor

