https://docs.nvidia.com/cutlass/media/docs/cpp/cute/03_tensor.html

直接看reed神的播客吧，配合做实验

https://zhuanlan.zhihu.com/p/663093816

Tensor就是在Layout的基础上包含了存储，即Tensor = Layout + storage

cute中的Tensor并不同于深度学习框架中的Tensor，深度学习框架中的Tensor更强调数据的表达实体，通过Tensor实体与实体之间的计算产生新的Tensor实体，即多份数据实体，cute中的Tensor更多的是对Tensor进行分解和组合等操作，而这些操作多是对Layout的变换（只是**逻辑层面的数据组织形式**），底层的数据实体一般不变更。

# Tensor的生成

```c++
// 栈上对象：需同时指定类型和Layout，layout必须是静态shape
Tensor make_tensor<T>(Layout layout);
// 举例
auto tensor_1 = cute::make_tensor<int>       (cute::make_layout(cute::make_shape(cute::Int<2>{}, cute::Int<4>{})));
cute::print_tensor(tensor_1);
/*
ptr[32b](0x7ffc7aa1f180) o (_2,_4):(_1,_2):
    0    0    0    0
    0    0    0    0
*/


// 堆上对象：需指定pointer和Layout，layout可动可静
Tensor make_tensor(Pointer pointer, Layout layout);
// 举例
int* data = new int[8];
for (int i = 0; i < 8; ++i) {
    data[i] = i;
}
auto tensor_2 = cute::make_tensor(data, 									     cute::make_layout(cute::make_shape(2, 4),                                   cute::make_stride(4, 1)));
cute::print_tensor(tensor_2);
/*
ptr[32b](0x5565857724f0) o (2,4):(4,1):
    0    1    2    3
    4    5    6    7
*/

// 栈上对象，tensor的layout必须是静态的
Tensor make_tensor_like(Tensor tensor); 
// 举例
auto tensor_3 = cute::make_tensor_like(tensor_1);
cute::print_tensor(tensor_3);
/*
ptr[32b](0x7ffe10388560) o (_2,_4):(_1,_2):
    0    0    0    0
    0    0    0    0
*/

// 栈上对象，tensor的layout必须是静态的
Tensor make_fragment_like(Tensor tensor);
cute::Tensor tensor_4 = cute::make_fragment_like(tensor_1);
cute::print_tensor(tensor_4);
/*
ptr[32b](0x7ffc0fe6abc0) o (_2,(_4)):(_1,(_2)):
    0    0    0    0
    0    0    0    0
*/

```

# Tensor的成员函数

```c++
// 成员函数
Tensor::layout();
Tensor::shape();
Tensor::stride();
Tensor::size();

// 全局函数, 可以获取完整信息，或者通过<>获取某一个维度
auto cute::layout<>(Tensor tensor);
auto cute::shape<>(Tensor tensor);
auto cute::stride<>(Tensor tensor);
auto cute::size<>(Tensor tensor);
auto cute::rank<>(Tensor tensor); // (1, (2, 3)) => rank 2 
auto cute::depth<>(Tensor tensor);

// ***************************************************
int* data = new int[8];
for (int i = 0; i < 8; ++i) {
    data[i] = i;
}
auto tensor_2 = cute::make_tensor(data, 									     cute::make_layout(cute::make_shape(2, 4),                                   cute::make_stride(4, 1)));
cute::print_tensor(tensor_2);

// layout()
cute::print_layout(tensor_2.layout());
/*
(2,4):(4,1)
      0   1   2   3 
    +---+---+---+---+
 0  | 0 | 1 | 2 | 3 |
    +---+---+---+---+
 1  | 4 | 5 | 6 | 7 |
    +---+---+---+---+
*/

// shape()
cute::print(tensor_2.shape());
/*
(2,4)
*/

// stride()
cute::print(tensor_2.stride()); std::cout<<std::endl;
/*
(4,1)
*/

// size()
cute::print(tensor_2.size()); std::cout<<std::endl;
/*
8
*/

std::cout << "layout<0>: " << cute::layout<0>(tensor_2) << "   layout<1>: " << cute::layout<1>(tensor_2) << std::endl;
std::cout << "shape<0>: " << cute::shape<0>(tensor_2) << "   shape<1>: " << cute::shape<1>(tensor_2) << std::endl;
std::cout << "stride<0>: " << cute::stride<0>(tensor_2) << "   stride<1>: " << cute::stride<1>(tensor_2) << std::endl;
std::cout << "depth<0>: " << cute::depth<0>(tensor_2) << "   depth<1>: " << cute::depth<1>(tensor_2) << std::endl;
/*
layout<0>: 2:4   layout<1>: 4:1
shape<0>: 2   shape<1>: 4
stride<0>: 4   stride<1>: 1
depth<0>: _0   depth<1>: _0
*/

// 通过()/[]访问
auto tensor_5 = cute::make_tensor<int>(cute::make_layout(cute::make_shape(cute::Int<100>{}, cute::Int<200>{})));
auto coord = cute::make_coord(20, 30);
tensor_5(0) = 1;
tensor_5(1,2) = 100;
tensor_5(coord) = 200;
std::cout << "tensor_5(0): " << tensor_5(0) << "   tensor_5(1,2): " << tensor_5(1,2) << "   tensor_5(coord): " << tensor_5(coord) << std::endl;

// 获取Tensor地址
std::cout << "address of tensor_5: " << tensor_5.data() << std::endl;

// Slice，通过_来筛选特定的轴
int* data1 = new int[120];
cute::Tensor tensor_6 = cute::make_tensor(data1, cute::make_shape(cute::_4{}, cute::_5{}, cute::_6{}));
cute::Tensor tensor_6_1 = tensor_6(cute::_, cute::_, 3);
cute::print(tensor_6_1.shape()); std::cout<<std::endl;  // (_4,_5)

// Take，take出指定轴上的数据
cute::Tensor tensor_6_2 = cute::take<0,1>(tensor_6);
cute::print_tensor(tensor_6_2);
/*
ptr[32b](0x55f449388520) o (_4):(_1):
    0
    0
    0
    0
*/

// flatten，将layout展开为一层 没有get到用法
cute::Tensor tensor_7 = cute::flatten(tensor_6);

// Tensor的层级合并coalesce  todo
Tensor tensor = make_tensor(ptr, make_shape(M, N));
Tensor tensor1 = coalesce(tensor);

// Tensor的主轴层级化group_modes todo
// Tensor的划分logical_divide/tiled_divide/zipped_divide  todo
// Layout的乘积logical/zipped/tiled/blocked/raked  todo

// Tensor的局部化切块local_tile
// 这是一个经常用到的函数，用于对tensor分块
// 展示了将维度为MNK的张量按照2x3x4的小块进行划分，取出其中的第(1, 2, 3)块
int* data2 = new int[24000];
for (int i = 0; i < 24000; ++i) {
    data2[i] = i;
}
auto tensor_8 = cute::make_tensor(data2, cute::make_shape(20, 30, 40));
auto tensor_8_1 = cute::local_tile(tensor_8, cute::make_shape(cute::_2{}, cute::_3{}, cute::_4{}), cute::make_coord(1, 2, 3));
cute::print_tensor(tensor_8_1);
/*
ptr[32b](0x56061df3e978) o (_2,_3,_4):(_1,20,600):
 7322 7342 7362
 7323 7343 7363
---------------
 7922 7942 7962
 7923 7943 7963
---------------
 8522 8542 8562
 8523 8543 8563
---------------
 9122 9142 9162
 9123 9143 9163
*/

// Tensor的局部数据提取local_partition
// 传参是(tensor, layout, partition)
int* data3 = new int[24];
for (int i = 0; i < 24; ++i) {
    data3[i] = i;
}
    auto tensor_9 = cute::make_tensor(data3, cute::make_shape(6, 4));
    auto tile_layout = cute::make_layout(cute::make_shape(3, 2));
    auto tensor_9_1 = cute::local_partition(tensor_9, tile_layout, 0);
    cute::print_tensor(tensor_9);
    cute::print_tensor(tensor_9_1);
/*
ptr[32b](0x560f8fc03e20) o (6,4):(_1,6):
    0    6   12   18
    1    7   13   19
    2    8   14   20
    3    9   15   21
    4   10   16   22
    5   11   17   23
ptr[32b](0x55fe4213ce20) o (2,2):(3,12):
    0   12
    3   15
*/
```

怎么理解上面的local_partition？下面这张图

![img](https://pica.zhimg.com/v2-940f54c0095cfc2860f772d985cbb726_r.jpg)

继续

```c++
// Tensor数据类型转换recast
// 类似于C++中的reinterpret_cast语义
auto tensor_10 = cute::make_tensor(data4, cute::make_shape(6, 4));
auto tensor_10_1 = cute::recast<int>(tensor_10);
cute::print_tensor(tensor_10);
cute::print_tensor(tensor_10_1);

// Tensor内容的填充fill和清除clear
cute::fill(tensor_10, 3.14f);
cute::print_tensor(tensor_10);
cute::clear(tensor_10);
cute::print_tensor(tensor_10);
/*
ptr[32b](0x55f6a53f6e90) o (6,4):(_1,6):
  3.14e+00  3.14e+00  3.14e+00  3.14e+00
  3.14e+00  3.14e+00  3.14e+00  3.14e+00
  3.14e+00  3.14e+00  3.14e+00  3.14e+00
  3.14e+00  3.14e+00  3.14e+00  3.14e+00
  3.14e+00  3.14e+00  3.14e+00  3.14e+00
  3.14e+00  3.14e+00  3.14e+00  3.14e+00
ptr[32b](0x55f6a53f6e90) o (6,4):(_1,6):
  0.00e+00  0.00e+00  0.00e+00  0.00e+00
  0.00e+00  0.00e+00  0.00e+00  0.00e+00
  0.00e+00  0.00e+00  0.00e+00  0.00e+00
  0.00e+00  0.00e+00  0.00e+00  0.00e+00
  0.00e+00  0.00e+00  0.00e+00  0.00e+00
  0.00e+00  0.00e+00  0.00e+00  0.00e+00
*/

// 构造只有形状没有类型的tensor，用于一些特定变换
Tensor tensor = make_identity_tensor(shape);
```

使用Tensor实现Vector Add

```c++
// z = ax + by + c
template<int kNumElmPerThread>
__global__ void vector_add_local_tile_multi_elem_per_thread_half(half* z, int num, const half* x, const half* y, const half a, const half b, const half c) {
    using namespace cute;

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx > num / kNumElmPerThread) return;

    Tensor tz = make_tensor(make_gmem_ptr(z), make_shape(num)); // Global memory tensor for output
    Tensor tx = make_tensor(make_gmem_ptr(x), make_shape(num)); // Global memory tensor for input
    Tensor ty = make_tensor(make_gmem_ptr(y), make_shape(num)); // Global memory tensor for input

    // local tile
    Tensor tzr = local_tile(tz, make_shape(Int<kNumElmPerThread>{}), make_coord(idx));
    Tensor txr = local_tile(tx, make_shape(Int<kNumElmPerThread>{}), make_coord(idx));
    Tensor tyr = local_tile(ty, make_shape(Int<kNumElmPerThread>{}), make_coord(idx));

    Tensor txR = make_tensor_like(txr);
    Tensor tyR = make_tensor_like(tyr);
    Tensor tzR = make_tensor_like(tzr);

    auto tzR2 = recast<half2>(tzR);
    auto txR2 = recast<half2>(txR);
    auto tyR2 = recast<half2>(tyR);

    // LDG.128
    copy(txr, txR);
    copy(tyr, tyR);

    half2 a2 = {a, a};
    half2 b2 = {b, b};
    half2 c2 = {c, c};

    #pragma unroll
    for (int i = 0; i < size(tzR2); ++i) {
        tzR2(i) = txR2(i) * a2 + tyR2(i) * b2 + c2;

    }

    auto tzRx = recast<half>(tzR2);

    copy(tzRx, tzr);
}
```

