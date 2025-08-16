cute的核心抽象是layout，是从坐标空间到索引空间的映射

`Layout`提供了多维数组访问的通用接口，抽象了数组元素在内存中的组织方式



# 基本概念

## Integer

Cute充分利用了动态整数和静态整数

* Dynamic integers(或者说运行时)只是普通的整数类型，比如int、size_t、uint16_t
* Static integers(编码时整数)，是 `std：：integral_constant<Value>` 等类型的实例化。这些类型将值编码为`静态 constexpr` 成员。它们还支持强制转换为其底层动态类型，因此它们可以在具有动态整数的表达式中使用。在Cute中，定义了自己的 CUDA 兼容静态整数类型 `cute：：C<Value>` 以及重载的数学运算符，以便静态整数的数学运算产生静态整数。CuTe 将快捷方式别名 `Int<1>`、`Int<2>`、`Int<3>` 和 `_1`、`_2`、`_3` 定义为方便

处理整数的一些方法：

* cute::is_integral<T> & cute::is_std_integral<T>： 检查是动态整数还是静态整数
* cute::is_static<T> 检查 `T` 是否为空类型（因此实例化不能依赖于任何动态信息）。相当于 `std::is_empty`
* cute::is_constant<N,T> 检查 `T` 是否为静态整数，其值是否等于 `N`



# Tuple

tuple是是零个或多个元素的有限有序列表，cute::tuple类似于std::tuple



# IntTuple

CuTe 将 IntTuple 概念定义为整数或 IntTuples 的元组。请注意递归定义。应该就是tuple可以嵌套

一些例子

```c++
int{2}   // dynamic int
Int<3>{} // static int
make_tuple(int{2}, Int<3>{})
make_tuple(uint16_t{42}, make_tuple(Int<1>{}, int32_t{3}), Int<17>{})
```

对`IntTuple`的操作

* rank(IntTuple)，IntTuple` 中的元素数。单个整数的秩为 1，元组的秩为 `tuple_size
* get<I>(IntTuple) 获得IntTuple中的第I个元素
* depth(IntTuple)，是指嵌套的深度，
* size(IntTuple)，所有元素的**乘积**，
* 使用()表示层级，例如`(3,(6,2),8)`



# Shapes and Strides

这两个都是IntTuple的概念



# Layout

是`(Shape, Stride)`的Tuple，语义上来说，通过Stride实现从坐标到索引的映射，或者说从物理位置到逻辑位置的映射

# Tensor

可以简单理解为数据+布局



# 创建和使用布局

* rank(Layout) 相当于Layout中Shape的元组大小
* get<I>(Layout) 获得第I个子Layout
* depth(Layout) Layout的深度
* shape(Layout)
* stride(Layout)
* size(Layout)相当于size(shape(Layout))，就是数据的元素个数？
* cosize(Layout)  不太理解

# 分层访问

`IntTuple` 和 `Layout` 可以任意嵌套，为了方便，cute定义了函数用于访问嵌套的`IntTuple` 和 `Layout`。

* get<I0,I1,...,IN>(x) := get<IN>(...(get<I1>(get<I0>(x)))...) 提取 `IN`th 的 ...`x` 的第 `I0` 个元素的 `I1`st 的  **不理解**
* rank<I...>(x) := rank(get<I...>(x))
* depth<I...>(x) := depth(get<I...>(x))
* shape<I...>(x) := shape(get<I...>(x))
* size<I...>(x) := size(get<I...>(x))



# 构建布局

创建Layout有很多方法

```c++
Layout s8 = make_layout(Int<8>{});      // _8:_1
Layout d8 = make_layout(8);             // 8:_1
Layout s2xs4 = make_layout(make_shape(Int<2>{},Int<4>{}));  // (_2,_4):(_1,_2)
Layout s2xd4 = make_layout(make_shape(Int<2>{},4)); // (_2,4):(_1,_2)

Layout s2xd4_a = make_layout(make_shape (Int< 2>{},4),
                            make_stride(Int<12>{},Int<1>{}));   // (_2,4):(_12,_1)
Layout s2xd4_col = make_layout(make_shape(Int<2>{},4),
                            LayoutLeft{});                      // (_2,4):(_1,_2)
Layout s2xd4_row = make_layout(make_shape(Int<2>{},4),
                            LayoutRight{});         // (_2,4):(4,_1)

Layout s2xh4 = make_layout(make_shape (2,make_shape (2,2)),
                        make_stride(4,make_stride(2,1)));   // (2,(2,2)):(4,(2,1))
Layout s2xh4_col = make_layout(shape(s2xh4),
                            LayoutLeft{});   // (2,(2,2)):(_1,(2,4))    
```

由上面的例子可知，`LayoutLeft{}`对应的是左边维度变快，所以是列主序，而`LayoutRight{}`是行主序，如果不指定stride，默认是列主序。

后面两个例子是带有嵌套的，就是之前看过的分层布局的情况。



# 使用Layout

查看映射，打印Layout对象中每个元素的内存偏移索引

```c++
template <class Shape, class Stride>
void print2D(Layout<Shape,Stride> const& layout)
{
  for (int m = 0; m < size<0>(layout); ++m) {
    for (int n = 0; n < size<1>(layout); ++n) {
      printf("%3d  ", layout(m,n));
    }
    printf("\n");
  }
}

print2D(s2xs4);
/*
  0    2    4    6  
  1    3    5    7 
*/
print2D(s2xd4);
/*
  0    2    4    6  
  1    3    5    7 
*/

print2D(s2xd4_a);
/*
  0    1    2    3  
 12   13   14   15  
*/

print2D(s2xd4_col);
/*
  0    2    4    6  
  1    3    5    7 
*/
print2D(s2xd4_row);
/*
  0    1    2    3  
  4    5    6    7 
*/
// 这个不太好理解了
print2D(s2xh4);
/*
  0    2    1    3  
  4    6    5    7
*/
// 这个好理解一点
print2D(s2xh4_col);
/*
  0    2    4    6  
  1    3    5    7  
*/

// 可以使用print_layout, 会同时打印布局和可视化布局
/*
print_layout(s2xh4)
(2,(2,2)):(4,(2,1))
      0   1   2   3 
    +---+---+---+---+
 0  | 0 | 2 | 1 | 3 |
    +---+---+---+---+
 1  | 4 | 6 | 5 | 7 |
    +---+---+---+---+
*/
// 生成可视化布局的letax表示，不太会用
print_latex(s2xh4);
```

# 矢量布局 Vector Layouts

将向量定义为`任何 rank == 1` 的 `Layout`。例如，布局 `8：1` 可以解释为索引连续的 8 元素向量。

主要帮助理解上面的嵌套的情况。

从最简单的开始`_8:_1`可以理解为索引连续的8元素向量

```bash
Layout:  8:1
Coord :  0  1  2  3  4  5  6  7
Index :  0  1  2  3  4  5  6  7
```

复杂一点`_8:_2`理解为8元素向量，但是跨度是2了

```bash
Layout:  8:2
Coord :  0  1  2  3  4  5  6  7
Index :  0  2  4  6  8 10 12 14
```

再复杂一点，将`(4,2):(2,1)`也看成向量。前4个元素跨度是2，所以是0，2，4，6，然后每个元素中的两个跨度是1，所以如下

```bash
Layout:  ((4,2)):((2,1))
Coord :  0  1  2  3  4  5  6  7
Index :  0  2  4  6  1  3  5  7
```

同理，`(4,2):(1,4)`，4个元素跨度1，所以是0，1，2，3，然后每个元素中的两个元素跨度4，所以是4，5，6，7

```bash
Layout:  ((4,2)):((1,4))
Coord :  0  1  2  3  4  5  6  7
Index :  0  1  2  3  4  5  6  7
```

***那怎么理解`((2,2),2):((4,1),2)`?***

最外层是4行两列的，相邻两列步幅为2

内层步幅是4，外层步幅是1

```bash
0  2
4  5
1  3
5  7
```



# Layout Concepts

Loyout接受坐标并完成坐标到索引的映射关系

## 兼容性

两个布局兼容是指两个布局的形状兼容

* A和B的size相同(乘积)
* A的所有坐标都是B的有效坐标

举例

* Shape 24 & Shape 32不兼容
* Shape 24 & Shape(4,6)兼容
* Shape(4,6)和Shape((2,2),6)兼容
* Shape ((2,3),4) is NOT compatible with Shape ((2,2),(3,2))
* Shape ((2,2),(3,2)) is NOT compatible with Shape ((2,3),4).
* Shape 24 is compatible with Shape (24).
* Shape (24) is NOT compatible with Shape 24.

简单理解就是，对于每一个层级，都要满足A可以由B的乘积表示，这样A就兼容B

**非对称，没有传递性**



## 布局坐标(Layouts Coordinates)

每个布局都可以接受多种坐标，每个布局接受与其兼容的任何shape的坐标。CuTe 通过共列顺序提供这些坐标集之间的映射(不太好理解)

两个基本映射

* 通过 `Shape` 从输入坐标到相应自然坐标的map
* 通过`Stride`从自然坐标到索引的map

### 坐标映射（Coordinate Mapping）

从输入坐标到自然坐标的映射是在 `Shape` 中应用并列顺序（从右到左阅读，而不是从左到右读取的“字典顺序”）。

Take the shape `(3,(2,3))`，此形状具有三个坐标集：一维坐标、二维坐标和自然 （h-D） 坐标

**不是太好理解**

![image-20250804005553346](C:\Users\Qufanhua\AppData\Roaming\Typora\typora-user-images\image-20250804005553346.png)

对于上面的每个坐标而言，都有两个等效坐标，并且映射到相同的自然坐标，也就是说对于同样的输入而言，可以表示成一维数组，二维数组或者是自然坐标。

cute::idx2crd(idx, shape)负责进行坐标映射

```c++
auto shape = Shape<_3,Shape<_2,_3>>{};// 嵌套shape = (3,(2,3))⇒总元素数量 = 3 × 2 × 3 = 18
// 将线性索引 16 映射为嵌套坐标 (i, (j, k))
print(idx2crd(   16, shape));                                // (1,(1,2))
print(idx2crd(_16{}, shape));                                // (_1,(_1,_2))
print(idx2crd(make_coord(   1,5), shape));                   // (1,(1,2))
print(idx2crd(make_coord(_1{},5), shape));                   // (_1,(1,2))
print(idx2crd(make_coord(   1,make_coord(1,   2)), shape));  // (1,(1,2))
print(idx2crd(make_coord(_1{},make_coord(1,_2{})), shape));  // (_1,(1,_2))
```

**理解不了，剩下一点先跳过**
