# 深度神经网络编译器

> **source from** [**https://microsoft.github.io/AI-System/**](https://microsoft.github.io/AI-System/)**,thanks Microsoft for its contribution to AI education.**

**编译器（compiler）在计算机语言编译中往往指一种计算机程序，它会将某种编程语言写成的源代码（原始语言）转换成另一种编程语言（目标语言），在转换的过程中进行的程序优化就是编译优化过程。**下图展示了经典计算机程序语言中较成功的开源编译框架项目LLVM的编译过程示意图。

<figure><img src="../../.gitbook/assets/5-1-1-compiler.png" alt=""><figcaption></figcaption></figure>

随着深度学习的应用场景的不断泛化，深度学习计算任务也需要部署在不同的计算设备（如服务器，个人电脑，手机，手表，机器人等）和不同的硬件架构上（如X86, ARM, RISC-V等）；同时，实际部署或训练场景对性能往往也有着更为激进的要求，例如一个大规模的上线部署的模型在计算性能上的优化可以直接转换为计算成本的节省，同时，对深度学习任务来说，性能的优化能够为算法提供更大的空间，从而能让算法开发人员在合理的时间内尝试更大或更复杂的模型。这些需求在通用的深度学习计算框架中已经难已得到满足。由于深度学习计算任务在现有的计算框架中往往以DSL（Domain Specific Language）的方式进行编程和表达，这本身使得深度学习计算任务的优化和执行天然符合传统计算机语言的编译和优化过程。因此，与传统程序语言编译器类似，深度神经网络编译器的提出主要是解决多种设备适配性和性能优化的问题。**具体来说，深度神经网络编译就是将当前的深度学习计算任务通过一层或多层中间表达进行翻译和优化，最终转化成目标硬件上的可执行代码的过程。**

​

<figure><img src="../../.gitbook/assets/5-1-2-overview.png" alt="" width="375"><figcaption></figcaption></figure>

上图展示了一个典型的深度神经网络编译器架构图。<mark style="color:red;">与传统编译器类似，深度神经网络编译器也分为前端（Frontend）、后端（Backend）、中间表达（Intermediate Representation, IR）和优化过程（Optimizaiton Pass）等。</mark>

### 前端

深度神经网络编译器的前端一般共享深度学习框架的前端表达，如TensorFlow和PyTorch，即一般为基本Python的DSL（Domain Specific Language）。这样的编程语言中的基本数据结构一般为张量（Tensor），用来描述由基本数据类型（如int, float, string等）元素构成的高维的数组，其元数据可以由一个元素类型和张量形状（如\[128, 512]）来表示。**在张量上进行的基本计算操作称作算子（Operator），通常为一些基本的线性代数计算组成，如矩阵乘法、向量加减乘除等。**下图列举了一些深度学习计算中常用的算子。

<figure><img src="../../.gitbook/assets/5-1-3-op.png" alt="" width="563"><figcaption></figcaption></figure>

|       |             |           |
| ----- | ----------- | --------- |
| Add   | Log         | While     |
| Sub   | MatMul      | Merge     |
| Mul   | Conv        | BroadCast |
| Div   | BatchNorm   | Reduce    |
| Relu  | Loss        | Map       |
| Tanh  | Transpose   | Reshape   |
| Exp   | Concatenate | Select    |
| Floor | Sigmoid     | .....     |

基于张量和基本算子，当前深度学习框架一般利用Python作为胶水语言，把一个深度学习的计算模型描述成一系列算子的操作。下面展示了一个简单的MNIST训练模型在PyTorch中的实现。

```python
 class Net(nn.Module):
   def __init__(self):
     super(Net, self).__init__()
     self.conv1 = nn.Conv2d(1, 32, 3, 1)
     self.conv2 = nn.Conv2d(32, 64, 3, 1)
     self.dropout1 = nn.Dropout2d(0.25)
     self.dropout2 = nn.Dropout2d(0.5)
     self.fc1 = nn.Linear(9216, 128)
     self.fc2 = nn.Linear(128, 10)
 ​
   def forward(self, x):
     x = self.conv1(x)
     x = F.relu(x)
     x = self.conv2(x)
     x = F.relu(x)
     x = F.max_pool2d(x, 2)
     x = self.dropout1(x)
     x = torch.flatten(x, 1)
     x = self.fc1(x)
     x = F.relu(x)
     x = self.dropout2(x)
     x = self.fc2(x)
     output = F.log_softmax(x, dim=1)
     return output
```

### 后端

**深度神经网络编译器的后端指最终变化后的代码要执行的设备或神经网络加速器，目前常见的支持深度学习的计算设备有CPU、GPU、FPGA、TPU等其它专用加速器。**不同的类型的计算设备往往采用完全不同的芯片架构，从而对应的编程模型和优化也完全不同。如CUDA GPU采用的是多个并行的流式处理器（Streaming Multiprocessor）和共享内存的架构，在GPU上执行的代码需要符合SIMT（Single Instruction Multiple Threads）的计算模型；而CPU一般采用的是多核架构，以及多线程模型（如线程池）来实现高性能的计算任务。更进一步，尽管是相同类型的设备，不同的型号都会有不同的硬件参数，如内存大小、算力、带宽等，这些参数都会极大的影响编译优化过程。

### 中间表达

从前端语言到后端代码的编译过程，和传统编译器类似，需要经过若干中间表达（Intermediate Representation, IR）。**目前在神经网络编译器中较为常用的中间表达主要包括计算图（DAG）和算子表达式等。**计算图作为连接深度学习框架和前端语言的主要格式，也是标准化深度学习计算模型的常用格式，如ONNX格式即为一种深度学习模型的标准可交换格式，目前主流框架如TensorFlow和PyTorch的大部分程序都可以被转换或导出成ONNX格式。除此之个，每个DNN编译器也会定义自己的计算图格式，一般这些计算图都可以进行等价互相转换。**计算图的节点是算子，边表示张量，所有的节点和边构成一张有向无坏图，节点之间的依赖关系表示每个算子的执行顺序。**下图为一个简单的计算图示例。

​

<figure><img src="../../.gitbook/assets/5-1-5-dag.png" alt="" width="375"><figcaption></figcaption></figure>

除计算图之外，在将算子继续向下转换到下层并生成设备代码时，算子表达式（Tensor Expression）作为另一类中间表达被广泛使用在不同的神经网络编译器中，如TVM、Ansor、Tensor Comprehension等。**算子表达式的主要作用是描述算子的计算逻辑，从而可以被下层编译器进一步翻译和生成目标设备的可执行代码。**下面代码展示了TVM中的算子表达式形式，其表达了一个矩阵乘法算子的计算逻辑。

```python
 C = t.compute((m, n),
     lambda i, j: t.sum(A[i, k] * B[j * k], axis = k)
```

### 优化过程

最后，深度神经网络编译器的优化过程（Optimization Pass）是构成一个编程器的最核心部分，**是定义在每一种中间表达上的函数，其输入是某一种中间表达，经过一系统优化和变化过程，输出一个新的被优化过后的中间表达。**如在计算图上就有非常多的经典优化过程，如**常数传播、公共子表达式消除**等，这些过程对输入的计算图经过一系列等价变化从而输出新的计算图。也有一些优化过程是将一个高层的中间表达翻译到低层的中间表达，甚至最终的可执行代码。这些**优化过程又可分为设备相关和设备无关的优化**。

### 计算图优化

计算图作为连接深度学习框架和前端语言的主要中间表达，被目前主流框架如TensorFlow和PyTorch所使用或者作为标准文件格式来导出模型。 计算图是一个有向无环图（DAG），节点表示算子，边表示张量或者控制边（control flow），节点之间的依赖关系表示每个算子的执行顺序。

计算图的优化被定义为作为在计算图上的函数，**通过一系列等价或者近似的优化操作将输入的计算图变换为一个新的计算图。其目标是通过这样的图变换来化简计算图，从而降低计算复杂度或内存开销。**在深度神经网络编译器中，有大量优化方法可以被表示为计算图的优化，包括一些在传统程序语言编译器中常用的优化方法。下图列举了一些常见的图优化方法：

​

<figure><img src="../../.gitbook/assets/5-2-1-graph_opt (1).png" alt="" width="563"><figcaption></figcaption></figure>

#### 算术表达式化简

一类最常见的计算图优化就是算术表达式化简，在计算图中的一些子图所对应的算术表达式，在数学上有等价的化简方法来简化表达式，这反应在计算图上就是将子图转化成一个更简单的子图（如更少的节点），从而降低计算量。下图展示了一个利用算术表达式化简计算图的例子，左边的子图包含了两个算法：Const算子（返回元素值为0的常量张量）和Mul算子（计算两个相同形状的算子的元素乘积），通过表达式化简，这个子图可以直接被化简成右边的只包括Const算子的子图。下表i列举了一些常见的算术表达式化简规则，其中X和Y表示张量，0和1表示常量张量，其它操作符均对应张量上的算子。

​

<figure><img src="../../.gitbook/assets/5-2-2-simp.png" alt="" width="563"><figcaption></figcaption></figure>

| 变化前               | 变换后          |
| ----------------- | ------------ |
| X \* 0            | 0            |
| X \* Broadcast(0) | Broadcast(0) |
| X \* 1            | X            |
| X \* Broadcast(1) | X            |
| X + 0             | X            |
| X + Broadcast(0)  | X            |
| Log(Exp(X)/Y)     | X - Log(Y)   |

#### 公共子表达式消除

公共子表达式消除（Common Subexpression Elimination, CSE）也是经典编译优化中常用的优化。其目的是通过找到程序中等价的计算表达式，然后通过复用结果的方式消除其它冗余表达式的计算。同理，在计算图中，公共子表达式消除就等同于寻找并消除冗余的计算子图。**一个简单的实现算法是按照图的拓扑序（保证一个访问一个节点时，其前继节点均已经访问）遍历图中节点，每个节点按照输入张量和节点类型组合作为键值进行缓存，后续如果有节点有相同的键值则可以被消除，并且将其输入边连接到缓存的节点的输入节点上。**下图为一个公共子表达式消除的示例，图左边蓝色椭圆中的节点为不在缓存中的节点，也就是必须要执行的节点，而红色椭圆中的节点的计算和前面蓝色节点的计算重复，也就是其键值可以被缓存到，因此可以安全的消除这些节点，于是最终左边的计算图可以被优化成右边的计算图。

​

<figure><img src="../../.gitbook/assets/5-2-3-cse (1).png" alt="" width="563"><figcaption></figcaption></figure>

考虑到下列代码：

```
   a = b * c + g;
   d = b * c + e;
```

可以观察到 `b * c` 是两项表达式中的公共子表达式。如果计算这个子表达式并将其计算结果存储起来的开销，低于重复计算这个子表达式的开销，则能够将以上代码转换成以下代码：

```
   temp = b * c;
   a = temp + g;
   d = temp + e;
```

#### 常数传播

常数传播（constant propagation）就叫常数折叠（constant folding），也是经典编译优化中的常用优化，**其主要方法是通过在编译期计算出也是常数表达式的值，用计算出的值来替换原来的表达式，从而节省运行时的开销。在计算图中，如果一个节点的所有输入张量都是常数张量的话，那么这个节点就可以在编译期计算出输入张量，并替换为一个新的常数张量。**如下图所示为一个常数传播的示例，其中红色方框内的两个节点都可以被提前计算出来，因此可以在编译期优化掉。值得注意的是，常数传播需要编译器具有计算的能力，甚至对于一些较大的算子还需要能够在加速硬件上（如GPU）上计算，否则优化的过程就会非常的慢。常数传播的优化在深度学习尤其是模型推理的时候非常有用，因为在推理时，模型中的参数张量全部固定为常数张量，大量计算可以在编译期计算好，极大的化简了推理运算时的计算开销。但是，在深度学习的场景中，常数传播有时候也会带来否优化，如增加计算内存甚至计算时间，一个典型的例子就是一个标量常数张量后面跟一个`Broadcast`的算子时，如果做了常数传播就会增加内存占用，如果后面是访存密集型的算子的话，也会增加内存压力，从而增加计算时间。

<figure><img src="../../.gitbook/assets/5-2-4-cf (2).png" alt="" width="563"><figcaption></figcaption></figure>

#### ​矩阵乘自动融合

矩阵乘在深度学习计算图中被广泛应用，如常见的神经网络的线性层、循环神经网络的单元层、注意力机制层等都有大量的矩阵乘法。在同一个网络里，经常会出现形状相同的矩阵乘法，根据一些矩阵的等价规则，如果把些矩阵乘算子融合成一个大的矩阵乘算子，可以更好的利用到`GPU`的算力，从而加速模型计算。下图为其中一种常见的矩阵乘自动融合的示例，其中，如果有两个矩阵乘法共享同一个输入张量（图中方框内左侧），我们就可以自动把其中的两个输入张量拼接成一个大的矩阵乘算子（图中方框内右侧），其计算的结果刚好是原算子计算结果的拼接。利用这种规则，图中最右侧的`GRU`网络中的两组矩阵乘算子可以分别融合成两个大的矩阵乘算子。类似的融合规则还有`BatchMatMul`，可以把两个相同形状的矩阵拼接成一个新的`BatchMatMul`算子。

<figure><img src="../../.gitbook/assets/5-2-5-gemm.png" alt="" width="563"><figcaption></figcaption></figure>

#### 算子融合

上小节中介绍的算子融合方法是针对矩阵乘算子特有的，**在深度学习模型中，针对大量的小算子的融合都可以提高GPU的利用率，减少内核启动开销、减少访存开销等好处。例如，Element-wise的算子（如Add，Mul，Sigmoid，Relu等）其计算量非常小，主要计算瓶颈都在内存的读取和写出上，如果前后的算子能够融合起来，前面算子的计算结果就可以直接被后面算子在寄存器中使用，避免数据在内存的读写，从而提交整体计算效率。**下图展示了一个Mul算子和一个Add算子融合的示例，还有其对应的融合前后的CUDA代码示例，在没有融合前，执行两个算子需要启动两个GPU内核，前一个计算的结果需要写出到主存中，下一个内核计算的时候需要再次读取到计算核上。然后，融合后的代码只需要启动一个内核，并且可以有效复用中间计算结果。

<figure><img src="../../.gitbook/assets/5-2-6-fusion.png" alt="" width="563"><figcaption></figcaption></figure>

```cpp
 //融合前为两个单独内核函数
  __global__ mul(float *x0, float *x1, float *y)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    y[idx] = x0[idx] * x1[idx];
  }
 ​
  __global__ add(float *x0, float *x1, float *y)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    y[idx] = x0[idx] + x1[idx];
  }
```

```cpp
 //融合后为一个单独内核函数
  __global__ fused_muladd(float *x0, float *x1, float *x2, float *y)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    y[idx] = x0[idx] * x1[idx] + x2[idx];
  }
```

#### 子图替换和随机子图替换

鉴于算子融合在深度学习计算中能够带来较好的性能优化，然而在实际的计算图中有太多算子无法做到自动的算子融合，主要原因包括算子的内核实现逻辑不透明、算子之前无法在特有加速器上融合等等。**为了在这些的情况下还能进行优化，用户经常会实现一些手工融合的算子来提升性能。那么，编译器在计算图中识别出一个子图并替换成一个等价的新的算子或子图的过程就是子图替换优化。**下图展示的是基于规则的子图替换示例，需要在系统中注册系列替换规则，如Conv和Relu的子图可以替换为Conv+Relu融合后的算子。

<figure><img src="../../.gitbook/assets/5-2-8-replace.png" alt="" width="563"><figcaption></figcaption></figure>

