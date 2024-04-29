# 🤒 GPU编程模型

### GPU编程模型

软件层面上不管什么计算设备，大部分异构计算都会分成主机代码和设备代码。整体思考过程就是 应用分析、内存资源分配、线程资源分配再到具体核函数的实现。&#x20;

CUDA中线程也可以分成三个层次：线程、线程块和线程网络。&#x20;

### thread,block,grid

* 线程是CUDA中基本执行单元，由硬件支持、开销很小，每个线程执行相同代码；&#x20;
* 线程块（Block）是若干线程的分组，Block内一个块至多512个线程、或1024个线程（根据不 同的GPU规格），线程块可以是一维、二维或者三维的；&#x20;
* 线程网络（Grid）是若干线程块的网格，Grid是一维和二维的。 线程用ID索引，线程块内用局部ID标记threadID，配合blockDim和blockID可以计算出全局ID，用 于SIMT（`Single Instruction Multiple Thread`单指令多线程）分配任务。

<figure><img src="../../.gitbook/assets/图片 (1) (1) (1) (1) (1).png" alt=""><figcaption></figcaption></figure>

首先需要关注的是具体线程数量的划分，在并行计算部分里也提到数据划分和指令划分的概念， GPU有很多线程，在CUDA里被称为thread，同时我们会把一组thread归为一个block，而block又 会被组织成一个grid。

### Warp

SM采用的SIMT(Single-Instruction, Multiple-Thread，单指令多线程)架构，warp(线程束)是最基本的执行单元，一个warp包含32个并行thread，这些thread以不同数据资源执行相同的指令。

当一个kernel被执行时，grid中的线程块被分配到SM上，一个线程块的thread只能在一个SM上调度，SM一般可以调度多个线程块，大量的thread可能被分到不同的SM上。每个thread拥有它自己的程序计数器和状态寄存器，并且用该线程自己的数据执行指令，这就是所谓的Single Instruction Multiple Thread(SIMT)。

一个CUDA core可以执行一个thread，一个SM的CUDA core会分成几个warp（即CUDA core在SM中分组)，由warp scheduler负责调度。尽管warp中的线程从同一程序地址，但可能具有不同的行为，比如分支结构，因为GPU规定warp中所有线程在同一周期执行相同的指令，warp发散会导致性能下降。一个SM同时并发的warp是有限的，因为资源限制，SM要为每个线程块分配共享内存，而也要为每个线程束中的线程分配独立的寄存器，所以SM的配置会影响其所支持的线程块和warp并发数量。每个block的warp数量可以由下面的公式计算获得：

$$
WarpPerBlock=ceil(\frac{ThreadsPerBlock}{WarpSize})
$$

一个warp中的线程必然在同一个block中，如果block所含线程数目不是warp大小的整数倍，那么多出的那些thread所在的warp中，会剩余一些inactive的thread，需要注意的是，即使这部分thread是inactive的，也会消耗SM资源。由于warp的大小一般为32，所以block所含的thread的大小一般要设置为32的倍数。

<figure><img src="../../.gitbook/assets/图片 (2) (1) (1) (1).png" alt=""><figcaption></figcaption></figure>

### 软件和硬件的对应关系

GPU在管理线程(thread)的时候是以block(线程块)为单元调度到SM上执行。每个block中以warp(一般32个线程或64线 程)作为一次执行的单位(真正的同时执行)。

1. 一个 GPU 包含多个 Streaming Multiprocessor ，而每个 Streaming Multiprocessor 又包含多个 core 。 Streaming Multiprocessors 支持并发执行多达几百的 thread 。&#x20;
2. 一个 thread block 只能调度到一个 Streaming Multiprocessor 上运行，直到 thread block 运行完毕。一个 Streaming Multiprocessor 可以同时运行多个thread block （因为有多个core）。&#x20;

通俗点讲：stream multiprocessor(SM)是一块硬件，包含了固定数量的运算单元，寄存器和缓存。 写cuda kernel的时候，跟SM对应的概念是block，每一个block会被调度到某个SM执行，一个SM可以执行多个block。 你的cuda程序就是很多的blocks(一般来说越多越好)均匀的喂给这80个SM来调度执行。具体每个block喂给哪个SM你没 法控制。&#x20;

不同的GPU规格参数也不一样，比如 Fermi 架构（2010年的比较老）:&#x20;

* 每一个SM上最多同时执行8个block。(不管block大小)&#x20;
* 每一个SM上最多同时执行48个warp。&#x20;
* 每一个SM上最多同时执行48\*32=1,536个线程。&#x20;

当warp访问内存的时候，processor(处理器)会做context switch(上下文切换)，让其他warp使用硬件资源。因为是硬件来做，所以速度非常快。

<figure><img src="../../.gitbook/assets/图片 (5) (1).png" alt="" width="563"><figcaption></figcaption></figure>

### 网格（Grid）、线程块（Block）和线程（Thread）的组织关系以及线程索引的 计算公式

#### 格（Grid）、线程块（Block）和线程（Thread）的组织关系&#x20;

CUDA的软件架构由网格（Grid）、线程块（Block）和线程（Thread）组成，相当于把GPU上的计算单元分为若干 （2\~3）个网格，每个网格内包含若干（65535）个线程块，每个线程块包含若干（512/1024）个线程，三者的关系如 下图：

<figure><img src="../../.gitbook/assets/图片.png" alt="" width="375"><figcaption></figcaption></figure>

Thread，block，grid是CUDA编程上的概念，为了方便程序员软件设计，组织线程。&#x20;

* thread：一个CUDA的并行程序会被以许多个threads来执行。&#x20;
* block：数个threads会被群组成一个block，同一个block中的threads可以同步，也可以通过shared memory通信。
* grid：多个blocks则会再构成grid。

### 网格（Grid）、线程块（Block）和线程（Thread）的最大数量

CUDA中可以创建的网格数量跟GPU的计算能力有关，可创建的Grid、Block和Thread的最大数量参看以下表格：

<figure><img src="../../.gitbook/assets/图片 (1).png" alt=""><figcaption></figcaption></figure>

### CUDA内核函数和配置&#x20;

主机调用设备代码的唯一接口就是Kernel函数，使用限定符:`__global__`。&#x20;

调用内核函数需要在内核函数名后添加`<<<>>>`指定内核函数配置， `<<<>>>`运算符完整的执行配置参数形式是`<<<Dg, Db, Ns, S>>>`&#x20;

* 参数Dg用于定义整个grid的维度和尺寸，即一个grid有多少个block。为dim3类型。Dim3 Dg(Dg.x, Dg.y, 1)表示grid 中每行有Dg.x个block，每列有Dg.y个block，第三维恒为1(目前一个核函数只有一个grid)。整个grid中共有Dg.x\*Dg. y个block，其中Dg.x和Dg.y最大值为65535。
* 参数Db用于定义一个block的维度和尺寸，即一个block有多少个thread。为dim3类型。Dim3 Db(Db.x, Db.y, Db.z) 表示整个block中每行有Db.x个thread，每列有Db.y个thread，高度为Db.z。Db.x和Db.y最大值为512，Db.z最大值 为62。 一个block中共有Db.x_Db.y_Db.z个thread。计算能力为1.0,1.1的硬件该乘积的最大值为768，计算能力为 1.2,1.3的硬件支持的最大值为1024。
* 参数Ns是一个可选参数，用于设置每个block除了静态分配的shared Memory以外，最多能动态分配的shared memory大小，单位为byte。不需要动态分配时该值为0或省略不写。
* 参数S是一个cudaStream\_t类型的可选参数，初始值为零，表示该核函数处在哪个流之中。

### CUDA限定符

函数限定符：

<figure><img src="../../.gitbook/assets/图片 (3).png" alt=""><figcaption></figcaption></figure>

变量限定符：

<figure><img src="../../.gitbook/assets/图片 (4).png" alt=""><figcaption></figcaption></figure>

### 同步&#x20;

* CPU启动kernel函数是异步的，它并不会阻塞等到GPU执行完kernel函数才执行后面的CPU部 分，因此如果后续程序立即需要用到上一个kernel函数的结果我们需要显式设置同步障来阻塞 CPU程序。
* 一个线程块内需要同步共享存储器的共享变量（`__shared__`）时，需要在使用前显式调用 `__syncthreads()`同步块内所有线程。
* <mark style="color:red;">同一个Grid中不同Block之间无法设置同步。</mark>

### CUDA运行时API&#x20;

这里介绍最基础的内存管理函数，其他详见官网

#### cudaMemcpy

```c
 __host__ cudaError_t cudaMemcpy( void* dst, const void* src, size_t count, cudaMemcpyKind kind )
```

用于在主机和设备之间拷贝数据，其中cudaMemcpyKind枚举类型常用有 cudaMemcpyHostToDevice表示把主机数据拷贝到设备内存以及 cudaMemcpyDeviceToHost 把设备内存数据拷贝到主机上。

#### cudaMalloc

```c
 __host__ __device__ cudaError_t cudaMalloc( void** devPtr, size_t size )
```

在设备上分配动态内存，两个限定符表示既可以在主机上也可以在设备上执行或者调用。

#### cudaFree

```c
 __host__ __device__ cudaError_t cudaFree( void* devPtr )
```

释放回收在设备上分配动态内存，两个限定符表示既可以在主机上也可以在设备上执行或者调用。

### 线程索引的计算公式

一个Grid可以包含多个Blocks，Blocks的组织方式可以是一维的，二维或者三维的。block包含多个Threads，这些 Threads的组织方式也可以是一维，二维或者三维的。 CUDA中每一个线程都有一个唯一的标识ID(ThreadIdx)，这个ID随着Grid和Block的划分方式的不同而变化，这里给出 Grid和Block不同划分方式下线程索引ID的计算公式。&#x20;

* threadIdx是一个uint3类型，表示一个线程的索引。&#x20;
* blockIdx是一个uint3类型，表示一个线程块的索引，一个线程块中通常有多个线程。&#x20;
* blockDim是一个dim3类型，表示线程块的大小。&#x20;
* gridDim是一个dim3类型，表示网格的大小，一个网格中通常有多个线程块。&#x20;

下面这张图比较清晰的表示的几个概念的关系：

<figure><img src="../../.gitbook/assets/图片 (2).png" alt="" width="375"><figcaption></figcaption></figure>

<mark style="color:red;">占坑索引方式</mark>







### reference

* [https://zhuanlan.zhihu.com/p/123170285](https://zhuanlan.zhihu.com/p/123170285)
* [https://www.cnblogs.com/dama116/p/6909629.html](https://www.cnblogs.com/dama116/p/6909629.html)
