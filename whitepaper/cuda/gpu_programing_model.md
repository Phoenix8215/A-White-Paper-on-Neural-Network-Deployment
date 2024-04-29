# 🤒 GPU编程模型

### GPU编程模型

软件层面上不管什么计算设备，大部分异构计算都会分成主机代码和设备代码。整体思考过程就是 应用分析、内存资源分配、线程资源分配再到具体核函数的实现。&#x20;

CUDA中线程也可以分成三个层次：线程、线程块和线程网络。&#x20;

* 线程是CUDA中基本执行单元，由硬件支持、开销很小，每个线程执行相同代码；&#x20;
* 线程块（Block）是若干线程的分组，Block内一个块至多512个线程、或1024个线程（根据不 同的GPU规格），线程块可以是一维、二维或者三维的；&#x20;
* 线程网络（Grid）是若干线程块的网格，Grid是一维和二维的。 线程用ID索引，线程块内用局部ID标记threadID，配合blockDim和blockID可以计算出全局ID，用 于SIMT（`Single Instruction Multiple Thread`单指令多线程）分配任务。

<figure><img src="../../.gitbook/assets/图片 (1) (1) (1) (1).png" alt=""><figcaption></figcaption></figure>

首先需要关注的是具体线程数量的划分，在并行计算部分里也提到数据划分和指令划分的概念， GPU有很多线程，在CUDA里被称为thread，同时我们会把一组thread归为一个block，而block又 会被组织成一个grid。

假如我们要对一个长度为1024的数组做reduce\_sum（减少和求和），恰好我们正好有1024个 thread，此时直接一一对应就行，但如果是一张很大的图片呢？ 如果有很多核函数要处理不同的数据呢？GPU上有很多thread，但要完全和实际应用中需要处理的 数据大小完全匹配是不可能的，事实上在满足规定的情况下我们可以给一个block内部分配很多 thread，对于到硬件上也真的是相应数量的thread会自动归为一组直接在一个SM上实行吗？答案 当然不是，此时我们就要关注硬件，引入了wrap概念，GPU上有很多计算核心也就是Streaming Multiprocessor (SM)，在具体的硬件执行中，一个SM会同时执行一组线程，在CUDA里叫warp， 我们不用拘泥于称呼，直接可以理解这组硬件线程会在这个SM上同时执行一部分指令，这一组的 数量一般为32或者64个线程。一个block会被绑定到一个SM上，即使这个block内部可能有1024 个线程，但这些线程组会被相应的调度器来进行调度，在逻辑层面上我们可以认为1024个线程同 时执行，但实际上在硬件上是一组线程同时执行，这一点其实就和操作系统的线程调度一样。 意 思就是假如一个SM同时能执行64个线程，但一个block有1024个线程，那这1024个线程是分 1024/64=16次执行。

解释完了执行层面，再来分析一下内存层面上的对应，一个block不光要绑定在一个SM上，同时一 个block内的thread是共享一块share memory（一般就是SM的一级缓存，越靠近SM的内存就越快 ）。GPU和CPU也一样有着多级cache还有寄存器的架构，把全局内存的数据加载到共享内存上再 去处理可以有效的加速。所以结合具体的硬件具体的参数（SM和寄存器数量、缓存大小等）做出 合适的划分，确保最大化的利用各种资源（计算、内存、带宽）是做异构计算的核心。

<figure><img src="../../.gitbook/assets/图片 (2) (1) (1).png" alt=""><figcaption></figcaption></figure>

GPU在管理线程(thread)的时候是以block(线程块)为单元调度到SM上执行。每个block中以warp(一般32个线程或64线 程)作为一次执行的单位(真正的同时执行)。 1.\
2.\
一个 GPU 包含多个 Streaming Multiprocessor ，而每个 Streaming Multiprocessor 又包含多个 core 。 Streaming Multiprocessors 支持并发执行多达几百的 thread 。 一个 thread block 只能调度到一个 Streaming Multiprocessor 上运行，直到 thread block 运行完毕。一个 Streaming Multiprocessor 可以同时运行多个thread block （因为有多个core）。 通俗点讲：stream multiprocessor(SM)是一块硬件，包含了固定数量的运算单元，寄存器和缓存。 写cuda kernel的时候，跟SM对应的概念是block，每一个block会被调度到某个SM执行，一个SM可以执行多个block。 你的cuda程序就是很多的blocks(一般来说越多越好)均匀的喂给这80个SM来调度执行。具体每个block喂给哪个SM你没 法控制。 不同的GPU规格参数也不一样，比如 Fermi 架构（2010年的比较老）: 每一个SM上最多同时执行8个block。(不管block大小) 每一个SM上最多同时执行48个warp。 每一个SM上最多同时执行48\*32=1,536个线程。 当warp访问内存的时候，processor(处理器)会做context switch(上下文切换)，让其他warp使用硬件资源。因为是硬件 来做，所以速度非常快。

### 软件和硬件的对应关系

GPU在管理线程(thread)的时候是以block(线程块)为单元调度到SM上执行。每个block中以warp(一般32个线程或64线 程)作为一次执行的单位(真正的同时执行)。

1. 一个 GPU 包含多个 Streaming Multiprocessor ，而每个 Streaming Multiprocessor 又包含多个 core 。 Streaming Multiprocessors 支持并发执行多达几百的 thread 。&#x20;
2. 一个 thread block 只能调度到一个 Streaming Multiprocessor 上运行，直到 thread block 运行完毕。一个 Streaming Multiprocessor 可以同时运行多个thread block （因为有多个core）。&#x20;

通俗点讲：stream multiprocessor(SM)是一块硬件，包含了固定数量的运算单元，寄存器和缓存。 写cuda kernel的时候，跟SM对应的概念是block，每一个block会被调度到某个SM执行，一个SM可以执行多个block。 你的cuda程序就是很多的blocks(一般来说越多越好)均匀的喂给这80个SM来调度执行。具体每个block喂给哪个SM你没 法控制。&#x20;

不同的GPU规格参数也不一样，比如 Fermi 架构（2010年的比较老）:&#x20;

* 每一个SM上最多同时执行8个block。(不管block大小)&#x20;
* 每一个SM上最多同时执行48个warp。&#x20;
* 每一个SM上最多同时执行48\*32=1,536个线程。&#x20;

当warp访问内存的时候，processor(处理器)会做context switch(上下文切换)，让其他warp使用硬件资源。因为是硬件 来做，所以速度非常快。

<figure><img src="../../.gitbook/assets/图片 (5).png" alt="" width="563"><figcaption></figcaption></figure>



















