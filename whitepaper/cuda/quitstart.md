# 🤑 CPU|GPU程序执行流程

### CUDA和GPU简介

&#x20;CUDA（Compute Unified Device Architecture），是显卡厂商[NVIDIA](https://baike.baidu.com/item/NVIDIA?fromModule=lemma\_inlink)推出的运算平台。 CUDA™是一种由NVIDIA推出 的通用[并行计算](https://baike.baidu.com/item/%E5%B9%B6%E8%A1%8C%E8%AE%A1%E7%AE%97/113443?fromModule=lemma\_inlink)架构，该架构使[GPU](https://baike.baidu.com/item/GPU?fromModule=lemma\_inlink)能够解决复杂的计算问题。 它包含了CUDA[指令集架构（ISA）](https://baike.baidu.com/item/%E6%8C%87%E4%BB%A4%E9%9B%86%E6%9E%B6%E6%9E%84?fromModule=lemma\_inlink)以及GPU内部的并行 计算引擎。 开发人员可以使用[C语言](https://baike.baidu.com/item/C%E8%AF%AD%E8%A8%80?fromModule=lemma\_inlink)来为CUDA™架构编写程序，所编写出的程序可以在支持CUDA™的处理器上以超高 性能运行。CUDA3.0已经开始支持[C++](https://baike.baidu.com/item/C%2B%2B?fromModule=lemma\_inlink)和[FORTRAN](https://baike.baidu.com/item/FORTRAN?fromModule=lemma\_inlink)。&#x20;

GPU（Graphic Processing Unit）：图形处理器，显卡的处理核心。电脑显示器上显示的图像，在显示在显示器上之 前，要经过一些列处理，这个过程有个专有的名词叫“渲染"。以前的计算机上没有GPU，渲染就是CPU负责的。渲染是 个什么操作呢，其实就是做了一系列图形的计算，但这些计算往往非常耗时，占用了CPU的一大部分时间。而CPU还要处 理计算机器许多其他任务。因此就专门针对图形处理的这些操作设计了一种处理器，也就是GPU。这样CPU就可以从繁 重的图形计算中解脱出来。&#x20;

NVIDIA公司在1999年发布Geforce 256图形处理芯片时首先提出GPU的概念，随后大量复杂的应 用需求促使整个产业蓬勃发展至今。&#x20;

最早的GPU是专门为了渲染设计的，那么他也就只能做渲染的那些事情。渲染这个过程具体来说就 是几何点、位置和颜色的计算，这些计算在数学上都是用四维向量和变换矩阵的乘法，因此GPU也 就被设计为专门适合做类似运算的专用处理器了。但随着GPU的发展，GPU的功能也越来越多， 比如现在很多GPU还支持了硬件编解码。

全球GPU巨头：NVIDIA(英伟达)AMD(超威半导体)。

<figure><img src="../../.gitbook/assets/图片 (2) (1) (1) (1) (1) (1) (1) (1) (1).png" alt="" width="563"><figcaption></figcaption></figure>

### GPU与CPU的区别

GPU采用流式并行计算模式，可对每个数据行独立的并行计算。&#x20;

1. CPU基于低延时设计，由运算器（ALU，Arithmetic and Logic Unit 算术逻辑单元）和控制器 （CU,Control Unit），以及若干个寄存器和高速缓冲存储器组成，功能模块较多，擅长逻辑控制，**串行运算。**&#x20;
2. GPU基于大吞吐量设计，拥有更多的ALU用于数据处理，适合对密集数据进行并行处理，擅长大规模并发计算，因此GPU也被应用于AI训练等需要大规模**并发计算**场景。

<figure><img src="../../.gitbook/assets/图片 (76).png" alt="" width="563"><figcaption></figcaption></figure>

### 串行处理与并行处理的区别

<mark style="color:red;">**Sequential processing(串行)**</mark>&#x20;

* 指令/代码块依次执行&#x20;
* 前一条指令执行结束以后才能执行下一条语句&#x20;
* 一般来说,当程序有数据依赖or分支等这些情况下需要串行
* 使用场景: 复杂的逻辑计算(比如:操作系统)

<figure><img src="../../.gitbook/assets/图片 (14).png" alt=""><figcaption></figcaption></figure>

{% hint style="info" %}
data dependency的种类&#x20;

* Flow dependency
* &#x20;Anti dependency&#x20;
* Output dependency&#x20;
* Control dependency
{% endhint %}

看一个简单的案例：statement 2 依赖 statement 1 ，statement 5 依赖 statement 4 ，statement 3 是一个 循环，跟所有的statement没有任何关系 ，有四个core可以使用，如果是串行需要多少个时钟周期呢？

<figure><img src="../../.gitbook/assets/图片 (15).png" alt=""><figcaption></figcaption></figure>

一共用时46个时钟周期，改进下，没有任何依赖关系的statement让他们并行执行：

<figure><img src="../../.gitbook/assets/图片 (7).png" alt="" width="508"><figcaption></figcaption></figure>

statement3的循环可以分割成多个子代码执行(每个子代码快 5cycles)，进一步可以优化：

<figure><img src="../../.gitbook/assets/图片 (65).png" alt="" width="442"><figcaption></figcaption></figure>

statement 5 可以在 statement 4 彻底执行结束前就知道所依赖的结果了

<figure><img src="../../.gitbook/assets/图片 (66).png" alt="" width="380"><figcaption></figcaption></figure>

回顾一下，我们在这里都做了哪些事情&#x20;

* 把没有数据依赖的代码分配到各个core各自执行 (schedule, 调度)&#x20;
* 把一个大的loop循环给分割成多个小代码，分配到各个core执行(loop optimization)&#x20;
* 在一个指令彻底执行完以前，如果已经得到了想要得到的数据，可以提前执行下 一个指令(pipeling, 流水线）&#x20;

我们管这一系列的行为，称作parallelization (并行化)。我们得到的可以充分利用多 核多线程的程序叫做parallelized program(并行程序）

<mark style="color:red;">**Parallel processing(并行)**</mark>&#x20;

* 指令/代码块同时执行&#x20;
* 充分利用multi-core(多核)的特性,多个core一起去完成一个或多个任务
* 使用场景:科学计算,图像处理,深度学习等等

<figure><img src="../../.gitbook/assets/图片 (67).png" alt=""><figcaption></figcaption></figure>

* Loop parallelization&#x20;

大部分消耗时间长的程序中,要不然就是在I/O上的内存读写消耗时间上长,要不然 就是在loop上。针对loop的并行优化是很重要的一个优化策略 在图像处理/深度学习中很多地方都是用到了循环&#x20;

* 比如说:pre/post process (前处理后处理) ，resize, crop, blur, bgr2rgb, rgb2gray, dbscan, findCounters&#x20;
* 比如说:DNN中的卷积(convolution layer)以及全连接层(Fullyconnected layer)

<mark style="color:red;">**容易混淆的几个概念**</mark>

"并行"与"并发"的区别

* 并行(parallel) 物理意义上同时执行
* 并发(concurrent) -逻辑意义上同时执行

<figure><img src="../../.gitbook/assets/图片 (68).png" alt="" width="563"><figcaption></figcaption></figure>

“进程”与“线程”的关系

* 线程是进程的子集。一个进程可以有多个线程

"多核"与"加速比"的关系&#x20;

* 双核的加速不一定就是两倍&#x20;
* 8核的加速比有时会差于4核

<figure><img src="../../.gitbook/assets/图片 (69).png" alt="" width="563"><figcaption></figcaption></figure>

### CPU与GPU在并行处理的优化方向

* CPU: 目标在于减少memory latency&#x20;
* GPU: 目标在于提高throughput

**memory latency是什么？**

CPU/GPU从memory获取数据所需要的等待时间



<figure><img src="../../.gitbook/assets/图片 (70).png" alt="" width="481"><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/图片 (3) (1) (1) (1) (1) (1) (1).png" alt=""><figcaption></figcaption></figure>

`cache miss` 这个时候，CPU core由于没有数据，所以 在等待数据的到来。这个状态叫做`stall`，

如果数据不在cache里，那么就需要往下级 memory中寻找数据，然而访问下级memory是很耗时的。

<figure><img src="../../.gitbook/assets/图片 (1) (1) (1) (1) (1) (1) (1) (1) (1) (1).png" alt=""><figcaption></figcaption></figure>

不同memory在latency上的比较

### CPU是如何进行优化的？

* <mark style="color:red;">`Pipeline`</mark>&#x20;

提高throughput的一种优化

<figure><img src="../../.gitbook/assets/图片 (11).png" alt="" width="563"><figcaption></figcaption></figure>

* IF: `Instruction fetch` 把指令从memory中取出来&#x20;
* ID: `Instruction decode` 把取出来的指令给解码成机器可以识别的信号&#x20;
* OF: `Operand fetch` 把数据从memory中取出来放到register中 &#x20;
* EX: `Execution` 使用ALU(负责运算的unit)来进行计算
* WB: `write back` 把计算完的结果写回register

<figure><img src="../../.gitbook/assets/图片 (13).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/图片 (75).png" alt="" width="563"><figcaption></figcaption></figure>

* `cache hierarchy` (多级缓存)：减少`memory latency`的一种优化
  * L1: 逻辑核独占&#x20;
  * L2: 物理核独占&#x20;
  * L3: 所有物理核共享
* `Pre-fetch` :减少memory latency的一种优化

<figure><img src="../../.gitbook/assets/图片 (73).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/图片 (74).png" alt="" width="563"><figcaption></figcaption></figure>

{% hint style="info" %}
* 当程序中出现 branch的话，如何进行pre-fetch?

如果预先知道计算A结束后就执行计算B的话，可以预先(pre)把计算B所需要的指令和 数据取出来(fetch)，放入到cache中 (hiding memory latency
{% endhint %}

* `Branch-prediction`(分支预测)

简单来说，就是根据以往的branch得走向，去预测下一次branch的走向，并提前执行了相应的指令,CPU硬件上有专门负责分支预测的Unit

```c
while(i < 100) {
    sum += a[i];
    i ++;
}
```

<figure><img src="../../.gitbook/assets/图片 (9).png" alt=""><figcaption></figcaption></figure>

因为循环判断条件一直都是true，所以预测下一次 循环也是true，那么先把数据取出来进行 pipeline。如果预测失败，rollback回去就好了。

* `Multi-threading`

充分利用计算资源的一种技术 ，让因为数据依赖或者`cache miss`而`stall`的`core`去做一些其他的事情 ，提高throughput的一种技术。

由于CPU处理的大多数都是一些复杂逻辑的计算，有大量的分支以及难以预测的分支方向，所以增加core的数 量，增加线程数而带来的throughput的收益往往并不是那么高。

去掉复杂的逻辑计算，去掉分支，把大量的简单运算放在一起的话，是不是就可以最大化的提高throughput呢？ 答案是yes，这个就是GPU所做的事情。

### 基础GPU架构

<figure><img src="../../.gitbook/assets/图片 (77).png" alt=""><figcaption></figcaption></figure>

GPU为图形图像专门设计，在矩阵运算，数值计算方面具有独特优势，特别是浮点和并行计算上能 优于CPU的数十数百倍的性能。 GPU的优势在于快，而不是效果好。 比如用美团软件给一张图要加上模糊效果，CPU处理的时候从左到右从上到下进行处理。可以考虑 开多核，但是核数毕竟有限制，比如4核、8核 分块处理。

<figure><img src="../../.gitbook/assets/图片 (78).png" alt="" width="563"><figcaption></figcaption></figure>

使用GPU进行处理，因为分块之前没有相互的关联关系，可以通过GPU并行处理，就不单只是4、 8分块了，可以切换更多的块，比如16、64等

### GPU的特点

* `multi-threading`技术&#x20;

大量的core，可以支持大量的线程 ，CPU并行处理的threads数量规模：数十个， GPU并行处理的threads数量规模：数千个

* 每一个`core`负责的运算逻辑十分`simple`

CUDA core: D = A \* B + C&#x20;

Tensor core:&#x20;

* 4x4x4的matrix calculation&#x20;
* D = A \* B + C

<figure><img src="../../.gitbook/assets/1 HSAoDDRGYzf8Tecj79RP0A.webp" alt=""><figcaption></figcaption></figure>



<figure><img src="../../.gitbook/assets/0 am1WNF8XjEAA-WVe.webp" alt="" width="509"><figcaption></figcaption></figure>

* `SIMT`

类似于SIMD的一种概念 ，将一条指令分给大量的thread去执行，thread间的调度是由warp来负责管理 • GPU体系架构中有一个warp schedular，专门负责管理线程调度的。Warp schedular是GPU体系 架构中特有的概念，CPU中 没有这个。

<figure><img src="../../.gitbook/assets/图片 (4) (1) (1) (1) (1) (1) (1).png" alt=""><figcaption></figcaption></figure>

* 由于`throughput`非常的高，所以相比与`CPU`，`cache miss`所产生的`latency`对性能的影响比较小&#x20;
* `GPU`主要负责的任务是大规模计算(图像处理、深度学习等等)，所以一旦`fetch`好了数据以后，就会一直连续 处理，并且很少`cache miss`

<figure><img src="../../.gitbook/assets/图片 (6) (1).png" alt="" width="563"><figcaption></figcaption></figure>

### summary

* `CPU`:&#x20;
  * 适合复杂逻辑的运算&#x20;
  * 优化方向在于减少`memory latency`&#x20;
  * 相关的技术有,`cache hierarchy`, `pre-fetch`, `branch-prediction`,`multi-threading`&#x20;
  * 不同于GPU,CPU硬件上有复杂的分支预测器去实现`branch-prediction`&#x20;
  * 由于CPU经常处理复杂的逻辑,过大的增大`core`的数量并不能很好的提高`throughput`&#x20;
* `GPU`:&#x20;
  * 适合简单单一的大规模运算。比如说科学计算,图像处理,深度学习等等&#x20;
  * 优化方向在于提高`throughput`&#x20;
  * 相关的技术有,`multi-threading`,`warp schedular`&#x20;
  * 不同于CPU,GPU硬件上有复杂的`warp schedular`去实现多线程的`multi-threading`&#x20;
  * 由于GPU经常处理大规模运算,所以在throughput很高的情况下,GPU内部的`memory latency`上带来的性 能损失不是那么明显&#x20;
  * 然而CPU和GPU间通信时所产生的`memory latency`需要重视

### reference

* [https://www.zhihu.com/question/35063258](https://www.zhihu.com/question/35063258)
* [https://lwn.net/Articles/252125/](https://lwn.net/Articles/252125/)
* [https://medium.com/codex/understanding-the-architecture-of-a-gpu-d5d2d2e8978b](https://medium.com/codex/understanding-the-architecture-of-a-gpu-d5d2d2e8978b)
* [https://arxiv.org/pdf/1603.07285.pdf](https://arxiv.org/pdf/1603.07285.pdf)
* [https://pdf.dfcfw.com/pdf/H3\_AP202111101528199831\_1.pdf?1636553716000.pdf](https://pdf.dfcfw.com/pdf/H3\_AP202111101528199831\_1.pdf?1636553716000.pdf)
