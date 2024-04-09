# Tensor Core VS CUDA Core

### Tensor core和cuda core 的概念

tensor core和cuda core 都是运算单元，是硬件名词，其主要的差异是算力和运算场景。

* 场景：`cuda core`是全能通吃型的浮点运算单元，`tensor core`专门为深度学习矩阵运算设计。
* 算力：在高精度矩阵运算上 tensor cores吊打cuda cores。

#### cuda core

回顾一下NVIDIA的显卡架构出道顺序：

Tesla1.0 （2006年, 代表GeForce8800) -> Tesla2.0 (GT200) -> Fermi(算力可以支撑深度学习啦) -> Kepler(core增长) -> Maxwell（core继续增长） -> Pascal（算力提升）-> Volta(第一代tensor core) -> Turning(第二代 tensor core) -> Ampere(第三代tensor core)。

**cuda core的结构**

<figure><img src="../../.gitbook/assets/图片.png" alt=""><figcaption></figcaption></figure>

一个cuda core 包含了一个整数运算单元integer arithmetic logic unit (ALU) 和一个浮点运算单元floating point unit (FPU)。然后，这个core能进行一种fused multiply-add (FMA)的操作，通俗一点就是一个加乘操作的融合。特点：在不掉精度的情况下，<mark style="color:red;">单指令完成乘加操作</mark>，并且这个是支持32-bit精度。更通俗一点，就是深度学习里面的操作变快了。

\
\
作者：kaiyuan\
链接：https://www.zhihu.com/question/451127498/answer/1813864500\
来源：知乎\
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

Tesla V100 的张量核是可编程矩阵乘积单元，可为训练和推理应用提供高达 125 Tensor TFLOPS 的性能。Tesla V100 GPU 包含 640 个张量核心：每个 SM 8 个。张量内核及其相关数据路径是定制的，可显著提高浮点计算吞吐量，而面积和功耗成本却不高。时钟门被广泛用于最大限度地节省功耗。

