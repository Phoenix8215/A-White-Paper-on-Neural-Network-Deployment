# Tensor Core VS CUDA Core

### Tensor core和cuda core 的概念

tensor core和cuda core 都是运算单元，是硬件名词，其主要的差异是算力和运算场景。

* 场景：`cuda core`是全能通吃型的浮点运算单元，`tensor core`专门为深度学习矩阵运算设计。
* 算力：在高精度矩阵运算上 `tensor core`吊打`cuda core`。

#### CUDA Core

回顾一下NVIDIA的显卡架构出道顺序：

Tesla1.0 （2006年, 代表GeForce8800) -> Tesla2.0 (GT200) -> Fermi(算力可以支撑深度学习啦) -> Kepler(core增长) -> Maxwell（core继续增长） -> Pascal（算力提升）-> Volta(第一代tensor core) -> Turning(第二代 tensor core) -> Ampere(第三代tensor core)。

<figure><img src="../../.gitbook/assets/图片 (5) (1) (1) (1).png" alt=""><figcaption></figcaption></figure>

一个cuda core 包含了一个整数运算单元integer arithmetic logic unit (ALU) 和一个浮点运算单元floating point unit (FPU)。然后，这个core能进行一种fused multiply-add (FMA)的操作，通俗一点就是一个加乘操作的融合。特点：在不掉精度的情况下，<mark style="color:red;">单指令完成乘加操作</mark>，并且这个是支持32-bit精度。更通俗一点，就是深度学习里面的操作变快了。

#### **Tensor Core**

Tesla V100 的Tensor Core是可编程矩阵乘积单元，可为训练和推理应用提供高达 125 Tensor TFLOPS 的性能。Tesla V100 GPU 包含 640 个Tensor Core：每个 SM 8 个。Tensor Core及其相关数据路径是定制的，可显著提高浮点计算吞吐量，而功耗成本却不高。

<mark style="color:red;">如下图所示，每个Tensor Core提供一个 4x4x4 矩阵处理单元，执行 D = A \* B + C 的运算，其中 A、B、C 和 D 均为 4×4 矩阵。矩阵乘法输入 A 和 B 是 FP16 矩阵，而累加矩阵 C 和 D 可能是 FP16 或 FP32 矩阵。</mark>



<figure><img src="../../.gitbook/assets/图片 (2) (1) (1) (1) (1).png" alt=""><figcaption></figcaption></figure>

如下图所示，每个Tensor Core每时钟可执行 64 次浮点 FMA 混合精度操作（FP16 输入乘法与全精度乘积和 FP32 累加），一个 SM 中的 8 个Tensor Core每时钟总共可执行 1024 次浮点运算。与使用标准 FP32 运算的 Pascal GP100 相比，每个 SM 的深度学习应用吞吐量大幅提高了 8 倍，因此 Volta V100 GPU 的吞吐量与 Pascal P100 GPU 相比总共提高了 12 倍。张量核通过 FP32 累加对 FP16 输入数据进行运算。如图 所示，在 4x4x4 矩阵乘法中，FP16 乘法产生的全精度结果与给定点乘中的其他乘积一起在 FP32 运算中累加。

<figure><img src="../../.gitbook/assets/图片 (3) (1) (1) (1) (1).png" alt=""><figcaption></figcaption></figure>

在程序执行过程中，多个Tensor Core可通过一个完整的线程同时执行。一个 warp 中的线程可提供较大的 16x16x16 矩阵操作供Tensor Core处理。

Tensor Core基础能力在于，一个时钟周期内可以完成一个64 floating point 的FMA，而cuda core是搞不定的，分多次。

其次，Tensor Core也能堆叠，V100上面就堆了640个。而且Tensor Core经过了几次升级，其操作的精度更加丰富了：

<figure><img src="../../.gitbook/assets/图片 (1) (1) (1) (1) (1) (1) (1) (1).png" alt=""><figcaption></figcaption></figure>

### More examples

<figure><img src="../../.gitbook/assets/图片 (3) (1) (1).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/图片 (1) (1) (1) (1) (1) (1) (1).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/图片 (2) (1) (1) (1).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/图片 (3) (1) (1) (1).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/图片 (4) (1) (1).png" alt=""><figcaption></figcaption></figure>

### Reference

* [https://www.zhihu.com/question/451127498/answer/1813864500](https://www.zhihu.com/question/451127498/answer/1813864500)
* [https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/](https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/)
