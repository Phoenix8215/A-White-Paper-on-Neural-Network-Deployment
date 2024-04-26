# 😽 手算Ampere架构各个精度的Throughout

### 相关数据

<figure><img src="../../.gitbook/assets/图片 (11) (1) (1).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/图片 (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1).png" alt=""><figcaption></figcaption></figure>

### FP64 CUDA Core

<figure><img src="../../.gitbook/assets/图片 (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1).png" alt=""><figcaption></figcaption></figure>

使用CUDA Core的FP64最好理解 一共32个计算FP64的CUDA Core，每一个clk(时钟周期)计算一个FP64&#x20;

* 频率：1.41 GHz • SM数量：108&#x20;
* &#x20;一个SM中计算FP64的CUDA core的数量: 32&#x20;
* 一个CUDA core一个时钟周期可以处理的FP64: 1&#x20;
* 乘加: 2&#x20;

Throughput = 1.41 GHz \* 108 \* 32 \* 1\* 2 = 9.7 TFLOPS

### FP32 CUDA Core

<figure><img src="../../.gitbook/assets/图片 (2) (1) (1) (1) (1) (1) (1) (1) (1).png" alt=""><figcaption></figcaption></figure>

使用CUDA Core的FP32跟FP64差不多 一共64个计算FP32的CUDA Core，每一个clk计算一个FP32&#x20;

* 频率：1.41 GHz&#x20;
* SM数量：108&#x20;
* 一个SM中计算FP32的CUDA core的数量: 64&#x20;
* &#x20;一个CUDA core一个时钟周期可以处理的FP32: 1&#x20;
* 乘加: 2&#x20;

Throughput = 1.41 GHz \* 108 \* 64 \* 1\* 2 = 19.4 TFLOPS

### FP16 CUDA Core

<figure><img src="../../.gitbook/assets/图片 (3) (1) (1) (1) (1) (1) (1) (1) (1).png" alt=""><figcaption></figcaption></figure>

Ampere中没有专门针对FP16的CUDA core，而是将FP32的CUDA Core和FP64的CUDA Core一起使用来计算FP16。我们暂且说一个 SM中计算FP16的CUDA core的数量是: 256 ( = 32 \* 4 + 64 \* 2 )&#x20;

* 频率：1.41 GHz&#x20;
* SM数量：108&#x20;
* 一个SM中计算FP16的CUDA core的数量: 256&#x20;
* &#x20;一个CUDA core一个时钟周期可以处理的FP16: 1&#x20;
* 乘加: 2&#x20;

Throughput = 1.41 GHz \* 108 \* 256 \* 1 \* 2 = 78 TFLOPS

### INT8 CUDA Core

<figure><img src="../../.gitbook/assets/图片 (4) (1) (1) (1) (1) (1) (1).png" alt=""><figcaption></figcaption></figure>

Ampere中没有专门针对INT8的CUDA core，而是用INT32的CUDA Core计算INT8。我们暂且说一个SM中计算INT8的CUDA core的数量 是: 256 ( = 64 \* 4 )&#x20;

* 频率：1.41 GHz • SM数量：108&#x20;
* 一个SM中计算INT8的CUDA core的数量: 256&#x20;
* 一个CUDA core一个时钟周期可以处理的INT8: 1&#x20;
* 乘加: 2&#x20;

Throughput = 1.41 GHz \* 108 \* 256 \* 1 \* 2 = 78 TOPS

### FP16 Tensor core

<figure><img src="../../.gitbook/assets/图片 (7) (1) (1).png" alt=""><figcaption></figcaption></figure>

我们来分析Tensor Core Ampere架构使用的是第三代Tensor Core，可以一个clk完成一个 1024 ( = 256 \* 4)个FP16运算。准确来说是4x8的矩阵与8x8的矩阵的 FMA

* 频率：1.41 GHz&#x20;
* &#x20;SM数量：108&#x20;
* 一个SM中计算FP16的Tensor core的数量: 4&#x20;
* &#x20;一个Tensorcore一个时钟周期可以处理的FP16: 256&#x20;
* &#x20;乘加: 2&#x20;

Throughput = 1.41 GHz \* 108 \* 4 \* 256 \* 2 = 312 TFLOPS

### INT8 Tensor Core

<figure><img src="../../.gitbook/assets/图片 (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1) (1).png" alt=""><figcaption></figcaption></figure>

我们来分析Tensor Core Ampere架构使用的是第三代Tensor Core，可以一个clk完成一个 2048 ( = 256 \* 2 \* 4)个INT8运算。准确来说是4x8的矩阵与8x8的矩 阵的FMA&#x20;

* 频率：1.41 GHz • SM数量：108&#x20;
* 一个SM中计算INT8的Tensor core的数量: 4
* 一个Tensorcore一个时钟周期可以处理的INT8: 512
* 乘加: 2&#x20;

Throughput = 1.41 GHz \* 108 \* 4 \* 512 \* 2 = 624 TOPS





