# 👊 Post-Training Quantization

### 量化的基本原理: 基本术语(PTQ, QAT)

根据量化的时机，一般我们会把量化分为&#x20;

* PTQ(Post-Training Quantization)，训练后量化&#x20;
* QAT(Quantization-Aware Training)，训练时量化 PTQ一般是指对于训练好的模型，通过calibration算法等来获取dynamic range来进行量化。 但量化普遍上会产生精度下降。所以QAT为了弥补精度下降，在学习过程中通过Fine-tuning权 重来适应这种误差，实现精度下降的最小化。所以一般来讲，QAT的精度会高于PTQ。但并不绝对。

<figure><img src="../../.gitbook/assets/图片 (40).png" alt=""><figcaption></figcaption></figure>

### PTQ是什么&#x20;

PTQ(Post-training quantization)也被称作隐式量化(implicit quantization)。我们并不显式的 对算子添加量化节点(Q/DQ)，calibration之后TensorRT根据情况进行量化

<figure><img src="../../.gitbook/assets/图片 (41).png" alt="" width="563"><figcaption></figcaption></figure>

trtexec在选择参数进行fp16或者int8指定的时候，使用的就是PTQ。(int8的时候需要指定 calibration dataset)。很方便使用，但是我们需要先理解PTQ的利弊。

具体使用就是，我们**导出ONNX模型**，转换为TensorRT的过程中可以使用trt提供的Calibration方法去校准，这个使用起来比较简单。可以直接使用trt官方提供的`trtexec`命令去实现，也可以使用trt提供的python或者C++的API接口去量化，比较容易。

目前，TensorRT提供的后训练量化算法也多了好多，分别适合于不同的任务：

* EntropyCalibratorV2

> Entropy calibration chooses the tensor’s scale factor to optimize the quantized tensor’s information-theoretic content, and usually suppresses outliers in the distribution. This is the current and recommended entropy calibrator and is required for DLA. Calibration happens before Layer fusion by default. <mark style="color:red;">It is recommended for CNN-based networks.</mark>

* MinMaxCalibrator

> This calibrator uses the entire range of the activation distribution to determine the scale factor. It seems to work better for NLP tasks. Calibration happens before Layer fusion by default. <mark style="color:red;">This is recommended for networks such as NVIDIA BERT (an optimized version of Google's official implementation).</mark>

* EntropyCalibrator

> This is the original entropy calibrator. It is less complicated to use than the LegacyCalibrator and typically produces better results. <mark style="color:red;">Calibration happens after Layer fusion by default.</mark>

* LegacyCalibrator

> This calibrator is for compatibility with TensorRT 2.0 EA. This calibrator requires user parameterization and is provided as a fallback option if the other calibrators yield poor results. Calibration happens after Layer fusion by default. You can customize this calibrator to implement percentile max, for example, 99.99% percentile max is observed to have best accuracy for NVIDIA BERT.

通过上述这些算法量化时，TensorRT会在优化网络的时候尝试INT8精度，假如某一层在INT8精度下速度优于默认精度（FP32或者FP16）则优先使用`INT8`。这个时候我们**无法控制某一层的精度**，因为TensorRT是以速度优化为优先的（很有可能某一层你想让它跑int8结果却是fp32）。即使我们使用API去设置也不行，比如`set_precision`这个函数，因为TensorRT还会做图级别的优化，它如果发现这个`op`（显式设置了`INT8`精度）和另一个`op`可以合并，就会忽略你设置的`INT8`精度。

说白了就是不好控制。我也尝试过这种方式，简单情况，简单模型问题不大（resnet系列），涉及到比较复杂的（transformer）这个设置精度可能不管用。

### PTQ(优缺点分析)&#x20;

**优点 ：** 方便使用，不需要训练。可以在部署设备上直接跑&#x20;

**缺点：**

1. **精度下降：**量化过程会导致精度下降。但PTQ没有类似于QAT这种fine-tuning的过程。所以权重不会更新来吸收这种误差
2. **量化不可控：**TensorRT会权衡量化后所产生的新添的计算或者访存， 是否用INT8还是FP16。TensorRT中的kernel autotuning会选择核函数来做FP16/INT8的计算。来查看是否在CUDA core上跑还是在Tensor core上跑，有可能FP16是在Tensor core上，但转为INT8之后就在CUDA core上了
3. **层融合问题：**量化后有可能出现之前可以融合的层，不能融合了 ，量化会添加reformatter这种更改tensor的格式的算子，如果本来融合的两个算子间添加了这 个就不能被融合了， 比如有些算子支持int8，但某些不支持。之前可以融合的，但因为精度不同不能融合了 如果INT8量化后速度反而会比FP16/FP32要慢，我们可以从以上的2和3去分析并排查原因

### 量化中的sensitive analysis&#x20;

从精度分析的角度去弥补PTQ的精度下降，我们可以进行`layer-wise`的量化分析。这种方法被称 作l`ayer-wise sensitive analysis`

<figure><img src="../../.gitbook/assets/图片 (43).png" alt=""><figcaption></figcaption></figure>

普遍来讲，模型框架中会有一些层的量化对精度的影响比较大。我们管它们叫做敏感层(sensitive layer)。对于这些敏感层的量化我们需要非常小心，尽量用FP16。<mark style="color:red;">敏感层一般靠近模型的输入输出。</mark>

<figure><img src="../../.gitbook/assets/图片 (44).png" alt=""><figcaption></figcaption></figure>

### 学习使用Polygraphy&#x20;

对于这种`sensitive analysis`，其实NVIDIA最近几年推出了一个叫做`polygraphy`的一个工具。可以用来 分析并查找模型精度下降并且影响比较大的地方。非常好的工具，做`TensorRT`量化必须要掌握。能做 的事情太多，比如：

* `onnxruntime`与TensorRT engine的`layer-wise`的精度分析
* 输出每一层`layer`的权重`histogram`&#x20;
* 截取影响整个网络中对精度影响最大的子网，并使用`onnx-surgeon`单独拿出来

### FP16/INT8对计算资源的利用

我们在做量化后，我们无法指定将量化后的conv或者gemm放在Tensor core还是在CUDA core上计算。 这些是TensorRT在帮我们选择核函数的时候自动完成的。那么我们如何查看我们是否在用Tensor core呢？ 我们一般有这么几个办法&#x20;

* 使用`dlprof`&#x20;
* 使用`nsight system`
* 使用`trtexec`

### DLProf&#x20;

DLProf (Deep learning Profiler)工具可以把模型在GPU上的执行情况以TensorBoard的形式打印出来， 分析TensorCore的使用情况。感兴趣的可以查看一下。但需要注意的是，DLProf不支持Jetson系列的 Profile。对于Jetson，我们可以使用Nsight system或者trtexec

<figure><img src="../../.gitbook/assets/图片 (45).png" alt="" width="563"><figcaption></figcaption></figure>

### Nsight System/trtexec&#x20;

如果是利用Nsight system的话，我们可以查看到哪一个kernel的时间占用率最高，之后从kernel的名字 取推测这个kernel是否在用Tensor Core。(从kernel名字推测kernel的计算设备需要经验)

<figure><img src="../../.gitbook/assets/图片 (46).png" alt=""><figcaption></figcaption></figure>

从kernel名字推测可以从kernel中的关键字去猜，比如&#x20;

* &#x20;h884 = HMMA = FP16 TensorCore&#x20;
* i8816 = IMMA = INT8 TensorCore&#x20;
* &#x20;hcudnn = FP16 normal CUDA kernel (`without TensorCore)`&#x20;
* &#x20;icudnn = INT8 normal CUDA kernel (`without TensorCore`)&#x20;
* scudnn = FP32 normal CUDA kernel (`without TensorCore`)

> `HMMA: Half-precision matrix multiply and accumulate`&#x20;
>
> `IMMA: Int-precision matrix multiply and accumulate`&#x20;

### reference

* [https://mp.weixin.qq.com/s/DlMC7H4Wh8CaDUEEHL3yNQ](https://mp.weixin.qq.com/s/DlMC7H4Wh8CaDUEEHL3yNQ)





