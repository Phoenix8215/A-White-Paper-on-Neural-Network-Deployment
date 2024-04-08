# 模型部署的几大误区

### FLOPs不能衡量模型性能&#x20;

* 因为FLOPs只是模型计算大小的单位 ,还需要考虑&#x20;
  * 访存量&#x20;
  * 跟计算无关的部分:reshape, shortcut, nchw2nhwc等等&#x20;
  * DNN以外的部分 :前处理、后处理这些

### 不能够完全依靠TensorRT

* TensorRT可以对模型做适当的优化，但是有上限 ，比如:
  * 计算密度低的1x1 conv， depthwise conv不会重构&#x20;
  * &#x20;GPU无法优化的地方会到CPU执行&#x20;
  * &#x20;可以手动修改代码实现部分，让部分cpu执行转到gpu执行，可以直接将前后处理加到模型中。
  * 有些冗长的计算，TensorRT可能不能优化 ，这时只能修改代码实现部分&#x20;
  * 存在TensorRT尚未支持的算子 ，可以自己写plugin&#x20;
  * TensorRT不一定会分配Tensor Core ，因为TensorRT kernel auto tuning会选择最合适的kernel

### CUDA Core和Tensor Core的使用&#x20;

* 有的时候TensorRT并不会分配Tensor Core ： kernel auto tuning自动选择最优解 ，所以有时会出现类似于INT8的速度比FP16反而慢了&#x20;
* 使用Tensor Core需要让tensor size为8或者16的倍数(8的倍数对应到精度为fp16，16的倍数对应到精度为int8)
* CUDA Core是NVIDIA GPU架构中的基本计算单元。它是用于执行并行计算任务的GPU硬件单元。
* CUDA Core执行单个浮点运算操作，并可以通过并行处理多个数据元素来提高计算效率。
* CUDA Core是用于执行通用计算任务的基本构建块，可用于编写并行计算的CUDA程序。
* TensorRT Core是TensorRT推理引擎中的组件，用于加速和优化深度学习模型的推理。
* TensorRT Core利用GPU硬件和深度学习模型的特性进行推理优化，以提高推理性能和效率。
* TensorRT Core实现了各种优化技术，如层融合、内存优化、精度调整等，以提高模型推理的速度和资源利用率。

总结来说，CUDA Core是NVIDIA GPU架构中的计算单元，用于执行通用并行计算任务，而TensorRT Core是TensorRT推理引擎中的组件，用于加速和优化深度学习模型的推理。CUDA Core是底层硬件层面的概念，而TensorRT Core是高层软件层面的概念。它们在不同层次上发挥着不同的作用，但都对深度学习模型的推理性能有重要影响。&#x20;

### 不能忽视 前处理/后处理 的overhead&#x20;

* 对于一些轻量的模型，相比于DNN推理部分，前处理/后处理可能会更耗时间,因为有些前处理/后处理的复杂逻辑不适合GPU并行,然而有很多种解决办法 :
  * 可以把前处理/后处理中可并行的地方拿出来让GPU并行化 ：比如RGB2BGR, Normalization, resize, crop, NCHW2NHWC&#x20;
  * 可以在cpu上使用一些针对图像处理的优化库：比如Halide，使用Halide进行blur, resize, crop, DBSCAN, sobel这些会比 CPU快
  * 并不是能GPU加速的地方就GPU加速，还要考虑GPU的占用率
  * 也可以在CPU上跑NMS等操作，然后在GPU上跑下一帧的数据

### 对使用TensorRT得到的推理引擎做benchmark和profiling&#x20;

* 使用TensorRT得到推理引擎并实现infer只是优化的第一步，还需要使用NVIDIA提供的benchmark tools进行profiling ，分析模型瓶颈在哪里 ，分析模型可进一步优化的地方在哪里 ，分析模型中多余的memory access在哪里
* 可以使用： nsys, nvprof, dlprof, Nsight这些工具

### **1x1 与 depthwise conv的部署缺点**

1x1卷积和深度可分离卷积（Depthwise Convolution）在深度学习模型中经常被使用，它们具有一些特定的部署缺点。下面是它们的一些常见缺点：

* 1x1卷积的部署缺点：
  * 计算复杂度：尽管1x1卷积的计算量相对较小，但在处理特征图时仍需要进行相乘和相加操作，因此在某些情况下仍可能成为性能瓶颈。
  * 内存占用：1x1卷积需要在内存中存储和处理大量的中间特征图，特别是在输入通道和输出通道数目较大的情况下，可能导致较高的内存占用。
  * 参数数量：1x1卷积涉及到一定数量的权重和偏置参数，特别是在具有较多输入通道和输出通道的情况下，会增加模型的参数数量。
* 深度可分离卷积的部署缺点：
  * 内存占用：深度可分离卷积通常需要在内存中存储和处理多个中间特征图，包括深度方向的分离卷积和逐点卷积的结果。这可能导致较高的内存占用。
  * 网络深度：深度可分离卷积通常用于减少参数数量和计算量，但在一些情况下，它可能导致网络变得更深，从而增加了模型的计算和内存开销。
  * 特征表示能力：相对于传统的卷积操作，深度可分离卷积的特征表示能力可能较弱。它更适用于具有较简单特征的任务，对于复杂特征的捕获可能有一定限制。

总结来说，1x1卷积和深度可分离卷积在深度学习模型中具有一些缺点。它们可能增加计算和内存开销，导致网络变得更复杂或限制特征表示能力。在使用这些卷积操作时，需要权衡性能、内存和模型表达能力，并根据具体任务的需求进行选择和优化。

### Reference

* [深度学习模型部署TensorRT加速（十）：TensorRT部署分析与优化方案（一）](https://blog.csdn.net/chenhaogu/article/details/132684052?app\_version=6.2.8\&csdn\_share\_tail=%7B%22type%22%3A%22blog%22%2C%22rType%22%3A%22article%22%2C%22rId%22%3A%22132684052%22%2C%22source%22%3A%22weixin\_41503009%22%7D\&utm\_source=app)
