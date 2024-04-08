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

### 不能忽视 前处理/后处理 的overhead&#x20;

* 对于一些轻量的模型，相比于DNN推理部分，前处理/后处理可能会更耗时间,因为有些前处理/后处理的复杂逻辑不适合GPU并行,然而有很多种解决办法 :
  * 可以把前处理/后处理中可并行的地方拿出来让GPU并行化 ：比如RGB2BGR, Normalization, resize, crop, NCHW2NHWC&#x20;
  * 可以在cpu上使用一些针对图像处理的优化库：比如Halide，使用Halide进行blur, resize, crop, DBSCAN, sobel这些会比 CPU快
  * 并不是能GPU加速的地方就GPU加速，还要考虑GPU的占用率
  * 也可以在CPU上跑NMS等操作，然后在GPU上跑下一帧的数据

### 对使用TensorRT得到的推理引擎做benchmark和profiling&#x20;

* 使用TensorRT得到推理引擎并实现infer只是优化的第一步，还需要使用NVIDIA提供的benchmark tools进行profiling ，分析模型瓶颈在哪里 ，分析模型可进一步优化的地方在哪里 ，分析模型中多余的memory access在哪里
* 可以使用： nsys, nvprof, dlprof, Nsight这些工具
