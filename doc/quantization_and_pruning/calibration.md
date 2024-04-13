# 👍 校准

量化中另外一个非常重要的概念：Calibration(校准) 对于一个训练好的模型，权重是固定的，所以可以通过一次计算就可以得到每一层的量化参数。 但是activation value(激活值)是根据输入的改变而改变的。所以需要通过类似于统计的方式去 寻找对于不同类型的输入的不同的dynamic range，这个过程叫做校准。

<figure><img src="../../.gitbook/assets/图片 (33).png" alt="" width="563"><figcaption></figcaption></figure>

针对不同的输入，各层layer的input activation value都 会有不同的分布和取值。大数据集的差别比较大。我们需 要通过训练数据集中的一部分数据来尝试表征整个数据集 的分布。 这个小数据集就是calibration dataset。一般往往很小， 但需要尽量有整体的表征。

calibration的过程一般是在模型训练以后进行的，所以一般与PTQ搭配使用，整体的流程就是:&#x20;

* &#x20;在calibration dataset中做一次FP32的推理&#x20;
* 以histogram的形式去统计每一层的floating point的分布 ，(注意，因为activation value是per-tensor quantization) ，寻找能够表征当前层的floating point分布的scale 。
* &#x20;这里会有几种不同的算法，比较常见的有&#x20;
  * Minmax calibration&#x20;
  * Entropy calibration&#x20;
  * Percentile calibration

### Minmax calibration

FP32->INT8的scale需要能够把FP32 中的最大最小值都给覆盖住。 如果floating point的分布比较离散， 各个区间下的分布都比较均匀， minmax是个不错的选择

然而，如果只是极个别数据分布在这种地方的话，会让dynamic range变得比较稀疏，不适合用minmax。

### Entropy clibration

通过计算KL散度，寻找阈值，能够最小化量化前的 FP32的浮点数分布于INT8的量化后整 形分布 目前TensorRT使用默认的是Entropy calibration。一般来讲使用entropy calibration精度可以比较好&#x20;

### Percentile calibration

如上图所示，表示的是FP32中占据 99.99%的浮点数参与量化。这样可以 避免极个别特殊点(误差)参与量化，导出量化出现问题。 Percentile有99.9%, 99.99%, 99.999%等等。(不同分位数可能将准确率提升两个百分点，这里要多尝试)

### 如何选择calibration algorithm

如同quantization granularity对不同的情况会有不同的策略。calibration的算法 选择也会根据输入是activation value还是weights而改变。那么我们该如何选择呢？

<mark style="color:red;">calibration算法的选择，我们按照这个规律去选</mark>&#x20;

* <mark style="color:red;">weight的calibration，选用minmax</mark>&#x20;
* <mark style="color:red;">activation的calibration，选用entropy或者percentile</mark>

<figure><img src="../../.gitbook/assets/图片 (34).png" alt=""><figcaption></figcaption></figure>

(左)固定activation values的精度，量化weights。(右)固定weights的精度，量化activation values 我们可以看出不同模型的量化选择上会有不同的策略。calibration的选择与模型架构相关

### calibration dataset与batch size的关系

在使用calibration dateset中构建histogram是需要注意的一个点：calibration时的batch size会影响精度。 更准确来说会影响histogram的分布，这个跟TensorRT在构建浮点数的histogram的算法有关

[https://github.com/NVIDIA/TensorRT/issues/774](https://github.com/NVIDIA/TensorRT/issues/774)

上面的说法表明：在创建histogram直方图的时候，如果出现了大于当前histogram可以表示的最大值的时 候，TensorRT会直接平方当前histogram的最大值，来扩大存储空间

* (思考)如果batchsize=1，最后一个batch的浮点数很大，那么最终的histogram会呈现什么形状？&#x20;
* (思考)如果batchsize=16，但每一个batch size的数据分布很均匀，histogram会呈现什么形状？&#x20;
* (思考)如果模型的鲁棒性很强，batchsize=1和batchsize=16/32/64/128的区别会有吗

1. example

<figure><img src="../../.gitbook/assets/图片 (36).png" alt="" width="563"><figcaption></figcaption></figure>

这时histogram的后半段很稀疏，甚至没有数据。在量化的时候会根据这个直方图来将 FP32转为INT8，很显然这块领域是多余的

<figure><img src="../../.gitbook/assets/图片 (37).png" alt="" width="563"><figcaption></figcaption></figure>

2. example

<figure><img src="../../.gitbook/assets/图片 (38).png" alt="" width="563"><figcaption></figcaption></figure>

我们希望每一个batch里面的数据比较均匀， 让比较大的数据出现的时候，histogram的 范围已经能够表现它了。 e.g. 当2.4出现的时候，如果之前已经出现 过1.54，那么hisogram的range不需要改变。否则range的最大值会变成5.76 (总的来讲，calibratio的batch size越大越好，但不是绝对的)

<figure><img src="../../.gitbook/assets/图片 (39).png" alt="" width="563"><figcaption></figcaption></figure>
