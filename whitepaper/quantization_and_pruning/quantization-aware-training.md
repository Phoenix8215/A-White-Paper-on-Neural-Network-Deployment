# ✊ Quantization-Aware Training

### QAT是什么&#x20;

QAT(Quantization Aware Training)也被称作显式量化。我们明确的在模型中添加Q/DQ节点 (量化/反量化)，来控制某一个算子的精度。并且通过fine-tuning来更新模型权重，让权重学习 并适应量化带来的精度误差

<figure><img src="../../.gitbook/assets/图片 (48).png" alt="" width="563"><figcaption></figcaption></figure>

QAT的核心就是通过添加fake quantization，也就是Q/DQ节点，来模拟量化过程。在谈QAT 的优缺点前，我们先看一下Q/DQ是什么，以及QAT的流程.

### Q/DQ是什么&#x20;

Q/DQ node也被称作fake quantization node，是用来模拟fp32->int8的量化的scale和 shift(zero-point)，以及int8->fp32的反量化的scale和shift(zero-point)。QAT通过Q和DQ node里面存储的信息对fp32或者int8进行线性变换

<figure><img src="../../.gitbook/assets/图片 (49).png" alt=""><figcaption></figcaption></figure>

我们回顾一下之前mapping range的小节。不妨我们把Q/DQ按照公式的形式展现一下。这里面有几个符号，说明一下：&#x20;

<figure><img src="../../.gitbook/assets/图片 (50).png" alt=""><figcaption></figcaption></figure>

$$
s=\frac{2^b-1}{\alpha-\beta}
$$

$$
z=-\text{round}(\beta\cdot s)-2^{b-1}
$$

$$
\operatorname{clip}(x,l,u)\begin{cases}l,&x<l\\x,&l\leq x\leq u\\u,&x>u\end{cases}
$$







那么Q的公式可以理解为：

$$
x_q=\text{quantize}(x,b,s,z)=\text{clip}(\text{round}(s\cdot x+z),-2^{b-1},2^{b-1}-1)
$$

DQ的公式可以理解为：

$$
\hat{x}=\text{dequantize}(x_q,s,z)=\frac{1}{s}(x_q-z)
$$

### Q/DQ是什么+ 可量化层的计算

在理解了上面的公式的基础上，我们在进一步理解这个：对于一个线性计算的op(conv或者linear)。我们把 fp32精度的op的计算简化成

$$
op_{FP32}\left(w,x\right)=w*x
$$

既然x和w是fp32的，那么我们也可以这么表示

$$
op_{FP32}\left(w,x\right)=dequantize(s_w,w_q)*dequantize(s_x,x_q)
$$

展开

$$
op_{FP32}\left(w,x\right)=\frac{1}{s_{w}}*w_{q}*\frac{1}{s_{x}}*x_{q}\\op_{FP32}\left(w,x\right)=\frac{1}{s_{w}*s_{x}}*w_{q}*x_{q}
$$

$$
\text{因为计算量的主要是}w_q*x_q\text{,是int8计算,所以我们可以把这个公式写成}.
$$

$$
op_{int8}\left(w_q,x_q,s_w,s_x\right)=\frac{1}{s_w*s_x}*w_q*x_q
$$

<mark style="color:red;">所以我们知道DQ + fp32精度的op可以拼成一个int8精度的op.</mark>

> 这里以NVIDIA采用的 对称量化量化与反量 化计算为例，计算过 程没有涉及zero-shift

如果DQ + fp32精度op可以拼成一个int8精度的op， 那么DQ + fp32精度op +Q是不是也可以融合呢？

$$
quantization(x^{\prime})=s_{x^{\prime}}*x^{\prime}
$$

> 这里的𝑥′是来自于上一 层的输出，是fp32

由于𝑥′是来自于上一层计算，可以把𝑥′展开

$$
quantization(x^{\prime})=s_{x^{\prime}}*\frac{1}{s_{w}*s_{x}}*w_{q}*x_{q}
$$

我们可以看到这个依然是一个线性变化。所以说DQ + fp32精度OP + Q可以融合在一起凑成一个int8的op， 所以我们可以把这个公式替换成：

$$
op_{int8}\big(x_{q},w_{q},s_{w},s_{x},s_{x\prime}\big)=s_{x^{\prime}}*\frac{1}{s_{w}*s_{x}}*w_{q}*x_{q}
$$

我们称这个op或者layer为quantizable layer，翻译为可量化层。&#x20;

* &#x20;这个可量化层的输入和输出都是int8&#x20;
* 计算的主体也是int8，可以节省带宽的同时，提高计算效率

<figure><img src="../../.gitbook/assets/图片 (3) (1) (1) (1).png" alt="" width="563"><figcaption></figcaption></figure>

将上面的公式用图比较直观的表示出来就是 这个样子：&#x20;

* 蓝色conv表示是fp32的op&#x20;
* 绿色conv表示是int8的op&#x20;
* 蓝色arrow表示的是fp32的tensor
* 绿色arrow表示的是int8的tensor

我们理解了Q/DQ之后，我们再回到这张图看一下，

<figure><img src="../../.gitbook/assets/图片 (4) (1) (1).png" alt=""><figcaption></figcaption></figure>

我们知道conv和Relu是可以融合在一起成为ConvReLU算子，同时根据之前的公式和图，我们知道：

* DQ和fp32精度的conv组合在一起，可以融合成一个int8精度的conv&#x20;
* fp32精度输出的conv和后面的Q也可以融合在一起，输出一个int8精度的activation value

<figure><img src="../../.gitbook/assets/图片 (5) (1) (1).png" alt=""><figcaption></figcaption></figure>

将这些虚线包围起来的算子融合在一起，用一个int8的op来替换后，整个网络就会变成这个样子

<figure><img src="../../.gitbook/assets/图片 (6) (1).png" alt=""><figcaption></figcaption></figure>

新生成的QConvRelu以及Qconv是int8精度的计算，速度很快并且TensorRT会很大几率分配tensor core执行 这个计算。这个就是TensorRT中对量化节点的优化方法之一。

### QAT的工作流

理解了Q/DQ再去看QAT就非常容易了。QAT是一种Fine-tuning方式，通常对一个pre-trained model进行添加Q/DQ节点模拟量化，并通过训练来更新权重去吸收量化过程所带来的误差。添 加了Q/DQ节点后的算子会以int8精度执行

<figure><img src="../../.gitbook/assets/图片 (7) (1).png" alt="" width="563"><figcaption></figcaption></figure>

pytorch支持对已经训练好的模型自动添加Q/DQ节点。详细可以参考 https://github.com/NVIDIA/TensorRT/tree/main/tools/pytorch-quantization

TensorRT对包含Q/DQ节点的onnx模型使用很多图优化，从而提高计算效率。主要分为&#x20;

* Q/DQ fusion&#x20;
  * 通过层融合，将Q/DQ中的线性计算与conv或者linear这种线性计算融合在一起，实现int8计算 • Q/DQ Propagation&#x20;
  * 将Q节点尽量往前挪，将DQ节点尽量往后挪，让网络中int8计算的部分变得更长

### TensorRT中QAT的Q/DQ Fusion技巧

<figure><img src="../../.gitbook/assets/图片 (8) (1).png" alt=""><figcaption></figcaption></figure>

### TensorRT中QAT的Q/DQ Propagation技巧

<figure><img src="../../.gitbook/assets/图片 (9) (1).png" alt=""><figcaption></figcaption></figure>

Max Pooling与Q/DQ的propagation (由于maxpooling的结果在量化前后是没有变化，所以我们可以把fp32的maxpool节点转为int8的maxpool，从而达到加速)

### QAT的学习过程&#x20;

* 主要是训练weight来学习误差&#x20;
* Q/DQ中的scale和zero-point也是可以训练的。通过训练来学习最好的scale来表示dynamic range
* 没有PTQ中那样人为的指定calibration过程 ,不是因为没有calibration这个过程来做histogram的统计，而是因为QAT会利用fine-tuning的数 据集在训练的过程中同时进行calibration，这个过程是我们看不见的。这就是为什么我们在 pytorch创建QAT模型的时候需要选定calibration algorithm.

### 我们在部署过程中应该按照什么样子的流程进行QAT

没有必要盲目的使用QAT，在使用QAT之前先看看PTQ是否已经达到了最佳。可以按 照左边的图进行量化测试：

1. 先进行PTQ
   1. 从多种calibration策略中选取最佳的算法
   2. 查看是否精度满足，如果不行再下一步。
2. 进行partial-quantization
   1. 通过layer-wise的sensitve analysis分析每一层的精度损失
   2. 尝试fp16 + int8的组合
   3. fp16用在敏感层(网络入口和出口)，int8用在计算密集处(网络的中间)
   4. 查看是否精度满足，如果不行再下一步。
   5. (注意，这里同时也需要查看计算效率是否得到满足)
3. 进行QAT来通过学习权重来适应误差
   1. 选取PTQ实验中得到的最佳的calibration算法
   2. 通过fine-tuning来训练权重(大概是原本训练的10%个epoch)
   3. 查看是否精度满足，如果不行查看模型设计是否有问题
   4. (注意，这里同时也需要查看层融合是否被适用，以及Tensor core是否被用)

<figure><img src="../../.gitbook/assets/图片 (2) (1) (1) (1) (1).png" alt="" width="360"><figcaption></figcaption></figure>

普遍来讲，量化后精度下降控制在相对精度损失<=2%是最好的。
