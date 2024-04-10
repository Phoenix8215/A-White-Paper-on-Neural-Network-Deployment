# 映射和偏移

### 近10年模型的变化与硬件的发展

DNN模型的大小，几乎在以每年10倍的FLOPs在增长

<figure><img src="../../.gitbook/assets/图片 (22).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/ahtlrvyj8072btgpgquo-1.webp" alt=""><figcaption></figcaption></figure>

相比于模型的发展，硬件的发展速度很慢。即便硬件有了，还需要有相对应的的编译器。有了基本的编译器后，还需要有编译器的优化(TensorRT 3.x-8.x的演变)，还需要有一套其他的SDK。 Vision Transformer虽然在2020年左右兴起了，但是真正在硬件上达到如同现在的CNN的优化效果， 可能需要8-10年时间。 所以，大家一般会考虑如果用现有的硬件基础上减少模型计算量、增大模型计算密度等等。所以针对 这些需求，就有了“量化(quantization)”，“剪枝(Prunning)”等这些优化方法。

### 模型量化回顾

<figure><img src="../../.gitbook/assets/图片 (24).png" alt=""><figcaption></figcaption></figure>

存储FP32需要4个字节，但存储INT8只需要一个字节。因为训练的时候，我们想训练到非常细节的部 分，所以我们使用FP32。但是如果我们在部署的时候只用8个bit就把FP32的数据给表现出来，那么 计算量以及能源消耗不就更好？模型量化就是做这个的。

### 什么是量化

模型量化是通过减少模型中计算精度从而减少模型计算所需要的访存量，进而 进一步提高计算密度的一种方法。计算精度可以分为FP32, FP16, FP8, INT8, INT32, TF32这些

量化针对的是&#x20;

* activation value&#x20;
* weight

所以一般来说我们会对conv或者linear这些计算密集型算子进行量化

<figure><img src="../../.gitbook/assets/图片.png" alt=""><figcaption></figcaption></figure>

### 量化会出现什么问题

fp32某个区间内的数值等于量化后int8上的一个点

<figure><img src="../../.gitbook/assets/图片 (1).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/图片 (23).png" alt=""><figcaption></figcaption></figure>

仅仅用256种数据去表现FP32的所有可能出现的 数据，有可能会造成表现力下降。如果能够比较 完美的用这256个数据去最大限度的表现FP32的 原始数据分布，是量化的一个很大挑战。 换句话说，就是如何合理的设计这个dynamic range是量化的重点。

### 量化的基本原理: 映射和偏移

example🌰🌰🌰

$$
R{:}\{x|x\in[-100,0]x\in Z\}\\Q{:}\{y|y\in[0,20],y\in Z\}
$$

倘若想把R中的数据用Q来表示，如何做？

<mark style="color:red;">方法一(非对称量化)</mark>：&#x20;

根据R和Q中x和y可以取的最大值和最小值，计算得到一个缩放比(ratio)

$$
ratio=\frac{(R_{max}-R_{min})}{(Q_{max}-Q_{min})}
$$

以及缩放后的A要在B的范围中显示，所需要的偏移量(distance)

$$
distance=Q_{min}-\frac{R_{min}}{ratio}
$$

最终，通过ratio和distance的到x和y的关系

$$
y=\frac{x}{ratio}+distance
$$

利用上面的公式求得：

$$
\begin{aligned}
&ratio=\frac{(R_{max}-R_{min})}{(Q_{max}-Q_{min})}=\frac{100}{20}=5 \\
&distance=0-\frac{-100}{5}=20
\end{aligned}
$$

通过ratio和distance我们可以这么理解： Q中每一个元素可以代表R中每5个元素，并且偏移量是20。但是存在一个问题：

如果说可以通过下面的公式将R中的数据映射到Q中的话，

$$
y=\frac{x}{ratio}+distance
$$

那么我们按照下面的公式反着计算的话，是不是就可以通过Q中的数据得到R呢？

$$
x=(y-distance)*ratio
$$

<pre class="language-latex"><code class="lang-latex">y = 0, x = -100
y = 1, x = -95
y = 2, x = -90
<strong>y = 3, x = -85
</strong>y = 4, x = -80
</code></pre>

相比于原本的101个R中的数据，如今我们只能够得到R中21个数据。 比如说-96， -93， -81是无法得到的 ,那么问题出现在哪里？以及如何解决呢？

看看下面分布不同的四个数据

$$
R\colon\{x|x\in[-100,0]x\in Z\}\\Q\colon\{y|y\in[0,20],y\in Z\}
$$

<figure><img src="../../.gitbook/assets/图片 (20).png" alt=""><figcaption></figcaption></figure>

很明显，虽然上述的4个example中数据都呈现-100\~0中，但是由于数据的分布形式不同，如 果我们统一都用一种ratio和distance的话，会有很大的误差。

* example1

R中的数据服从均匀分布，随机出现在 -100到0中。 这个时候因为数据分布是均匀，所以 将-100\~0均分20个部分映射到Q中效果是比较好的，误差也会比较小

<figure><img src="../../.gitbook/assets/图片 (18).png" alt="" width="375"><figcaption></figcaption></figure>

* example2

R中的数据呈现高斯分布，主要集中 在-80\~-20中 这个时候R中的数据分布是属于正态 分布，所以数据主要集中在靠拢中间 的部分，靠近边缘的数据出现的概率 比较低。如果依然等分为20个部分的 话，靠近中间的数据会出现较大的误 差。

<figure><img src="../../.gitbook/assets/图片 (26).png" alt="" width="375"><figcaption></figcaption></figure>

* example3

R中的数据呈现高斯分布，主要集中 在-65\~-45中 这个时候R中的数据分布是属于正态 分布，所以数据主要集中在靠拢中间 的部分。但是由于方差比较大，所以 接近两侧的数据几乎没有。如果按照 左图分配的话，Q中0-5, 15-20所代 表的数字没有意义。

<figure><img src="../../.gitbook/assets/图片 (27).png" alt="" width="375"><figcaption></figcaption></figure>

* example4

R中的数据呈现高斯分布，主要集中 在-30\~-15中 这个时候R中的数据分布是属于正态 分布，所以数据主要集中在靠拢中间 的部分。但是由于方差比较大，以及 均值有所偏移，所以R中-100\~-30这 段区域几乎没有。将这段区域映射到 Q是没有意义的。

<figure><img src="../../.gitbook/assets/图片 (28).png" alt="" width="375"><figcaption></figcaption></figure>

所以，为了能够让R到Q的映射合理，以及将Q中的数据还原为R时误差能够控制到最小，我们需 要根据R中的数据分布合理的设计这个ratio和distance。

### 量化的基本术语

目前为止我们所做的事情其实就是所谓的量化，我们稍微把一些概念和术语给整理一下&#x20;

* &#x20;R是一组FP32的数据，能够表现的数据种类有很多，大约是种(4亿)，



### Reference

* [https://www.lesswrong.com/posts/vnvGhfikBbrjZHMuD/predicting-gpu-performance](https://www.lesswrong.com/posts/vnvGhfikBbrjZHMuD/predicting-gpu-performance)
* [https://arxiv.org/abs/2202.05924](https://arxiv.org/abs/2202.05924)
