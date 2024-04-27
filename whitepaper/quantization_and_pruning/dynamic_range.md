# 👏 动态量化范围

### 什么是动态量化范围

动态量化范围是指在进行量化操作时，对权重或激活值进行量化的范围。动态范围量化将所有权重值从浮点精度转换为 8 位整数精度。为了进一步减少推理延迟，<mark style="color:red;">动态范围根据激活值的范围将其动态量化为 8 位整数</mark>，并使用 8 位整数的权重和激活值进行计算。乘法和累加运算完成之后，激活值将被去量化。下图显示了动态范围量化的整体机制。这种优化使推理延迟接近完全定点推理。

总之，动态范围量化在进行乘法运算和使用整数算术求和时，可加快执行速度，提高执行效率，减少执行时间和功耗。此外，与半精度浮点量化相比，模型尺寸进一步缩小。

<figure><img src="../../.gitbook/assets/图片 (31).png" alt=""><figcaption></figcaption></figure>

去量化后的实数不可能恢复到量化前的精确值。因此，动态范围量化可能会导致精度下降。与全整数量化一样，动态范围量化可以减少模型大小和推理时间。全整数量化和动态范围量化的区别在于，动态范围量化在推理过程中可以很快地将激活函数的输出值转换为整数格式。这是动态范围量化与全整数量化相比的优势之一，因为动态范围量化不需要任何校准数据集。不过，由于动态范围量化在量化前是以浮点格式存储激活值，因此无法在仅支持低精度算术运算的定制硬件（即 NPU）上运行量化的模型。

现在来看看动态范围的计算方法，动态范围的计算方法与量化的方式相关，对称量化和非对称量化使用的计算方法略有不同。在对称量化中，通常采用的是输入数据的绝对值的最大值作为动态范围的计算方法；而在非对称量化中，通常采用最小值和最大值的差作为动态范围的计算方法。

**之前对称量化中的动态范围的计算方法就是Max方法即采取输入数据的绝对值的最大值，但是这种计算方法存在问题，就是容易受到离散点即噪声的干扰，我们的考虑改进，或者说采取其它的动态范围计算方法，也就是接下来要讲解的**<mark style="color:red;">**Histogram**</mark>**以及**<mark style="color:red;">**Entropy**</mark>**方法。**

### histogram实现&#x20;

histogram方法为什么能克服Max方法中离散点即噪声干扰问题呢？主要在于直方图统计了数据出现的频率，它可以将数据按照一定的区间进行离散化处理，并计算每个区间中数据点的数量。这种方法相对于Max方法来说，能够更好地反映数据的分布情况，从而更准确地评估数据的动态范围。我们假设数据服从正态分布，即离散点在两边，我们可以通过从两边向中间靠拢的方法，去除离散点，算法具体流程如下：&#x20;

* 首先，统计输入数据的直方图和范围&#x20;
* 然后定义左指针和右指针分别指向直方图的左边界和右边界&#x20;
* **计算当前双指针之间的直方图覆盖率，如果小于等于设定的覆盖率阈值，则返回此刻的左指针指向的直方图值，如果不满足，则需要调整双指针的值，向中间靠拢**&#x20;
* **如果当前左指针所指向的直方图值大于右指针所指向的直方图值，则右指针左移，如果当前左指针所指向的直方图值小于右指针所指向的直方图值，左指针右移**&#x20;
* 一直循环下去，直到双指针覆盖的区域满足要求

下面是该算法流程的一个简单示例：

<figure><img src="../../.gitbook/assets/图片.png" alt="" width="563"><figcaption></figcaption></figure>

其中0\~5代表算法执行的步骤，首先双指针位于直方图的左右区间，然后计算其覆盖率发现不满足设定的阈值要求，计算此时的左指针的直方图值小于右指针的直方图值，左指针右移即步骤1，继续上述步骤发现覆盖率还是不满足，且此时右指针值小故右指针左移即步骤2，以此类推到步骤3、步骤4，到步骤4计算发现覆盖率满足阈值要求，故将此时的双指针的直方图值的绝对值返回即可。

* `from scratch`

```python
def scale_cal(x):
    max_val = np.max(np.abs(x))
    return max_val / 127

def histogram_range(x):
    hist, range = np.histogram(x, 100)
    total = len(x)
    left  = 0
    right = len(hist) - 1
    limit = 0.99
    while True:
        cover_percent = hist[left:right].sum() / total
        if cover_percent <= limit:
            break

        if hist[left] > hist[right]:
            right -= 1
        else:
            left += 1
    
    left_val = range[left]
    right_val = range[right]
    dynamic_range = max(abs(left_val), abs(right_val))
    return dynamic_range / 127

if __name__ == "__main__":
    np.random.seed(8215)
    data_float32 = np.random.randn(1000).astype('float32')
    # print(f"input = {data_float32}")

    scale = scale_cal(data_float32)
    scale2 = histogram_range(data_float32)
    print(f"scale = {scale}  scale2 = {scale2}")

```

输出结果：

```bash
scale = 0.027210250614196296  scale2 = 0.020877036522692582
```

<mark style="color:red;">`Histogram`</mark><mark style="color:red;">方法虽然能够解决</mark><mark style="color:red;">`Max`</mark><mark style="color:red;">方法中的离散点噪声问题，但是使用数据直方图进行动态范围的计算，要求数据能够比较均匀地覆盖到整个动态范围内。</mark>如果数据服从类似正态分布，则直方图的结果具有参考价值，因为此时的数据覆盖动态范围的概率较高。但如果数据分布极不均匀或出现大量离散群，则直方图计算的结果可能并不准确。此时可考虑其它的动态范围计算方法。&#x20;

{% hint style="info" %}


#### 函数

```python
numpy.histogram(a, bins=10, range=None, density=None, weights=None, *, keepdims=False)
```

#### 参数说明

* `a`: 数组，要计算直方图的数据。
* `bins`: 可以是整数，表示要创建的区间数量；也可以是序列，表示区间的边界。
* `range`: 元组，表示直方图的取值范围，如果不指定，将使用数据的最小值和最大值。
* `density`: 如果为 `True`，则直方图的结果是概率密度函数，即每个 bin 的高度表示数据点落在该 bin 内的概率。
* `weights`: 数组，为每个数据点指定一个权重，用于计算加权直方图。
* `keepdims`: 如果为 `True`，输出的 bin 的维度将与输入数据的维度相同。

#### 返回值

函数返回两个数组：

* `hist`: 直方图数组，表示每个 bin 中的数据点数量（或加权和）。
* `bin_edges`: 包含 bin 边界的数组，即每个 bin 的起始和结束点。
{% endhint %}

#### Entropy

在概率论或信息论中，**KL散度**(Kullback-Leibler divergence)又称为相对熵(relative entropy)，是描述两个概率分布P和Q差异的一种方法。

$$
D_{KL}=\sum_iP(x_i)log(\frac{P(x_i)}{Q(x_i)})
$$

KL散度值越小，代表两种分布越相似，量化误差越小；反之，KL散度值越大，代表二种分布差异越大，量化误差越大。

接下来用代码生成的数据直观感受下，如果两个随机变量KL散度较小，数值上差距究竟有多大？

```python
import numpy as np
import matplotlib.pyplot as plt

def cal_kl(p, q):
    KL = 0
    for i in range(len(p)):
        KL += p[i] * np.log(p[i]/q[i])
    return KL

def kl_test(x, kl_threshod = 0.01):
    y_out = []
    while True:
        y = [np.random.uniform(1, size+1) for i in range(size)]
        y /= np.sum(y)
        kl_result = cal_kl(x, y)
        if kl_result < kl_threshod:
            print(kl_result)
            y_out = y
            plt.plot(x)
            plt.plot(y)
            break
    return y_out

if __name__ == "__main__":
    np.random.seed(8215)
    size = 10
    x = [np.random.uniform(1, size+1) for i in range(size)]
    x /= np.sum(x)
    y_out = kl_test(x, kl_threshod = 0.01)
    plt.show()
    print(x, y_out)

```

在上述示例代码中，具体实现过程如下：

首先定义了一个计算KL散度的函数cal\_kl，用于计算两个概率分布P和Q之间的KL散度 然后定义了一个kl\_test函数，用于使用Entropy方法来估计数据的动态范围。在kl\_test中，先随机生成概率分布y，将其归一化后计算与输入的概率分布x之间的KL散度，如果小于阈值，则认为当前的概率分布y最优，结束迭代。否则继续生成一个新的随机概率分布y，重复KL散度计算，直到找到满足条件的最优概率分布 最后返回找到的最优概率分布y，并可视化原始数据分布x和最优概率分布y的差异 下面是KL散度阈值为0.01时原始数据x(蓝色)和最优概率分布y(橙色)的可视化图，可以看到此时的x和y的分布比较接近：&#x20;

<figure><img src="../../.gitbook/assets/图片 (80).png" alt="" width="563"><figcaption></figcaption></figure>























### reference

* [A Case Study of Quantizing Convolutional Neural Networks for Fast Disease Diagnosis on Portable Medical Devices](https://www.mdpi.com/1424-8220/22/1/219)
* https://blog.csdn.net/qq\_40672115/article/details/129942542

