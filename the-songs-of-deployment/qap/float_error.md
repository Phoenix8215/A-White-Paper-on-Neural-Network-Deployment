# 🫰 浮点运算产生的误差

### 如何将十进制转换为二进制

1. 整数🌰

<figure><img src="../../.gitbook/assets/图片 (58).png" alt="" width="563"><figcaption></figcaption></figure>

所以100的二进制表示就是1100100，大家也可以用`windows`自带的程序员计算器多试试几个例子

2. 小数🌰

<figure><img src="../../.gitbook/assets/1 _UDEaOX40DMPegaL47230A (1).webp" alt="" width="466"><figcaption></figcaption></figure>

所以0.375的二进制表示为0.011

在看一个稍微特殊的例子：0.1的二进制应该如何表示呢？

<figure><img src="../../.gitbook/assets/1 65lZwfJ7YpHe_j5xt4BKzA.webp" alt="" width="249"><figcaption></figcaption></figure>

可以看出是无限循环的，0.1的二进制表示为`000110011001100...`,于是误差就产生了。

### IEEE754标准下下的9.1

1. 二进制表示9为：`1001`
2. 二进制表示0.1为：`0001100110011001100...`
3. 9.1二进制转换后为`1001.0001100110011001100...`，写成科学表示法就是：`1.0010001100110011001100… x 2³`
4. 指数位为3+127 = 130，二进制表示为`10000010`

<figure><img src="../../.gitbook/assets/1 tknf_RSUU5CXyUYLsNwAsQ.webp" alt=""><figcaption></figcaption></figure>

{% hint style="info" %}
由于指数可以是负数，如果算出来的指数不足八位，需要在高位即左边补0至八位。
{% endhint %}

4. 分数位为`0010001100110011001100…`，如果不够23位则在后面补0
5. 最终9.1按照IEEE754表示如下

<figure><img src="../../.gitbook/assets/1 qRb9t3mYCqBQCH1SXFh8bQ (1).webp" alt=""><figcaption></figcaption></figure>

6.  检验下，再转回到十进制

    <figure><img src="../../.gitbook/assets/1 ewWDv5NEibcQ246U7bvzEw.webp" alt=""><figcaption></figcaption></figure>

正如所看到的，我们首先将 9.1 转换为 IEEE 754 标准，然后将 IEEE 754 值转换为十进制值，得到的却是9.10000038，这就是所谓的浮点误差。

0.1 和 0.2 这两个数值在二进制浮点数中并没有精确的表示，因此，当你将十进制转换成二进制，再将二进制转换成十进制时，就会损失精度。

所以当我们编写一个算法时，应当要十分注意这种浮点数带来的误差。(曾经在国内某个二线大厂的代码中也发现了这种错误😅)

