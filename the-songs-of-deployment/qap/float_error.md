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

{% hint style="info" %}
一些由于浮点误差导致的严重事故：[https://www-users.cse.umn.edu/\~arnold/disasters/disasters.html?source=post\_page-----5c879c480982--------------------------------](https://www-users.cse.umn.edu/\~arnold/disasters/disasters.html?source=post\_page-----5c879c480982--------------------------------)
{% endhint %}

### 浮点误差的解决方法

* 使用`double`而不是`float`类型
* 当添加浮点数的时候，先加最小的数
* 添加浮点数列表的时候，首先对他们进行排序

### 在 C++ 中的浮点数精度问题 <a href="#er-zai-c-zhong-de-fu-dian-shu-jing-du-wen-ti" id="er-zai-c-zhong-de-fu-dian-shu-jing-du-wen-ti"></a>

首先，让我们从一段程序开始这个问题

```cpp
#include <iostream>

int main() {
    float a = 10.0;
    float b = 0.1;
    float c = a - b;

    std::cout << "c = " << c << "\n";

    return 0;
}

// 输出结果：9.9
```

这看似没什么问题，实则问题隐藏于冰山之下。现在，让我们来解开它！

再次尝试下面的程序：

```cpp
#include <iostream>
#include <iomanip>

int main() {
    float a = 10.0;
    float b = 0.1;
    float c = a - b;

    std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(23)
        << "c = " << c << "\n";

    return 0;
}

// 输出结果：c = 9.89999961853027343750000
```

可以看到，当我们输出小数后 23 位后就能发现这个精度缺失问题了。

第一个例子之所以能得到正确的答案是因为 std::cout 输出流对小数做了四舍五入操作，通过处理给出了正确答案。

看到这，你或许觉得这有什么问题，能得到正确答案不就行了？其实不然，现在让我们将 `float a = 10.0;` 换为 `float a = 500000.0`，我们再次运行程序，看看能得到什么（使用默认输出位数）！

```cpp
c = 500000
```

好家伙，懵了吧（500000.0 并不是绝对的，准确来说要确保 a 足够的大）！

很明显，这出现了问题！当我们输出小数后 23 位时你就会得到：

```cpp
c = 499999.90625000000000000000000
```

所以，即使你可以通过**允许误差**的方式来判断浮点数是否相等（大概类似于使用 `a - b <= 1e-12` 来近似代替 `a - b == 0.0` 这样的判别式），你也无法避免输出显示的问题。而这一切都源于浮点数的精度丢失问题。

### 解决 C++ 中的浮点数精度问题 <a href="#san-jie-jue-c-zhong-de-fu-dian-shu-jing-du-wen-ti" id="san-jie-jue-c-zhong-de-fu-dian-shu-jing-du-wen-ti"></a>

目前，我尝试的解决办法为使用 [Boost 库](https://www.boost.org/)中的 `boost::multiprecision::cpp_dec_float_50` 来代替 `float` 或 `double` 类型。

```cpp
#include <iostream>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/cpp_int.hpp>

namespace mp = boost::multiprecision;     // 命名空间重命名——缩短命名空间
typedef mp::cpp_dec_float_50    float50;  // 变量类型名重命名——缩短变量类型名
typedef mp::cpp_dec_float_100    float100;// 变量类型名重命名——缩短变量类型名

int main() {
    float50 a0("50000.0");                // 推荐声明方式
    float50 b0("10.012");
    float50 c0 = a0 - b0;

    float50 a1(50000.0);                  // 因为 50000.0 不是字符串，所以已经出现了精度丢失。
    float50 b1(10.012);                   // 而用精度丢失后的数再创建高精度变量已经无意义。
    float50 c1 = a1 - b1;

    double a2 = 50000.0;                  // 演示精度丢失
    double b2 = 10.012;
    double c2 = a2 - b2;

    std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(25)
        << "a0\t=\t" << a0 << "\n"
        << "b0\t=\t" << b0 << "\n"
        << "c0\t=\t" << c0 << "\n\n"
        << "a1\t=\t" << a1 << "\n"
        << "b1\t=\t" << b1 << "\n"
        << "c1\t=\t" << c1 << "\n\n"
        << "a2\t=\t" << a2 << "\n"
        << "b2\t=\t" << b2 << "\n"
        << "c2\t=\t" << c2 << "\n\n";
 
    return 0;
}

/*
a0      =       50000.0000000000000000000000000
b0      =       10.0120000000000000000000000
c0      =       49989.9880000000000000000000000

a1      =       50000.0000000000000000000000000
b1      =       10.0120000000000004547473509
c1      =       49989.9879999999999995452526491

a2      =       50000.0000000000000000000000000
b2      =       10.0120000000000004547473509
c2      =       49989.9879999999975552782416344
*/
```

很明显，精度丢失问题在 Boost 库中得到了很好的解决。当然，Boost 库还有很多其他的关于浮点数操作的优化，而且基本都是基于源标准库的模式重写的，所以使用起来的非常顺手（注意命名空间即可）。

### reference

* [https://medium.com/@kusal95/computer-floating-point-arithmetic-and-round-off-errors-5c879c480982](https://medium.com/@kusal95/computer-floating-point-arithmetic-and-round-off-errors-5c879c480982)
* [https://seayj.cn/articles/6d33/](https://seayj.cn/articles/6d33/)
