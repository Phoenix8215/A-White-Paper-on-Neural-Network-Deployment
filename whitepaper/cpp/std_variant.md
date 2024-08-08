# 🌶️ std::variant|C++17

**`std::variant` 是 C++17 中引入的标准库类型，用于表示多个可能类型中的一个。它类似于联合体（union），但更安全且易于使用。`std::variant` 提供了一种类型安全的方式来处理不同类型的值。**

如果你用过 C 语言，你肯定用过 union。在 C 编程语言中，union 是一种复合数据类型，允许在同一内存位置存储不同类型的数据。<mark style="color:red;">与结构体一样，union 可以有多个成员，但与结构体不同的是，union 分配的内存只够容纳其最大的成员。这就意味着，union 的所有成员共享同一个内存空间。</mark>

> <mark style="color:red;">**`std::variant`**</mark><mark style="color:red;">**适用于之前使用**</mark><mark style="color:red;">**`union`**</mark><mark style="color:red;">**的场景。**</mark>

让我们看看下面的 C 代码：

```c
#include <stdint.h>
#include <stdio.h>

union
{
    struct
    {
        uint8_t low  : 4; // 表示该位域占用4个位，这使得我们能够直接操作和访问字节的高低半部分
        uint8_t high : 4;
    }nibles;

    uint8_t bytes;
}myByte;

int main()
{
    myByte.bytes = 0xAB ;

    printf("High Nible : 0x%X\n",myByte.nibles.high);
    printf("Low Nible : 0x%X\n",myByte.nibles.low);

    return 0;
}
```

输出结果：

```c
High Nible : 0xA
Low Nible : 0xB
```

{% hint style="info" %}
&#x20;`1010`，对应十六进制的 `A`

&#x20;`1011`，对应十六进制的 `B`
{% endhint %}

如上所述，它分配的内存与联合体中最长的元素一样多。所以是 1 个字节。它将写入其中的变量放在相同的地址。这样，如果名为 bytes 的变量发生变化，那么 union 中的另一个元素，即名为 nibles 的结构也会发生变化。

但是联合体在提供便利的同时也带来了一些麻烦，看下面的代码：

```c
#include <iostream>

union MyUnion {
    int intValue;
    double doubleValue;
};

int main() {
    MyUnion myUnion;

    myUnion.doubleValue = 3.14;

    std::cout << "Integer Value: " << myUnion.intValue << "\n";

    return 0;
}
```

我们将 3.14 赋值给 double 变量。然后我们尝试访问 int 变量。结果如下:

```c
Integer Value: 1374389535
```

结果有些莫名其妙，事实上我们不应该访问int类型的那个变量，<mark style="color:red;">在 C++17 中出现的 std::variant，就是为了解决这个潜在的问题，他为我们提供了类型安全的联合体。</mark>

### 什么是`std::variant`

std::variant 是 C++17 的一个特性，它提供了一个类型安全的联合体。它是 C++ 标准库的一部分，定义在 头文件中。

在 C++ 中，union 允许你在同一内存位置存储不同类型的数据，但它不提供类型安全。std::variant 是对传统 union 的改进，因为它确保了类型安全，并为处理不同类型的值提供了一种方便的方法。

让我们先测试一下类型安全功能。

```c
#include <iostream>
#include <variant>

int main() {
    std::variant<int, double> myVariant;

    myVariant = 42; /* Store an int */

    std::cout << std::get<double>(myVariant) << std::endl;

    return 0;
}

// g++ -std=c++17 demo.cpp -o demo
```

输出结果为：

```c
terminate called after throwing an instance of 'std::bad_variant_access'
  what():  Unexpected index
Aborted (core dumped)
```

我们创建了一个包含 int 和 double 的 std::variant 变量。我们将值 42 赋给 int 变量。然后，我们尝试访问 double 变量，得到了一个错误。就是这样！这就是类型安全！

我们再来看一个使用 std::variant 的例子。

```c
#include <iostream>
#include <variant>
#include <string>

int main() {
    /* Define std::variant; it can hold either an int or a std::string value */
    std::variant<int, std::string> myVariant;

    /* Assign an int to std::variant */
    myVariant = 42;

    /* Assign a std::string to std::variant */
    myVariant = "Hello, Variant!";

    /* Retrieve and use the value from std::variant */
    try {
        /* It's important to check which type is stored in std::variant before retrieving the value */
        if (std::holds_alternative<int>(myVariant))
        {
            std::cout << "Value as int: " << std::get<int>(myVariant) << std::endl;
        }
        else if (std::holds_alternative<std::string>(myVariant))
        {
            std::cout << "Value as string: " << std::get<std::string>(myVariant) << std::endl;
        }
        else
        {
            std::cout << "Unknown type!" << std::endl;
        }
    } catch (const std::bad_variant_access& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}
```

输出结果是：

```c
Value as string: Hello, Variant!
```

在主函数中，定义了一个名为 myVariant 的 std::variant，它可以保存一个 int 或一个 std::string。一个 int 值（本例中为 42）被赋值给 std::variant。然后一个 `std::string ("Hello, Variant!")`被分配给同一个 std::variant。这表明 std::variant 可以灵活地保存不同类型的值。try-catch 块用于处理访问存储在 std::variant 中的值时可能出现的异常。它使用 std::holds\_alternative 检查存储值的类型，然后提取并打印相应的值。如果遇到未知类型，则会打印错误信息。如果出现异常（例如试图访问错误的类型），它会捕获异常并使用 std::cerr 打印错误信息。

### `std::variant`公共成员函数 <a href="#id-1062" id="id-1062"></a>

#### 默认构造函数：

```c
std::variant<int, double, std::string> myVariant; /* Default construction */
```

#### 赋值构造函数

```c
std::variant<int, double> myVariant;
myVariant = 42; /* Assigns an int */
```

#### 使用 `index` 方法获取当前持有类型的索引

`index` 方法返回 `std::variant` 当前持有类型在模板参数列表中的索引。

```cpp
#include <variant>
#include <iostream>
#include <string>

int main() {
    std::variant<int, float, std::string> var;

    var = 42;
    std::cout << "Current index: " << var.index() << std::endl;  // 输出 0

    var = 3.14f;
    std::cout << "Current index: " << var.index() << std::endl;  // 输出 1

    var = "Hello";
    std::cout << "Current index: " << var.index() << std::endl;  // 输出 2

    return 0;
}
```

在这个示例中，`index` 方法返回了 `std::variant` 持有值的索引。索引 `0` 表示 `int` 类型，索引 `1` 表示 `float` 类型，索引 `2` 表示 `std::string` 类型。

#### 使用 `std::variant_alternative` 获取类型

`std::variant_alternative` 可以用来获取 `std::variant` 模板参数列表中指定索引处的类型。

```cpp
#include <variant>
#include <iostream>
#include <string>
#include <type_traits>

int main() {
    std::variant<int, float, std::string> var;

    // 获取索引 0 对应的类型
    using T0 = std::variant_alternative<0, decltype(var)>::type;
    std::cout << "Type at index 0: " << typeid(T0).name() << std::endl;  // 输出 int

    // 获取索引 1 对应的类型
    using T1 = std::variant_alternative<1, decltype(var)>::type;
    std::cout << "Type at index 1: " << typeid(T1).name() << std::endl;  // 输出 float

    // 获取索引 2 对应的类型
    using T2 = std::variant_alternative<2, decltype(var)>::type;
    std::cout << "Type at index 2: " << typeid(T2).name() << std::endl;  // 输出 std::string

    return 0;
}
```

在这个示例中，`std::variant_alternative` 用来获取 `std::variant` 模板参数列表中指定索引处的类型。`std::variant_alternative<0, decltype(var)>::type` 获取了 `var` 的第一个类型 `int`，`std::variant_alternative<1, decltype(var)>::type` 获取了第二个类型 `float`，依此类推。

#### 综合示例

```cpp
#include <variant>
#include <iostream>
#include <string>
#include <typeinfo>

int main() {
    std::variant<int, float, std::string> var = "Hello";

    std::cout << "Current index: " << var.index() << std::endl;  // 输出 2

    if (var.index() == 0) {
        using T = std::variant_alternative<0, decltype(var)>::type;
        std::cout << "Current type: " << typeid(T).name() << " with value " << std::get<T>(var) << std::endl;
    } else if (var.index() == 1) {
        using T = std::variant_alternative<1, decltype(var)>::type;
        std::cout << "Current type: " << typeid(T).name() << " with value " << std::get<T>(var) << std::endl;
    } else if (var.index() == 2) {
        using T = std::variant_alternative<2, decltype(var)>::type;
        std::cout << "Current type: " << typeid(T).name() << " with value " << std::get<T>(var) << std::endl;
    }

    return 0;
}
```

#### &#x20;`Valueless by Exception` <a href="#id-8af1" id="id-8af1"></a>

`valueless_by_exception()` 用于查看variant变量是否赋值了。

```cpp
std::variant<int, double> myVariant;
if (myVariant.valueless_by_exception()) {
    /* Handle the case where the variant has no value */
}
```

#### `Swap` <a href="#d47e" id="d47e"></a>

用于交换两个`std::variant`的数值(**注意:需要类型一致)**

```cpp
std::variant<int, double> var1 = 42;
std::variant<int, double> var2 = 3.14;
std::swap(var1, var2);
```

#### **`empalce()`**

emplace() 使用提供的参数进行**就地构造**。

```c
std::variant<int, std::string> myVariant;
myVariant.emplace<std::string>("Hello");
```

#### `visit()`

<mark style="color:red;">`std::visit`</mark> <mark style="color:red;"></mark><mark style="color:red;">函数用于访问</mark> <mark style="color:red;"></mark><mark style="color:red;">`std::variant`</mark> <mark style="color:red;"></mark><mark style="color:red;">中存储的值。它通过接受一个访问者（visitor）对象来对</mark> <mark style="color:red;"></mark><mark style="color:red;">`std::variant`</mark> <mark style="color:red;"></mark><mark style="color:red;">中当前存储的值进行操作。访问者对象可以是一个函数对象、lambda 表达式或具有重载</mark> <mark style="color:red;"></mark><mark style="color:red;">`operator()`</mark> <mark style="color:red;"></mark><mark style="color:red;">的结构体。</mark>

`std::visit` 的基本用法包括：

1. 定义一个 `std::variant` 对象，包含多个可能的类型。
2. 创建一个访问者对象，定义对每种可能类型的操作。
3. 使用 `std::visit` 传递访问者对象和 `std::variant` 对象。

#### 详细示例

以下是详细的示例，展示如何使用 `std::visit` 来访问和操作 `std::variant` 中存储的值。

**示例 1：基本用法**

```cpp
#include <variant>
#include <iostream>
#include <string>

int main() {
    std::variant<int, float, std::string> var = "Hello, World!";

    auto visitor = [](auto&& arg) {
        std::cout << "Value: " << arg << std::endl;
    };

    std::visit(visitor, var);  // 输出: Value: Hello, World!

    var = 42;
    std::visit(visitor, var);  // 输出: Value: 42

    var = 3.14f;
    std::visit(visitor, var);  // 输出: Value: 3.14

    return 0;
}
```



**示例 2：使用结构体作为访问者**

```cpp
#include <variant>
#include <iostream>
#include <string>

struct Visitor {
    void operator()(int i) const {
        std::cout << "Integer: " << i << std::endl;
    }

    void operator()(float f) const {
        std::cout << "Float: " << f << std::endl;
    }

    void operator()(const std::string& s) const {
        std::cout << "String: " << s << std::endl;
    }
};

int main() {
    std::variant<int, float, std::string> var;

    var = 42;
    std::visit(Visitor{}, var);  // 输出: Integer: 42

    var = 3.14f;
    std::visit(Visitor{}, var);  // 输出: Float: 3.14

    var = "Hello, World!";
    std::visit(Visitor{}, var);  // 输出: String: Hello, World!

    return 0;
}
```



**示例 3：访问多个 `std::variant`**

`std::visit` 还可以用于同时访问多个 `std::variant` 对象。此时需要一个接受多个参数的访问者对象。

```cpp
#include <variant>
#include <iostream>
#include <string>

struct Visitor {
    void operator()(int i, float f) const {
        std::cout << "Integer: " << i << ", Float: " << f << std::endl;
    }

    void operator()(int i, const std::string& s) const {
        std::cout << "Integer: " << i << ", String: " << s << std::endl;
    }

    void operator()(float f, const std::string& s) const {
        std::cout << "Float: " << f << ", String: " << s << std::endl;
    }

    void operator()(const auto& a, const auto& b) const {
        std::cout << "Other types: " << a << ", " << b << std::endl;
    }
};

int main() {
    std::variant<int, float, std::string> var1 = 42;
    std::variant<int, float, std::string> var2 = "Hello, World!";

    std::visit(Visitor{}, var1, var2);  // 输出: Integer: 42, String: Hello, World!

    var2 = 3.14f;
    std::visit(Visitor{}, var1, var2);  // 输出: Integer: 42, Float: 3.14

    return 0;
}
```

### **`std::variant`的优势** <a href="#ba8b" id="ba8b"></a>

1. **类型安全**

如前所述，当访问错误的类型时，传统的联合体类型可能会导致意想不到的错误。

```c
std::variant<int, double, std::string> myVariant = 42;
// Type-safe access
std::cout << std::get<int>(myVariant) << std::endl;
```

2. **可读性更高**

明确表达了该变量有多种数据类型。

```c
std::variant<int, double, std::string> myVariant = "Hello";
// Improved readability
if (std::holds_alternative<std::string>(myVariant)) {
    std::cout << std::get<std::string>(myVariant) << std::endl;
}
```

3. **可以处理函数有多个返回值类型的情况**

```c
#include <iostream>
#include <variant>
#include <string>

std::variant<int, double, std::string> getValue(bool flag) {
    if (flag) {
        return 42;         // 返回int类型
    } else {
        return "Hello";    // 返回std::string类型
    }
}

int main() {
    std::variant<int, double, std::string> value = getValue(true);  // 调用函数并获取返回值

    if (std::holds_alternative<int>(value)) {
        std::cout << "Returned int: " << std::get<int>(value) << std::endl;
    } else if (std::holds_alternative<std::string>(value)) {
        std::cout << "Returned string: " << std::get<std::string>(value) << std::endl;
    }

    value = getValue(false);  // 再次调用函数，获取不同类型的返回值

    if (std::holds_alternative<int>(value)) {
        std::cout << "Returned int: " << std::get<int>(value) << std::endl;
    } else if (std::holds_alternative<std::string>(value)) {
        std::cout << "Returned string: " << std::get<std::string>(value) << std::endl;
    }

    return 0;
}

```

```c
#include <iostream>
#include <variant>

int main() {
    auto example = [](int x) -> std::variant<int, std::string> {
        if (x > 0) {
            return x;
        } else {
            return "Negative or Zero";
        }
    };

    auto result1 = example(10);
    auto result2 = example(-5);

    if (std::holds_alternative<int>(result1)) {
        std::cout << "Result1: " << std::get<int>(result1) << std::endl;
    } else {
        std::cout << "Result1: " << std::get<std::string>(result1) << std::endl;
    }

    if (std::holds_alternative<int>(result2)) {
        std::cout << "Result2: " << std::get<int>(result2) << std::endl;
    } else {
        std::cout << "Result2: " << std::get<std::string>(result2) << std::endl;
    }

    return 0;
```

4. **与标准库算法兼容**

```c
std::vector<std::variant<int, double, std::string>> data;
// Can use standard algorithms with std::variant
std::for_each(data.begin(), data.end(), [](auto& item) {
    std::visit([](auto&& arg){ std::cout << arg << std::endl; }, item);
});
```

### _总结_

`std::variant<T, U, ...>`代表一个多类型的容器，容器中的值是制定类型的一种，是通用的 Sum Type，对应 Rust 的`enum`。是一种类型安全的`union`，所以也叫做`tagged union`。与`union`相比有两点优势：

1. 可以存储复杂类型，而 union 只能直接存储基础的 POD 类型，对于如`std::vector`和`std::string`就等复杂类型则需要用户手动管理内存。
2. 类型安全，variant 存储了内部的类型信息，所以可以进行安全的类型转换，c++17 之前往往通过`union`+`enum`来实现相同功能。

通过使用`std::variant<T, Err>`，用户可以实现类似 Rust 的`std::result`，即在函数执行成功时返回结果，在失败时返回错误信息，上文的例子则可以改成:

```
std::variant<ReturnType, Err> func(const string& in) {
    ReturnType ret;
    if (in.size() == 0)
        return Err{"input is empty"};
    // ...
    return {ret};
}
```

需要注意的是，c++17 只提供了一个库级别的 variant 实现，没有对应的模式匹配(Pattern Matching)机制，而最接近的`std::visit`又缺少编译器的优化支持，所以在 c++17 中`std::variant`并不好用，跟 Rust 和函数式语言中出神入化的 Sum Type 还相去甚远，但是已经有许多围绕`std::variant`的提案被提交给 c++委员会探讨，包括模式匹配，`std::expected`等等。

### reference

* [https://cengizhanvarli.medium.com/std-variant-in-c-c2fc83d34efe](https://cengizhanvarli.medium.com/std-variant-in-c-c2fc83d34efe)
* [https://mp.weixin.qq.com/s/jlM1NWRNpoOvW2qrBtxflQ](https://mp.weixin.qq.com/s/jlM1NWRNpoOvW2qrBtxflQ)
