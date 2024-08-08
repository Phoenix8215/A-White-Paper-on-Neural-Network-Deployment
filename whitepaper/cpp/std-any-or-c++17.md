# 🫛 std::any|C++17

<figure><img src="../../.gitbook/assets/image.png" alt=""><figcaption></figcaption></figure>

`std::any` 是 C++ 标准库在 C++17 中引入的一项功能。它属于 C++ 标准库中的类型安全容器类，提供了一种可以类型安全地存储和操作任意类型值的方法。这在需要处理异构对象集合或在编译时未知对象类型的情况下尤其有用。

`std::any` 是一个包装类，是 `void*` 的一个理想替代品。在 C++17 标准之前编写的代码中，`any` 类可以替代许多使用 `void*` 类型的地方。众所周知，类型为 `void*` 的指针可以保存任何类型对象的地址。但类型为 `void*` 的指针并不知道它所持有对象的类型，也无法控制其生命周期。<mark style="color:red;">**C++17 引入了**</mark><mark style="color:red;">** **</mark><mark style="color:red;">**`std::any`**</mark><mark style="color:red;">**，作为**</mark><mark style="color:red;">** **</mark><mark style="color:red;">**`void*`**</mark><mark style="color:red;">** **</mark><mark style="color:red;">**的理想替代品。**</mark><mark style="color:red;">与</mark> <mark style="color:red;"></mark><mark style="color:red;">`void*`</mark> <mark style="color:red;"></mark><mark style="color:red;">不同，</mark><mark style="color:red;">`std::any`</mark> <mark style="color:red;"></mark><mark style="color:red;">可以控制其所持有对象的生命周期，并且始终知道其所持有对象的类型。</mark>

**那么，`std::any` 在哪些地方使用呢？任何使用 `void*` 的地方都可以使用 `std::any`。**

<mark style="color:red;">`std::any`</mark> <mark style="color:red;"></mark><mark style="color:red;">不是类模板。当我们创建一个</mark> <mark style="color:red;"></mark><mark style="color:red;">`std::any`</mark> <mark style="color:red;"></mark><mark style="color:red;">对象时，不需要指定它将持有的值的类型。一个</mark> <mark style="color:red;"></mark><mark style="color:red;">`std::any`</mark> <mark style="color:red;"></mark><mark style="color:red;">对象可以持有任意类型的值，同时也知道它包含的值的类型。但这是如何实现的呢？一个对象怎么能存储任意类型的值呢？秘诀在于</mark> <mark style="color:red;"></mark><mark style="color:red;">`std::any`</mark> <mark style="color:red;"></mark><mark style="color:red;">对象除了存储实际值外，还包含该值的</mark> <mark style="color:red;"></mark><mark style="color:red;">`type_info`</mark><mark style="color:red;">。</mark>

要使用 `std::any`，必须包含 `<any>` 头文件。成员函数、辅助类和非成员函数如下。

<figure><img src="../../.gitbook/assets/image (1).png" alt=""><figcaption></figcaption></figure>

`std::any` 对象可以创建来持有特定类型的值，也可以为空：

```cpp
#include <iostream>
#include <any>
#include <bitset>
#include <vector>

int main()
{
    std::any any1 = 10;                              /* int */
    std::any any2{ 1.3 };                            /* double */
    std::any any3{ "Cengizhan" };                    /* const char* */
    std::any any4{ std::string("Cengizhan") };       /* std::string*/
    std::any any5{ std::vector<char>{'a','b','c'} }; /* std::vector */
    std::any any6{};                                 /* empty */
    std::any any7;                                   /* empty */

    return 0;
}
```

就像 `std::optional` 一样，`std::variant` 可以在创建对象时使用 `std::in_place_type` 构造。同样，也可以使用 `std::make_any<T>` 工厂函数。

{% hint style="info" %}
`std::in_place_type` 是 C++17 中引入的一个功能，用于在 `std::variant`、`std::optional` 和其他类似模板类中，以原位构造对象。这意味着对象是在其最终存储位置直接构造的，而不是先构造在临时位置然后移动或复制到最终位置。这样可以提高效率，避免不必要的拷贝或移动操作。
{% endhint %}

#### **`std::make_any`**

另一种创建 `any` 类型对象的方法是使用 `make_any<>` 辅助工厂函数。

```cpp
std::any a3 = std::make_any<bool>(true);  // make_any
```

### **Memory requirement for std::any objects** <a href="#b3b5" id="b3b5"></a>

`std::any` 对象的内存需求 `std::any<T>` 使用额外的内存，因为它不知道要使用或分配哪种类型。然而，为了提升性能，它采用了 SBO（Small Buffer Optimization，小缓冲优化）方法。因此，`sizeof(std::any)` 超过了它所存储类型的大小，并且取决于编译器对 Small Buffer Optimization 的实现。

```cpp
std::cout << "std::any sizeof: " << sizeof(std::any);
```

输出：**“std::any sizeof: 16”**

我的 C++ 编译器是 gcc 11.4。运行上述代码时，可能会因编译器的不同而得到不同的值。管理 `std::any` 的内存需求是一个非常动态的过程。`std::any` 类型会根据其存储的数据的大小和类型动态分配和释放内存。

如何为 `std::any` 对象分配新值？

`any` 类对象的值可以通过类的赋值操作符函数或 `emplace<>` 函数来更改。

```cpp
#include <string>
#include <any>
#include <vector>

int main()
{
    std::any any;
    auto lmb = [](int x)->int{ return x*x; };

    any = 123;
    any = "cengizhan";
    any.emplace<std::string>("Cengizhan");
    any.emplace<std::vector<int>>(5);
    any.emplace<decltype(lmb)>( lmb );
}
```

如何清空 `std::any`？

要清空 `std::any` 对象，可以调用该类的 `reset` 函数：

```cpp
any.reset();
```

通过调用这个函数，如果类型为 `any` 的变量 `a` 非空，那么变量 `a` 所持有的对象的生命周期将会终止。完成这个过程后，变量 `a` 变为空。

我们也可以通过将使用默认构造函数创建的临时对象赋值给变量来执行清空操作。

```cpp
any = {}; 
```

`std::any` 的观察者函数 `type()` 和 `has_value()` 函数是 `std::any` 的观察者函数。

我们可以通过 `has_value()` 函数检查 `std::any` 对象是否为空。

```cpp
#include <iostream>
#include <string>
#include <any>
#include <vector>

int main()
{
    std::any any;
    std::cout<<std::boolalpha<<any.has_value()<<"\n";

    any = "cengizhan";
    std::cout<<std::boolalpha<<any.has_value()<<"\n";

    any.reset();
    std::cout<<std::boolalpha<<any.has_value()<<"\n";
}

// false
// true
// false
```

```cpp
const std::type_info& type() const noexcept;
```

`type()` 函数的定义如上所示。

通过 `type()` 函数，可以了解 `std::any` 持有的对象的类型。

```cpp
int main()
{
    std::any any;

    std::cout << std::boolalpha<<(any.type() == typeid(void)) << '\n';

    any = 15;

    std::cout << std::boolalpha<<(any.type() == typeid(int)) << '\n';
}
```

#### **`std::any_cast` 函数**

C++ 中的 `std::any_cast` 函数用于安全地提取和转换 `std::any` 对象内存储的值。`std::any` 是一个类型安全的容器，可以持有任意类型的值。要访问 `std::any` 内的值，可以使用 `std::any_cast`。

```cpp
#include <iostream>
#include <any>

int main() {

    std::any any = 3;

    try {
       std::cout << "Value inside any: " << std::any_cast<int>(any) << std::endl;
    } 
    catch (const std::bad_any_cast& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}
```

如果类型转换失败？当然会抛出异常。

#### **`std::bad_any_cast`**

如果使用 `any_cast<>` 进行类型转换失败，会抛出一个 `bad_any_cast` 类型的异常：

```cpp
#include <iostream>
#include <any>

int main() {

    std::any any = 3;

    try {
        std::cout << "Value inside any: " << std::any_cast<std::string>(any) << std::endl;
    } catch (const std::bad_any_cast& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}

// Value inside any: Error: bad any_cast
```

### 案例

* **配合函数对象**

```cpp
#include <iostream>
#include <any>
#include <functional>

// 普通函数
void say_hello() {
    std::cout << "Hello, World!" << std::endl;
}

// 函数对象（仿函数）
struct Functor {
    void operator()() const {
        std::cout << "Hello from Functor!" << std::endl;
    }
};

int main() {
    // 使用 std::any 存储普通函数
    std::any a1 = say_hello;
    // 使用 std::any 存储仿函数对象
    std::any a2 = Functor();
    // 使用 std::any 存储 lambda 表达式
    std::any a3 = [] { std::cout << "Hello from Lambda!" << std::endl; };

    // 调用存储的普通函数
    std::any_cast<void(*)()>(a1)();

    // 调用存储的仿函数对象
    std::any_cast<Functor>(a2)();

    // 调用存储的 lambda 表达式
    std::any_cast<std::function<void()>>(a3)();

    return 0;
}

```

```cpp
#include <iostream>
#include <any>
#include <string>

void printValue(const std::any& data) {
    if (data.type() == typeid(int)) {
        std::cout << "Value: " << std::any_cast<int>(data) << std::endl;
    } else if (data.type() == typeid(double)) {
        std::cout << "Value: " << std::any_cast<double>(data) << std::endl;
    } else if (data.type() == typeid(std::string)) {
        std::cout << "Value: " << std::any_cast<std::string>(data) << std::endl;
    } else {
        std::cout << "Unknown type" << std::endl;
    }
}

int main() {
    std::any intValue = 42;
    std::any doubleValue = 3.14;
    std::any stringValue = std::string("Hello, World");

    printValue(intValue);
    printValue(doubleValue);
    printValue(stringValue);

    return 0;
}

```

**使用 `void*` 的代码**

传统上，如果你想要存储任意类型的值并在运行时处理它们，你可能会使用 `void*` 和 `type_info` 来实现。这种方法需要手动管理类型信息，并且容易出错。

```cpp
#include <iostream>
#include <string>
#include <typeinfo>

void printValue(void* data, const std::type_info& type) {
    if (type == typeid(int)) {
        std::cout << "Value: " << *static_cast<int*>(data) << std::endl;
    } else if (type == typeid(double)) {
        std::cout << "Value: " << *static_cast<double*>(data) << std::endl;
    } else if (type == typeid(std::string)) {
        std::cout << "Value: " << *static_cast<std::string*>(data) << std::endl;
    } else {
        std::cout << "Unknown type" << std::endl;
    }
}

int main() {
    int intValue = 42;
    double doubleValue = 3.14;
    std::string stringValue = "Hello, World";

    printValue(&intValue, typeid(int));
    printValue(&doubleValue, typeid(double));
    printValue(&stringValue, typeid(std::string));

    return 0;
}

```

**使用 `std::any` 的代码**

使用 `std::any` 可以简化这一过程，提供类型安全的存储和访问，并且不需要手动管理类型信息。

```cpp
#include <iostream>
#include <any>
#include <string>

void printValue(const std::any& data) {
    if (data.type() == typeid(int)) {
        std::cout << "Value: " << std::any_cast<int>(data) << std::endl;
    } else if (data.type() == typeid(double)) {
        std::cout << "Value: " << std::any_cast<double>(data) << std::endl;
    } else if (data.type() == typeid(std::string)) {
        std::cout << "Value: " << std::any_cast<std::string>(data) << std::endl;
    } else {
        std::cout << "Unknown type" << std::endl;
    }
}

int main() {
    std::any intValue = 42;
    std::any doubleValue = 3.14;
    std::any stringValue = std::string("Hello, World");

    printValue(intValue);
    printValue(doubleValue);
    printValue(stringValue);

    return 0;
}

```

### 总结

`std::any`是一个可以存储任何可拷贝类型的容器，C 语言中通常使用`void*`实现类似的功能，与`void*`相比，`std::any`具有两点优势：

1. <mark style="color:red;">`std::any`</mark><mark style="color:red;">更安全：在类型 T 被转换成</mark><mark style="color:red;">`void*`</mark><mark style="color:red;">时，T 的类型信息就已经丢失了</mark>，在转换回具体类型时程序<mark style="color:red;">无法判断当前的</mark><mark style="color:red;">`void*`</mark><mark style="color:red;">的类型是否真的是 T，容易带来安全隐患。</mark>而`std::any`会存储类型信息，`std::any_cast`是一个安全的类型转换。
2. `std::any`管理了对象的生命周期，在`std::any`析构时，会将存储的对象析构，而`void*`则需要手动管理内存。

`std::any`应当很少是程序员的第一选择，在已知类型的情况下，`std::optional`, `std::variant`和继承都是比它更高效、更合理的选择。只有当对类型完全未知的情况下，才应当使用`std::any`，比如动态类型文本的解析或者业务逻辑的中间层信息传递。

### reference

* [https://cengizhanvarli.medium.com/what-is-std-any-in-c-76f2575ad37b](https://cengizhanvarli.medium.com/what-is-std-any-in-c-76f2575ad37b)
* [https://mp.weixin.qq.com/s/jlM1NWRNpoOvW2qrBtxflQ](https://mp.weixin.qq.com/s/jlM1NWRNpoOvW2qrBtxflQ)
