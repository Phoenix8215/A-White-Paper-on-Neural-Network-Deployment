# 🌲 C++11右值引用和移动语义

## 右值引用和移动语义

作用：C++11中引用了右值引用和移动语义，**可以避免无谓的复制，提高了程序性能。**

左值是表达式结束后仍然存在的持久对象，**右值是指表达式结束时就不存在的临时对象。**

* <mark style="color:red;">区分左值和右值的便捷方法是看能不能对表达式取地址，如果能则为左值，否则为右值；</mark>
* <mark style="color:red;">将亡值是C++11新增的、与右值引用相关的表达式，比如：将要被移动的对象、T&&函数返回的值、std::move返回值和转换成T&&的类型的转换函数返回值。</mark>

**C++11中的所有的值必将属于左值、将亡值、纯右值三者之一，将亡值和纯右值都属于右值。 区分表达式的左右值属性：如果可对表达式用&符取址，则为左值，否则为右值。**

左值 lvalue 是有标识符、可以取地址的表达式，最常见的情况有：

* **变量、函数或数据成员的名字**
* **返回左值引用的表达式，如 ++x、x = 1、cout << ' '**
* **字符串字面量如 "hello world"**

纯右值 prvalue 是没有标识符、不可以取地址的表达式，一般也称之为“临时对象”。最常见的情况有：

* **返回非引用类型的表达式，如 x++、x + 1、make\_shared(42)**
* **除字符串字面量之外的字面量，如 42、true**

### `&&`的特性

右值引用就是对一个右值进行引用的类型。因为右值没有名字，所以我们只能通过引用的方式找到它。 <mark style="color:red;">无论声明左值引用还是右值引用都必须立即进行初始化，因为引用类型本身并不拥有所把绑定对象的内存，只是该对象的一个别名。</mark>

**通过右值引用的声明，该右值又“重获新生”，其生命周期其生命周期与右值引用类型变量的生命周期一 样，只要该变量还活着，该右值临时量将会一直存活下去。**

在C++中，并不是所有情况下 && 都代表是一个右值引用，具体的场景体现在模板和自动类型推导中，<mark style="color:red;">如果是模板参数需要指定为</mark><mark style="color:red;">`T&&`</mark><mark style="color:red;">，如果是自动类型推导需要指定为</mark><mark style="color:red;">`auto &&`</mark><mark style="color:red;">，在这两种场景下 &&被称作未定的引用类型。</mark>另外还有一点需要额外注意`const T&&`表示一个右值引用，不是未定引用类型。

先来看第一个例子，在函数模板中使用`&&`:

```cpp
template<typename T>
void f(T&& param);
void f1(const T&& param);
f(10); 	
int x = 10;
f(x); 
f1(x);	// error, x是左值
f1(10); // ok, 10是右值
```

在上面的例子中函数模板进行了自动类型推导，需要通过传入的实参来确定参数param的实际类型。

* 第4行中，对于`f(10)`来说传入的实参10是右值，因此`T&&`表示右值引用
* 第6行中，对于`f(x)`来说传入的实参是x是左值，因此`T&&`表示左值引用
* 第7行中，`f1(x)`的参数是`const T&&`不是未定引用类型，不需要推导，本身就表示一个右值引用

再来看第二个例子:

```cpp
int main()
{
    int x = 520, y = 1314;
    auto&& v1 = x;
    auto&& v2 = 250;
    decltype(x)&& v3 = y;   // error
    cout << "v1: " << v1 << ", v2: " << v2 << endl;
    return 0;
};
```

* 第4行中 `auto&&`表示一个整形的左值引用
* 第5行中 `auto&&`表示一个整形的右值引用
* 第6行中`decltype(x)&&`等价于`int&&`是一个右值引用不是未定引用类型，y是一个左值，`不能使用左值初始化一个右值引用类型。`

<mark style="color:red;">由于上述代码中存在</mark><mark style="color:red;">`T&&`</mark><mark style="color:red;">或者</mark><mark style="color:red;">`auto&&`</mark><mark style="color:red;">这种未定引用类型，当它作为参数时，有可能被一个右值引用初始化，也有可能被一个左值引用初始化，在进行类型推导时右值引用类型（&&）会发生变化，这种变化被称为引用折叠。在C++11中引用折叠的规则如下：</mark>

* <mark style="color:red;">通过右值推导</mark> <mark style="color:red;"></mark><mark style="color:red;">`T&&`</mark> <mark style="color:red;"></mark><mark style="color:red;">或者</mark> <mark style="color:red;"></mark><mark style="color:red;">`auto&&`</mark> <mark style="color:red;"></mark><mark style="color:red;">得到的是一个右值引用类型</mark>
* 通过非右值（**右值引用**、左值、左值引用、**常量右值引用**、常量左值引用）推导 T&& 或者 auto&& 得到的是一个左值引用类型

```cpp
int&& a1 = 5;
auto&& bb = a1;
auto&& bb1 = 5;

int a2 = 5;
int &a3 = a2;
auto&& cc = a3;
auto&& cc1 = a2;

const int& s1 = 100;
const int&& s2 = 100;
auto&& dd = s1;
auto&& ee = s2;

const auto&& x = 5;
```

* 第2行：`a1`为右值引用，推导出的`bb`为`左值引用`类型
* 第3行：`5`为右值，推导出的`bb1`为`右值引用`类型
* 第7行：`a3`为左值引用，推导出的`cc`为`左值引用`类型
* 第8行：`a2`为左值，推导出的`cc1`为`左值引用`类型
* 第12行：`s1`为常量左值引用，推导出的`dd`为`常量左值引用`类型
* 第13行：`s2`为常量右值引用，推导出的`ee`为`常量左值引用`类型
* 第15行：`x`为右值引用，不需要推导，只能通过右值初始化

再看最后一个例子，代码如下：

```cpp
#include <iostream>
using namespace std;

void printValue(int &i)
{
    cout << "l-value: " << i << endl;
}

void printValue(int &&i)
{
    cout << "r-value: " << i << endl;
}

void forward(int &&k)
{
    printValue(k);
}

int main()
{
    int i = 520;
    printValue(i);
    printValue(1314);
    forward(250);

    return 0;
};
```

测试代码输出的结果如下:

```cpp
l-value: 520
r-value: 1314
l-value: 250
```

根据测试代码可以得知，编译器会根据传入的参数的类型（左值还是右值）调用对应的重置函数（printValue），函数forward()接收的是一个右值，但是在这个函数中调用函数printValue()时，参数k变成了一个命名对象，编译器会将其当做左值来处理。

看下面这两张图可能更容易理解：

<figure><img src="../../.gitbook/assets/图片 (62).png" alt=""><figcaption></figcaption></figure>

> 图片来源：[https://blog.csdn.net/ChaoFreeandeasy\_/article/details/130229252?spm=1001.2014.3001.5501](https://blog.csdn.net/ChaoFreeandeasy\_/article/details/130229252?spm=1001.2014.3001.5501)

&& 的总结如下：

1. 左值和右值是独立于它们的类型的，右值引用类型可能是左值也可能是右值。
2. <mark style="color:red;">auto&& 或函数参数类型自动推导的 T&& 是一个未定的引用类型，被称为</mark> <mark style="color:red;"></mark><mark style="color:red;">`universal references`</mark><mark style="color:red;">， 它可能是左值引用也可能是右值引用类型，取决于初始化的值类型。</mark>
3. 所有的右值引用叠加到右值引用上仍然是一个右值引用，其他引用折叠都为左值引 用。当 T&& 为 模板参数时，输入左值，它会变成左值引用，而输入右值时则变为具名的右 值引用。
4. 编译器会将已命名的右值引用视为左值，而将未命名的右值引用视为右值。

### 右值引用优化性能，避免深拷贝

对于含有堆内存的类，我们需要提供深拷贝的拷贝构造函数，如果使用默认构造函数，会导致堆内存的重复删除，比如下面的代码：

```cpp
#include <iostream>
using namespace std;
class A
{
    public:
    A() :m_ptr(new int(0)) {
        cout << "constructor A" << endl;
    }
    ~A(){
        cout << "destructor A, m_ptr:" << m_ptr << endl;
        delete m_ptr;
        m_ptr = nullptr;
    }
    private:
    int* m_ptr;
};
// 为了避免返回值优化，此函数故意这样写
A Get(bool flag)
{
    A a;
    A b;
    cout << "ready return" << endl;
    if (flag)
        return a;
    else
        return b;
}
int main()
{
    {
        A a = Get(false); // 运行报错
    }
    cout << "main finish" << endl;
    return 0;
}
/*
constructor A 
constructor A 
ready return 
destructor A, m_ptr:0xf87af8 
destructor A, m_ptr:0xf87ae8 
destructor A, m_ptr:0xf87af8 
main finish 
*/
```

在上面的代码中，默认构造函数是浅拷贝，main函数的 a 和Get函数的 b 会指向同一个指针 m\_ptr，在 析构的时候会导致重复删除该指针。

{% hint style="info" %}
C++中类的拷贝有两种:深拷贝,浅拷贝:当出现类的等号低值时,即会调用拷贝函数&#x20;

两个的区别:在未定义显示拷贝构造函数的情况下,系统会调用默认的拷贝函数--即浅拷贝,它能够完成成员的一一复制。

* 当数据成员中没有指针时,浅拷贝是可行的;但当数据成员中有指针时,如果采用简单的浅拷贝,则两类中的两个指针将指向同一个地址,当对象快 结束时,会调用两次析构函数,而导致指针悬挂现象,所以,此时,必须采用深拷贝。&#x20;
* 深拷贝与浅拷贝的区别就在于深拷贝会在堆内存中另外申请空间来储存数据,从而也就解决了指针悬挂的问题。简而言之,当数据 成员中有指针时,必须要用深拷贝。
{% endhint %}

正确的做法是提供深拷贝的拷贝构造函数，比如下面的代码（关闭 返回值优化的情况下）：

```cpp
#include <iostream>
using namespace std;
class A
{
    public:
    A() :m_ptr(new int(0)) {
        cout << "constructor A" << endl;
    }
    A(const A& a) :m_ptr(new int(*a.m_ptr)) {
        cout << "copy constructor A" << endl;
    }
    ~A(){
        cout << "destructor A, m_ptr:" << m_ptr << endl;
        delete m_ptr;
        m_ptr = nullptr;
    }
    private:
    int* m_ptr;
};
// 为了避免返回值优化，此函数故意这样写
A Get(bool flag)
{
    A a;
    A b;
    cout << "ready return" << endl;
    if (flag)
        return a;
    else
        return b;
}
int main()
{
    {
        A a = Get(false); // 正确运行
    }
    cout << "main finish" << endl;
    return 0;
}
/*
constructor A
constructor A
ready return
copy constructor A
destructor A, m_ptr:0xea7af8
destructor A, m_ptr:0xea7ae8
destructor A, m_ptr:0xea7b08
main finish
*/
```

这样就可以保证拷贝构造时的安全性，但有时这种拷贝构造却是不必要的，比如上面代码中的拷贝构造 就是不必要的。<mark style="color:red;">上面代码中的 Get 函数会返回临时变量，然后通过这个临时变量拷贝构造了一个新的对象 b，临时变量在拷贝构造完成之后就销毁了，如果堆内存很大，那么，这个拷贝构造的代价会很大， 带来了额外的性能损耗。</mark>有没有办法避免临时对象的拷贝构造呢？答案是肯定的。看下面的代码：

```cpp
#include <iostream>
using namespace std;
class A
{
    public:
    A() :m_ptr(new int(0)) {
        cout << "constructor A" << endl;
    }
    A(const A& a) :m_ptr(new int(*a.m_ptr)) {
        cout << "copy constructor A" << endl;
    }
    A(A&& a) :m_ptr(a.m_ptr) {
        a.m_ptr = nullptr;
        cout << "move constructor A" << endl;
    }
    ~A(){
        cout << "destructor A, m_ptr:" << m_ptr << endl;
        if(m_ptr)
            delete m_ptr;
    }
    private:
    int* m_ptr;
};
// 为了避免返回值优化，此函数故意这样写
A Get(bool flag)
{
    A a;
    A b;
    cout << "ready return" << endl;
    if (flag)
        return a;
    else
        return b;
}
int main()
{
    {
        A a = Get(false); // 正确运行
    }
    cout << "main finish" << endl;
    return 0;
}

/*
constructor A
constructor A
ready return
move constructor A
destructor A, m_ptr:0
destructor A, m_ptr:0xfa7ae8
destructor A, m_ptr:0xfa7af8
main finish
*/
```

上面的代码中没有了拷贝构造，取而代之的是移动构造（ `Move Construct`）。<mark style="color:red;">从移动构造函数的实现 中可以看到，它的参数是一个右值引用类型的参数</mark> <mark style="color:red;"></mark><mark style="color:red;">`A&&`</mark><mark style="color:red;">，这里没有深拷贝，只有浅拷贝，这样就避免了对临时对象的深拷贝，提高了性能。这里的</mark> <mark style="color:red;"></mark><mark style="color:red;">`A&&`</mark> <mark style="color:red;"></mark><mark style="color:red;">用来根据参数是左值还是右值来建立分支，如果是临时 值，则会选择移动构造函数。移动构造函数只是将临时对象的资源做了浅拷贝，不需要对其进行深拷贝，从而避免了额外的拷贝，提高性能。</mark>这也就是所谓的移动语义（ move 语义），右值引用的一个重要目的是用来支持移动语义的。

**移动语义可以将资源（堆、系统对象等）通过浅拷贝方式从一个对象转移到另一个对象，这样能够减少不必要的临时对象的创建、拷贝以及销毁，可以大幅度提高 C++ 应用程序的性能，消除临时对象的维护 （创建和销毁）对性能的影响。**

以一个简单的 string 类为示例，实现拷贝构造函数和拷贝赋值操作符。

```cpp
#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <string.h>
using namespace std;
class MyString {
    private:
    char* m_data;
    size_t m_len;
    void copy_data(const char *s) {
        m_data = new char[m_len+1];
        memcpy(m_data, s, m_len);
        m_data[m_len] = '\0';
    }
    public:
    MyString() {
        m_data = NULL;
        m_len = 0;
    }
    MyString(const char* p) {
        m_len = strlen (p);
        copy_data(p);
    }
    MyString(const MyString& str) {
        m_len = str.m_len;
        copy_data(str.m_data);
        std::cout << "Copy Constructor is called! source: " << str.m_data <<
            std::endl;
    }
    MyString& operator=(const MyString& str) {
        if (this != &str) {
            m_len = str.m_len;
            copy_data(str.m_data);
        }
        std::cout << "Copy Assignment is called! source: " << str.m_data <<
            std::endl;
        return *this;
    }
    virtual ~MyString() {
        if (m_data) free(m_data);
    }
};
void test() {
    MyString a;
    a = MyString("Hello");
    std::vector<MyString> vec;
    vec.push_back(MyString("World"));
}
int main()
{
    test();
    return 0;
}
```

实现了调用拷贝构造函数的操作和拷贝赋值操作符的操作。<mark style="color:red;">`MyString(“Hello”)`</mark> <mark style="color:red;"></mark><mark style="color:red;">和</mark> <mark style="color:red;"></mark><mark style="color:red;">`MyString(“World”)`</mark> <mark style="color:red;"></mark><mark style="color:red;">都是临时对象，也就是右值。虽然它们是临时的，但程序仍然调用了拷贝构造和拷贝赋值，造成了没有意 义的资源申请和释放的操作。如果能够直接使用临时对象已经申请的资源，既能节省资源，有能节省资 源申请和释放的时间。这正是定义转移语义的目的。</mark>

用c++11的右值引用来定义这两个函数

```cpp
// 用c++11的右值引用来定义这两个函数
MyString(MyString&& str) {
    std::cout << "Move Constructor is called! source: " << str.m_data <<
        std::endl;
    m_len = str.m_len;
    m_data = str.m_data; //避免了不必要的拷贝
    str.m_len = 0;
    str.m_data = NULL;
}
MyString& operator=(MyString&& str) {
    std::cout << "Move Assignment is called! source: " << str.m_data <<
        std::endl;
    if (this != &str) {
        m_len = str.m_len;
        m_data = str.m_data; //避免了不必要的拷贝
        str.m_len = 0;
        str.m_data = NULL;
    }
    return *this;
}
```

<mark style="color:red;">有了右值引用和转移语义，我们在设计和实现类时，对于需要动态申请大量资源的类，应该设计右值引用的拷贝构造函数和赋值函数，以提高应用程序的效率。</mark>

### 移动(move )语义

我们知道移动语义是通过右值引用来匹配临时值的，<mark style="color:red;">那么，普通的左值是否也能借助移动语义来优化性 能呢？C++11为了解决这个问题，提供了std::move()方法来将左值转换为右值，从而方便应用移动语义。move是将对象的状态或者所有权从一个对象转移到另一个对象，只是转义，没有内存拷贝。</mark>

![](https://pic.superbed.cc/item/65a2401b792284bad676c2ed.jpg)

```cpp
#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <string.h>
using namespace std;
class MyString {
    private:
    char* m_data;
    size_t m_len;
    void copy_data(const char *s) {
        m_data = new char[m_len+1];
        memcpy(m_data, s, m_len);
        m_data[m_len] = '\0';
    }
    public:
    MyString() {
        m_data = NULL;
        m_len = 0;
    }
    MyString(const char* p) {
        m_len = strlen (p);
        copy_data(p);
    }
    MyString(const MyString& str) {
        m_len = str.m_len;
        copy_data(str.m_data);
        std::cout << "Copy Constructor is called! source: " << str.m_data <<
            std::endl;
    }
    MyString& operator=(const MyString& str) {
        if (this != &str) {
            m_len = str.m_len;
            copy_data(str.m_data);
        }
        std::cout << "Copy Assignment is called! source: " << str.m_data <<
            std::endl;
        return *this;
    }
    // 用c++11的右值引用来定义这两个函数
    MyString(MyString&& str) {
        std::cout << "Move Constructor is called! source: " << str.m_data <<
            std::endl;
        m_len = str.m_len;
        m_data = str.m_data; //避免了不必要的拷贝
        str.m_len = 0;
        str.m_data = NULL;
    }
    MyString& operator=(MyString&& str) {
        std::cout << "Move Assignment is called! source: " << str.m_data <<
            std::endl;
        if (this != &str) {
            m_len = str.m_len;
            m_data = str.m_data; //避免了不必要的拷贝
            str.m_len = 0;
            str.m_data = NULL;
        }
        return *this;
    }
    virtual ~MyString() {
        if (m_data) free(m_data);
    }
};
int main()
{
    MyString a;
    a = MyString("Hello"); // move
    MyString b = a; // copy
    MyString c = std::move(a); // move， 将左值转为右值
    return 0;
}

```

### forward 完美转发

<mark style="color:red;">右值引用类型是独立于值的，一个右值引用作为函数参数的形参时，在函数内部转发该参数给内部其他函数时，它就变成一个左值，并不是原来的类型了。</mark>如果需要按照参数原来的类型转发到另一个函数，可以使用C++11提供的std::forward()函数，该函数实现的功能称之为完美转发。

```cpp
// 函数原型
template <class T> T&& forward (typename remove_reference<T>::type& t) noexcept;
template <class T> T&& forward (typename remove_reference<T>::type&& t) noexcept;

// 精简之后的样子
std::forward<T>(t);
```

* <mark style="color:red;">`当T为左值引用类型时，t将被转换为T类型的左值`</mark>
* <mark style="color:red;">`当T不是左值引用类型时，t将被转换为T类型的右值`</mark>

下面通过一个例子演示一下关于forward的使用:

```c
#include <iostream>
using namespace std;

template<typename T>
void printValue(T& t)
{
    cout << "l-value: " << t << endl;
}

template<typename T>
void printValue(T&& t)
{
    cout << "r-value: " << t << endl;
}

template<typename T>
void testForward(T && v)
{
    printValue(v);
    printValue(move(v));
    printValue(forward<T>(v));
    cout << endl;
}

int main()
{
    testForward(520);
    int num = 1314;
    testForward(num);
    testForward(forward<int>(num));
    testForward(forward<int&>(num));
    testForward(forward<int&&>(num));

    return 0;
}
```

测试代码打印的结果如下:

```c
l-value: 520
r-value: 520
r-value: 520

l-value: 1314
r-value: 1314
l-value: 1314

l-value: 1314
r-value: 1314
r-value: 1314

l-value: 1314
r-value: 1314
l-value: 1314

l-value: 1314
r-value: 1314
r-value: 1314
```

* `testForward(520);`函数的形参为未定引用类型`T&&`，实参为右值，初始化后被推导为一个右值引用
  * `printValue(v);`已命名的右值v，编译器会视为左值处理，实参为`左值`
  * `printValue(move(v));`已命名的右值编译器会视为左值处理，通过move又将其转换为右值，实参为`右值`
  * `printValue(forward<T>(v));`forward的模板参数为右值引用，最终得到一个右值，实参为\`\`右值\`
* `testForward(num);`函数的形参为未定引用类型`T&&`，实参为左值，初始化后被推导为一个左值引用
  * `printValue(v);`实参为`左值`
  * `printValue(move(v));`通过move将左值转换为右值，实参为`右值`
  * `printValue(forward<T>(v));`forward的模板参数为左值引用，最终得到一个左值引用，实参为`左值`
* `testForward(forward<int>(num));`forward的模板类型为int，最终会得到一个右值，函数的形参为未定引用类型`T&&`被右值初始化后得到一个右值引用类型
  * `printValue(v);`已命名的右值v，编译器会视为左值处理，实参为`左值`
  * `printValue(move(v));`已命名的右值编译器会视为左值处理，通过move又将其转换为右值，实参为`右值`
  * `printValue(forward<T>(v));`forward的模板参数为右值引用，最终得到一个右值，实参为`右值`
* `testForward(forward<int&>(num));`forward的模板类型为int&，最终会得到一个左值，函数的形参为未定引用类型`T&&`被左值初始化后得到一个左值引用类型
  * `printValue(v);`实参为`左值`
  * `printValue(move(v));`通过move将左值转换为右值，实参为`右值`
  * `printValue(forward<T>(v));`forward的模板参数为左值引用，最终得到一个左值，实参为`左值`
* `testForward(forward<int&&>(num));`forward的模板类型为int&&，最终会得到一个右值，函数的形参为未定引用类型`T&&`被右值初始化后得到一个右值引用类型
  * `printValue(v);`已命名的右值v，编译器会视为左值处理，实参为`左值`
  * `printValue(move(v));`已命名的右值编译器会视为左值处理，通过move又将其转换为右值，实参为`右值`
  * `printValue(forward<T>(v));`forward的模板参数为右值引用，最终得到一个右值，实参为`右值`

**综合示例**

```cpp
#include "stdio.h"
#include <iostream>
#include<vector>
using namespace std;
class A
{
    public:
    A() :m_ptr(NULL), m_nSize(0){}
    A(int *ptr, int nSize)
    {
        m_nSize = nSize;
        m_ptr = new int[nSize];
        if (m_ptr)
        {
            memcpy(m_ptr, ptr, sizeof(sizeof(int) * nSize));
        }
    }
    A(const A& other) // 拷贝构造函数实现深拷贝
    {
        m_nSize = other.m_nSize;
        if (other.m_ptr)
        {
            delete[] m_ptr;
            m_ptr = new int[m_nSize];
            memcpy(m_ptr, other.m_ptr, sizeof(sizeof(int)* m_nSize));
        }
        else
        {
            m_ptr = NULL;
        }
        cout << "A(const int &i)" << endl;
    }
    // 右值应用构造函数
    A(A &&other)
    {
        m_ptr = NULL;
        m_nSize = other.m_nSize;
        if (other.m_ptr)
        {
            m_ptr = move(other.m_ptr); // 移动语义
            other.m_ptr = NULL;
        }
    }
    ~A()
    {
        if (m_ptr)
        {
            delete[] m_ptr;
            m_ptr = NULL;
        }
    }
    void deleteptr()
    {
        if (m_ptr)
        {
            delete[] m_ptr;
            m_ptr = NULL;
        }
    }
    int *m_ptr;
    int m_nSize;
};
void main()
{
    int arr[] = { 1, 2, 3 };
    A a(arr, sizeof(arr)/sizeof(arr[0]));
    cout << "m_ptr in a Addr: 0x" << a.m_ptr << endl;
    A b(a);
    cout << "m_ptr in b Addr: 0x" << b.m_ptr << endl;
    b.deleteptr();
    A c(std::forward<A>(a)); // 完美转换
    cout << "m_ptr in c Addr: 0x" << c.m_ptr << endl;
    c.deleteptr();
    vector<int> vect{ 1, 2, 3, 4, 5 };
    cout << "before move vect size: " << vect.size() << endl;
    vector<int> vect1 = move(vect);
    cout << "after move vect size: " << vect.size() << endl;
    cout << "new vect1 size: " << vect1.size() << endl;
}

```

### `emplace_back` 减少内存拷贝和移动

对于STL容器，C++11后引入了emplace\_back接口。 emplace\_back是就地构造，不用构造后再次复制到容器中。因此效率更高。 考虑这样的语句：

```cpp
vector<string> testVec;
testVec.push_back(string(16, 'a'));
```

上述语句足够简单易懂，将一个string对象添加到testVec中。底层实现：

* 首先，string(16, ‘a’)会创建一个string类型的临时对象，这涉及到一次string构造过程。
* 其次，vector内会创建一个新的string对象，这是第二次构造。
* 最后在push\_back结束时，最开始的临时对象会被析构。加在一起，这两行代码会涉及到两次 string构造和一次析构。

`c++11`可以用`emplace_back`代替`push_back`，`emplace_back`可以直接在`vector`中构建一个对象，而非创建一个临时对象，再放进`vector`，再销毁。`emplace_back`可以省略一次构建和一次析构，从而达到优化的目的.

```cpp
//time_interval.h
#ifndef TIME_INTERVAL_H
#define TIME_INTERVAL_H
#include <iostream>
#include <memory>
#include <string>
#ifdef GCC
#include <sys/time.h>
#else
#include <ctime>
#endif // GCC
class TimeInterval
{
    public:
    TimeInterval(const std::string& d) : detail(d)
    {
        init();
    }
    TimeInterval()
    {
        init();
    }
    ~TimeInterval()
    {
        #ifdef GCC
        gettimeofday(&end, NULL);
        std::cout << detail
            << 1000 * (end.tv_sec - start.tv_sec) + (end.tv_usec -
                                                     start.tv_usec) / 1000
            << " ms" << endl;
        #else
        end = clock();
        std::cout << detail
            << (double)(end - start) << " ms" << std::endl;
        #endif // GCC
    }
    protected:
    void init() {
        #ifdef GCC
        gettimeofday(&start, NULL);
        #else
        start = clock();
        #endif // GCC
    }
    private:
    std::string detail;
    #ifdef GCC
    timeval start, end;
    #else
    clock_t start, end;
    #endif // GCC
};
#define TIME_INTERVAL_SCOPE(d) std::shared_ptr<TimeInterval> time_interval_scope_begin = std::make_shared<TimeInterval>(d)
#endif // TIME_INTERVAL_H
```

{% hint style="info" %}
一个十分有用的计时类
{% endhint %}

```cpp
#include <vector>
#include <string>
#include "time_interval.h"
int main() {
    std::vector<std::string> v;
    int count = 10000000;
    v.reserve(count); //预分配十万大小，排除掉分配内存的时间
    {
        TIME_INTERVAL_SCOPE("push_back string:");
        for (int i = 0; i < count; i++)
        {
            std::string temp("ceshi");
            v.push_back(temp);// push_back(const string&)，参数是左值引用
        }
    }
    v.clear();
    {
        TIME_INTERVAL_SCOPE("push_back move(string):");
        for (int i = 0; i < count; i++)
        {
            std::string temp("ceshi");
            v.push_back(std::move(temp));// push_back(string &&), 参数是右值引用
        }
    }
    v.clear();
    {
        TIME_INTERVAL_SCOPE("push_back(string):");
        for (int i = 0; i < count; i++)
        {
            v.push_back(std::string("ceshi"));// push_back(string &&), 参数是右值引
            用
        }
    }
    v.clear();
    {
        TIME_INTERVAL_SCOPE("push_back(c string):");
        for (int i = 0; i < count; i++)
        {
            v.push_back("ceshi");// push_back(string &&), 参数是右值引用
        }
    }
    v.clear();
    {
        TIME_INTERVAL_SCOPE("emplace_back(c string):");
        for (int i = 0; i < count; i++)
        {
            v.emplace_back("ceshi");// 只有一次构造函数，不调用拷贝构造函数，速度最快
        }
    }
}
/*
push_back string:335 ms
push_back move(string):307 ms
push_back(string):285 ms
push_back(c string):295 ms
emplace_back(c string):234 ms
*/
```

第1中方法耗时最长，原因显而易见，将调用左值引用的`push_back`，且将会调用一次string的拷贝构造 函数，比较耗时，这里的string还算很短的，如果很长的话，差异会更大

第2、3、4中方法耗时基本一样，参数为右值，将调用右值引用的push\_back，故调用string的移动构造 函数，移动构造函数耗时比拷贝构造函数少，因为不需要重新分配内存空间。

第5中方法耗时最少，因为`emplace_back`只调用构造函数，没有移动构造函数，也没有拷贝构造函数。 为了证实上述论断，我们自定义一个类，并在普通构造函数、拷贝构造函数、移动构造函数中打印相应 描述：

```cpp
#include <vector>
#include <string>
#include "time_interval.h"
using namespace std;
class Foo {
    public:
    Foo(std::string str) : name(str) {
        std::cout << "constructor" << std::endl;
    }
    Foo(const Foo& f) : name(f.name) {
        std::cout << "copy constructor" << std::endl;
    }
    Foo(Foo&& f) : name(std::move(f.name)){
        std::cout << "move constructor" << std::endl;
    }
    private:
    std::string name;
};
int main() {
    std::vector<Foo> v;
    int count = 10000000;
    v.reserve(count); //预分配十万大小，排除掉分配内存的时间
    {
        TIME_INTERVAL_SCOPE("push_back T:");
        Foo temp("test");
        v.push_back(temp);// push_back(const T&)，参数是左值引用
        //打印结果：
        //constructor
        //copy constructor
    }
    cout << " ---------------------\n" << endl;
    v.clear();
    {
        TIME_INTERVAL_SCOPE("push_back move(T):");
        Foo temp("test");
        v.push_back(std::move(temp));// push_back(T &&), 参数是右值引用
        //打印结果：
        //constructor
        //move constructor
    }
    cout << " ---------------------\n" << endl;
    v.clear();
    {
        TIME_INTERVAL_SCOPE("push_back(T&&):");
        v.push_back(Foo("test"));// push_back(T &&), 参数是右值引用
        //打印结果：
        //constructor
        //move constructor
    }
    cout << " ---------------------\n" << endl;
    v.clear();
    {
        std::string temp = "test";
        TIME_INTERVAL_SCOPE("push_back(string):");
        v.push_back(temp);// push_back(T &&), 参数是右值引用
        //打印结果：
        //constructor
        //move constructor
    }
    cout << " ---------------------\n" << endl;
    v.clear();
    {
        std::string temp = "test";
        TIME_INTERVAL_SCOPE("emplace_back(string):");
        v.emplace_back(temp);// 只有一次构造函数，不调用拷贝构造函数，速度最快
        //打印结果：
        //constructor
    }
}
/*
constructor
copy constructor
push_back T:2 ms
constructor
move constructor
push_back move(T):0 ms
constructor
move constructor
push_back(T&&):0 ms
constructor
move constructor
push_back(string):0 ms
constructor
emplace_back(string):0 ms
*/
```

### 小结

C++11 在性能上做了很大的改进，最大程度减少了内存移动和复制，通过右值引用、 forward、 emplace 和一些无序容器我们可以大幅度改进程序性能。

* 右值引用仅仅是通过改变资源的所有者来避免内存的拷贝，能大幅度提高性能。
* forward 能根据参数的实际类型转发给正确的函数。
* emplace 系列函数通过直接构造对象的方式避免了内存的拷贝和移动。

### reference

* [https://blog.csdn.net/u010700335/article/details/39830425](https://blog.csdn.net/u010700335/article/details/39830425)
* [https://subingwen.cn/cpp/move-forward/](https://subingwen.cn/cpp/move-forward/)
* [https://subingwen.cn/cpp/rvalue-reference/](https://subingwen.cn/cpp/rvalue-reference/)
