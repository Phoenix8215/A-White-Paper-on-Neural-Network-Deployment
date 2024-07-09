# 🌴 智能指针|C++11

## 智能指针

C++程序设计中使用堆内存是非常频繁的操作，堆内存的申请和释放都由程序员自己管理。程序员自己 管理堆内存可以提高了程序的效率，但是整体来说堆内存的管理是麻烦的，C++11中引入了智能指针的 概念，方便管理堆内存。使用普通指针，容易造成堆内存泄露（忘记释放），二次释放，程序发生异常 时内存泄露等问题等，使用智能指针能更好的管理堆内存。

C++里面的四个智能指针: ~~`auto_ptr`~~,`unique_ptr`,`shared_ptr`, `weak_ptr` 其中后三个是C++11支持， 并且第一个已经被C++11弃用。

<mark style="color:red;">使用这些智能指针时需要引用头文件</mark><mark style="color:red;">`<memory>`</mark><mark style="color:red;">。</mark>

### `shared_ptr`共享的智能指针

共享的智能指针 `std::shared_ptr`使用引用计数，每一个`shared_ptr`的拷贝都指向相同的内存。再最后一个`shared_ptr`析 构的时候，内存才会被释放。 `shared_ptr`共享被管理对象，同一时刻可以有多个`shared_ptr`拥有对象的所有权，当最后一个 `shared_ptr`对象销毁时，被管理对象自动销毁。 简单来说，`shared_ptr`实现包含了两部分，

* <mark style="color:red;">一个指向堆上创建的对象的裸指针:</mark><mark style="color:red;">`raw_ptr`</mark>
* <mark style="color:red;">一个指向内部隐藏的、共享的管理对象:</mark><mark style="color:red;">`share_count_object`</mark>

第一部分没什么好说的，第二部分是需要关注的重点. `use_count`：当前这个堆上对象被多少对象引用了，简单来说就是引用计数。

#### `shared_ptr`的基本用法

1. **初始化**

通过构造函数、`std::shared_ptr`辅助函数和`reset`方法来初始化`shared_ptr`，代码如下：

```cpp
// 智能指针初始化
std::shared_ptr<int> p1(new int(1));
std::shared_ptr<int> p2 = p1;
std::shared_ptr<int> p3;
p3.reset(new int(1));
if(p3) {
	cout << "p3 is not null";
}
```

**我们应该优先使用make\_shared来构造智能指针，因为他更高效。**

```cpp
auto sp1 = make_shared<int>(100);
//相当于
shared_ptr<int> sp1(new int(100));
```

{% hint style="info" %}
当使用 `std::shared_ptr` 来管理动态分配的对象时，通常会涉及两个主要的动态内存分配：

1. **对象的内存分配**：这是为了分配存储对象本身的内存。在C++中，这通常是通过 `new` 表达式完成的，它分配了足够的内存来存储特定类型的对象，并调用该类型的构造函数来初始化这块内存。
2. **控制块的内存分配**：`std::shared_ptr` 需要一个额外的控制块来存储与对象管理相关的信息，如引用计数和可能的自定义删除器。这个控制块也需要动态分配内存。

#### 使用 `new` 和 `std::shared_ptr` 构造函数

当你手动使用 `new` 表达式创建一个对象，并将其传递给 `std::shared_ptr` 的构造函数时，首先是对象的内存分配，然后是控制块的内存分配：

```cpp
std::shared_ptr<MyClass> myObject(new MyClass(42, 3.14));
```

在这种情况下，`new MyClass(42, 3.14)` 首先分配内存并构造 `MyClass` 对象，然后 `std::shared_ptr` 构造函数为其控制块分配另一块内存。

#### 使用 `std::make_shared`

相比之下，`std::make_shared` 只进行一次动态内存分配：

```cpp
std::shared_ptr<MyClass> myObject = std::make_shared<MyClass>(42, 3.14);
```

在这个例子中，`std::make_shared` 同时为 `MyClass` 对象和控制块分配了一块连续的内存。这种方法更高效，因为它减少了内存分配的次数，并且由于对象和控制块在内存中是相邻的，还可以提高缓存的使用效率。
{% endhint %}

不能将一个原始指针直接赋值给一个智能指针，例如，下面这种方法是错误的：

```c++
std::shared_ptr<int> p = new int(1);
```

`shared_ptr`不能通过“直接将原始这种赋值”来初始化，需要通过构造函数和辅助方法来初始化。 对于一个未初始化的智能指针，可以通过`reset`方法来初始化，<mark style="color:red;">当智能指针有值的时候调用</mark><mark style="color:red;">`reset`</mark><mark style="color:red;">会引起引用计数减1。</mark>另外智能指针通过重载的`bool`类型操作符来判断是否为空。

```cpp
#include <iostream>
#include <memory>
using namespace std;

void test(shared_ptr<int> sp) {
    // sp在test里面的作用域
    cout << "test sp.use_count()" <<sp.use_count() << endl;
}
int main()
{
    auto sp1 = make_shared<int>(100); // 优先使用make_shared来构造智能指针
    //相当于
    shared_ptr<int> sp2(new int(100));

//    std::shared_ptr<int> p = new int(1);  // 不能将一个原始指针直接赋值给一个智能指针

    std::shared_ptr<int> p1;
    p1.reset(new int(1));  // 有参数就是分配资源
    std::shared_ptr<int> p2 = p1;

    // 引用计数此时应该是2
    cout << "p2.use_count() = " << p2.use_count()<< endl;  // 2
    cout << "p1.use_count() = " << p1.use_count()<< endl;  // 2
    p1.reset();   // 没有参数就是释放资源
    cout << "p1.reset()\n";
    // 引用计数此时应该是1
    cout << "p2.use_count()= " << p2.use_count() << endl;  // 1
    cout << "p1.use_count() = " << p1.use_count()<< endl;  // 0
    if(!p1) {  // p1是空的
        cout << "p1 is empty\n";
    }
    if(p2) { // p2非空
        cout << "p2 is not empty\n";
    }
    p2.reset();
    cout << "p2.reset()\n";
    cout << "p2.use_count()= " << p2.use_count() << endl;
    if(!p2) {
        cout << "p2 is empty\n";
    }

    shared_ptr<int> sp5(new int(100));
    test(sp5);
    cout << "sp5.use_count()" << sp5.use_count() << endl;
    return 0;
}

/*
p2.use_count() = 2
p1.use_count() = 2
p1.reset()
p2.use_count()= 1
p1.use_count() = 0
p1 is empty
p2 is not empty
p2.reset()
p2.use_count()= 0
p2 is empty
test sp.use_count()2
sp5.use_count()1
*/

```

2. **获取原始指针**

当需要获取原始指针时，可以通过`get`方法来返回原始指针，代码如下所示：

```cpp
std::shared_ptr<int> ptr(new int(1));
int *p = ptr.get(); //万一不小心 delete p;
```

**谨慎使用`p.get()`的返回值，如果你不知道其危险性则永远不要调用get()函数。**

`p.get()`的返回值就相当于一个裸指针的值，不合适的使用这个值，上述陷阱的所有错误都有可能发生， 遵守以下几个约定：&#x20;

* <mark style="color:red;">不要保存</mark><mark style="color:red;">`p.get()`</mark><mark style="color:red;">的返回值 ，无论是保存为裸指针还是</mark><mark style="color:red;">`shared_ptr`</mark><mark style="color:red;">都是错误的。保存为裸指针不知什么时候就会变成空悬指针，保存为</mark><mark style="color:red;">`shared_ptr`</mark><mark style="color:red;">则产生了独立指针</mark>
* <mark style="color:red;">不要</mark><mark style="color:red;">`delete`</mark> <mark style="color:red;">`p.get()`</mark><mark style="color:red;">的返回值 ，会导致对一块内存</mark><mark style="color:red;">`delete`</mark><mark style="color:red;">两次的错误</mark>

3. **指定删除器**

如果用`shared_ptr`管理非`new`对象或是没有析构函数的类时，应当为其传递合适的删除器。

```cpp
#include <iostream>
#include <memory>
using namespace std;

void DeleteIntPtr(int *p) {
    cout << "call DeleteIntPtr" << endl;
    delete p;
}

int main()
{
    std::shared_ptr<int> p(new int(1), DeleteIntPtr);
    std::shared_ptr<int> p2(new int(1), [](int *p) {
        cout << "call lambda1 delete p" << endl;
        delete p;});
    std::shared_ptr<int> p3(new int[10], [](int *p) {
        cout << "call lambda2 delete p" << endl;
        delete [] p; // 数组删除
    });
    return 0;
}

/*
call lambda2 delete p
call lambda1 delete p
call DeleteIntPtr
*/

```

当p的引用计数为0时，自动调用删除器`DeleteIntPtr`来释放对象的内存。删除器可以是一个`lambda`表达式，上面的写法可以改为：

```cpp
std::shared_ptr<int> p(new int(1), [](int *p) {
	cout << "call lambda delete p" << endl;
	delete p;});
```

<mark style="color:red;">当我们用</mark><mark style="color:red;">`shared_ptr`</mark><mark style="color:red;">管理动态数组时，需要指定删除器，因为</mark><mark style="color:red;">`shared_ptr`</mark><mark style="color:red;">的默认删除器不支持数组对象(</mark><mark style="color:red;">**后来**</mark><mark style="color:red;">**`C++17`**</mark><mark style="color:red;">**支持了**</mark><mark style="color:red;">)</mark>，代码如下所示：

```cpp
std::shared_ptr<int> p3(new int[10], [](int *p) { delete [] p;});
std::shared_ptr<int[]>p3(new int[10]);// C++17
```

{% hint style="info" %}
在删除数组内存时，除了自己编写删除器，也可以使用C++提供的`std::default_delete<T>()`函数作为删除器，这个函数内部的删除功能也是通过调用`delete`来实现的，要释放什么类型的内存就将模板类型T指定为什么类型即可。具体处理代码如下：

```cpp
int main()
{
    shared_ptr<int> ptr(new int[10], default_delete<int[]>());
    return 0;
}
```

另外，我们还可以自己封装一个make\_shared\_array方法来让shared\_ptr支持数组，代码如下:

```cpp
#include <iostream>
#include <memory>
using namespace std;

template <typename T>
shared_ptr<T> make_share_array(size_t size)
{
    // 返回匿名对象
    return shared_ptr<T>(new T[size], default_delete<T[]>());
}

int main()
{
    shared_ptr<int> ptr1 = make_share_array<int>(10);
    cout << ptr1.use_count() << endl;
    shared_ptr<char> ptr2 = make_share_array<char>(128);
    cout << ptr2.use_count() << endl;
    return 0;
}
```

😅但是`std::make_shared` 并不支持创建数组类型的 `shared_ptr`。如果你需要创建一个数组并想利用 `make_shared` 的性能优势（如减少内存分配次数），你可能需要考虑其他解决方案，例如使用 `std::vector` 或其他容器类，或者自行管理内存。
{% endhint %}

<mark style="color:red;">从 C++20 开始，</mark><mark style="color:red;">`std::make_shared`</mark> <mark style="color:red;"></mark><mark style="color:red;">支持创建数组，可以直接使用</mark> <mark style="color:red;"></mark><mark style="color:red;">`std::make_shared`</mark> <mark style="color:red;"></mark><mark style="color:red;">来创建并管理数组。</mark>

**示例：使用 `std::make_shared` 创建数组（C++20 及之后）**

```c
#include <iostream>
#include <memory>

int main() {
    // 使用 std::make_shared 创建一个共享指针，管理一个数组
    auto array = std::make_shared<int[]>(10);

    // 初始化数组元素
    for (int i = 0; i < 10; ++i) {
        array[i] = i;
    }

    // 输出数组元素
    for (int i = 0; i < 10; ++i) {
        std::cout << array[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}

```

在这个示例中，`std::make_shared<int[]>(10)` 创建了一个 `std::shared_ptr<int[]>`，管理一个大小为 10 的数组。

* 智能指针什么时候需要指定删除器

<mark style="color:red;">在需要 delete 以外的析构行为的时候用. 因为</mark> <mark style="color:red;"></mark><mark style="color:red;">`shared_ptr`</mark> <mark style="color:red;"></mark><mark style="color:red;">在引用计数为 0 后默认调用 delete ptr; 如果不满足需求就要提供定制的删除器.</mark>

<mark style="color:red;">一些场景:</mark>

* 资源不是 `new` 出来的(一般也意味着不能 `delete`), 比如可能是 `malloc` 出来的
* 资源是被第三方库管理的 (第三方提供 资源获取 和 资源释放 接口, 那么要么写一个 `wrapper` 类要么就提供定制的 deleter)
* 资源不是 `RAII` 的, 意味着析构函数不会把资源完全释放掉，也就是单纯 `delete`还不够。

#### 使用`shared_ptr`要注意的问题

1. <mark style="color:red;">不要用一个原始指针初始化多个</mark><mark style="color:red;">`shared_ptr`</mark><mark style="color:red;">，例如下面错误范例：</mark>

```cpp
int *ptr = new int;
shared_ptr<int> p1(ptr);
shared_ptr<int> p2(ptr); // 逻辑错误
```

2. <mark style="color:red;">不要在函数实参中创建</mark><mark style="color:red;">`shared_ptr`</mark><mark style="color:red;">，对于下面的写法：</mark>

```cpp
function(shared_ptr<int>(new int), g()); //有缺陷
```

因为C++的函数参数的计算顺序在不同的编译器不同的约定下可能是不一样的，一般是从右到左，但也 可能从左到右，所以，可能的过程是先`new int`，然后调用`g()`，如果恰好`g()`发生异常，而`shared_ptr`还 没有创建， 则`int`内存泄漏了，正确的写法应该是先创建智能指针，代码如下：

```c
shared_ptr<int> p(new int);
function(p, g());
```

> * 形参和实参的区别和联系:
>   * 形参变量只有在函数被调用时才会分配内存，调用结束后，立刻释放内存，所以形参变量只有在函数内部有效，不能在函数外部使用。
>   * 实参可以是常量、变量、表达式、函数等，无论实参是何种类型的数据，在进行函数调用时，它们都必须有确定的值，以便把这些值传送给形参，所以应该提前用赋值、输入等办法使实参获得确定值。
>   * 实参和形参在数量上、类型上、顺序上必须严格一致，否则会发生“类型不匹配”的错误。当然，如果能够进行自动类型转换，或者进行了强制类型转换，那么实参类型也可以不同于形参类型。
>   * 函数调用中发生的数据传递是单向的，只能把实参的值传递给形参，而不能把形参的值反向地传递给实参；换句话说，一旦完成数据的传递，实参和形参就再也没有瓜葛了，所以，在函数调用过程中，形参的值发生改变并不会影响实参。

请看下面的例子：

```c
#include <stdio.h>

//计算从m加到n的值
int sum(int m, int n) {
    int i;
    for (i = m+1; i <= n; ++i) {
        m += i;
    }
    return m;
}

int main() {
    int a, b, total;
    printf("Input two numbers: ");
    scanf("%d %d", &a, &b);
    total = sum(a, b);
    printf("a=%d, b=%d\n", a, b);
    printf("total=%d\n", total);

    return 0;
}
/*
运行结果：
Input two numbers: 1 100
a=1, b=100
total=5050
*/
```

在这段代码中，函数定义处的 m、n 是形参，函数调用处的 a、b 是实参。通过 scanf() 可以读取用户输入的数据，并赋值给 a、b，在调用 sum() 函数时，这份数据会传递给形参 m、n。

从运行情况看，输入 a 值为 1，即实参 a 的值为 1，把这个值传递给函数 sum() 后，形参 m 的初始值也为 1，在函数执行过程中，形参 m 的值变为 5050。函数运行结束后，输出实参 a 的值仍为 1，可见实参的值不会随形参的变化而变化。

3. 通过`shared_from_this()`返回this指针。不要将`this`指针作为`shared_ptr`返回出来，因为this指针 本质上是一个裸指针，因此，这样可能会导致重复析构，看下面的例子。

```cpp
#include <iostream>
#include <memory>

using namespace std;

class A
{
public:
    shared_ptr<A>GetSelf()
    {
        return shared_ptr<A>(this); // 不要这么做
    }
    ~A()
    {
        cout << "Deconstruction A" << endl;
    }
};

int main()
{
    shared_ptr<A> sp1(new A);
    shared_ptr<A> sp2 = sp1->GetSelf();
    cout << "sp1.use_count() = " << sp1.use_count()<< endl;
    cout << "sp2.use_count() = " << sp2.use_count()<< endl;
    return 0;
}

/*
sp1.use_count() = 1
sp2.use_count() = 1
Deconstruction A
Deconstruction A
*/

```

<mark style="color:red;">在这个例子中，由于用同一个指针（this)构造了两个智能指针</mark><mark style="color:red;">`sp1`</mark><mark style="color:red;">和</mark><mark style="color:red;">`sp2`</mark><mark style="color:red;">，而他们之间是没有任何关系 的，在离开作用域之后this将会被构造的两个智能指针各自析构，导致重复析构的错误。</mark>

正确返回`this`的`shared_ptr`的做法是：让目标类通过`std::enable_shared_from_this`类，然后使用基类的 成员函数`shared_from_this()`来返回`this`的`shared_ptr`，如下所示。

```cpp
#include <iostream>
#include <memory>

using namespace std;

class A: public std::enable_shared_from_this<A>
{
public:
    shared_ptr<A>GetSelf()
    {
        return shared_from_this(); //
    }
    ~A()
    {
        cout << "Deconstruction A" << endl;
    }
};

int main()
{
    // auto spp = make_shared<A>();
    shared_ptr<A> sp1(new A);
    shared_ptr<A> sp2 = sp1->GetSelf();  // ok
    //    shared_ptr<A> sp2;
    //    {
    //        shared_ptr<A> sp1(new A);
    //        sp2 = sp1->GetSelf();  // ok
    //    }
    cout << "sp1.use_count() = " << sp1.use_count()<< endl;
    cout << "sp2.use_count() = " << sp2.use_count()<< endl;

    return 0;
}
/*
sp1.use_count() = 2
sp2.use_count() = 2
Deconstruction A
*/
```

4. 避免循环引用。循环引用会导致内存泄漏，比如：

```cpp
#include <iostream>
#include <memory>
using namespace std;


class A;
class B;

class A {
public:
    std::shared_ptr<B> bptr;
    ~A() {
        cout << "A is deleted" << endl;
    }
};

class B {
public:
    std::shared_ptr<A> aptr;
    ~B() {
        cout << "B is deleted" << endl;  // 析构函数后，才去释放成员变量
    }
};


int main()
{
    std::shared_ptr<A> pa;

    {
        std::shared_ptr<A> ap(new A);
        std::shared_ptr<B> bp(new B);
        ap->bptr = bp;
        bp->aptr = ap;
        pa = ap;
        // 是否可以手动释放呢？
//        ap.reset();
        ap->bptr.reset(); // 手动释放成员变量才行
    }
    cout<< "main leave. pa.use_count()" << pa.use_count() << endl;  // 循环引用导致ap bp退出了作用域都没有析构
    return 0;
}



/*
B is deleted
main leave. pa.use_count()1
A is deleted
*/
```

循环引用导致`ap`和`bp`的引用计数为2，在离开作用域之后，`ap`和`bp`的引用计数减为1，并不回减为0，导 致两个指针都不会被析构，产生内存泄漏。<mark style="color:red;">解决的办法是把A和B任何一个成员变量改为</mark><mark style="color:red;">`weak_ptr`</mark><mark style="color:red;">。</mark>

> * 什么是循环引用？
>
> 循环引用（circular reference）是指在编程中，两个或多个对象之间形成一个循环的引用关系，导致这些对象之间的内存无法被正确释放，从而引发内存泄漏。这种情况也被称为循环依赖或循环关联。

### `unique_ptr`独占的智能指针

`unique_ptr`是一个独占型的智能指针，它不允许其他的智能指针共享其内部的指针，不允许通过赋值将 一个`unique_ptr`赋值给另一个`unique_ptr`。下面的错误示例。

```cpp
unique_ptr<T> my_ptr(new T);
unique_ptr<T> my_other_ptr = my_ptr; // 报错，不能复制
```

<mark style="color:red;">`unique_ptr`</mark><mark style="color:red;">不允许复制，但可以通过函数返回给其他的</mark><mark style="color:red;">`unique_ptr`</mark><mark style="color:red;">，还可以通过</mark><mark style="color:red;">`std::move`</mark><mark style="color:red;">来转移到其 他的</mark><mark style="color:red;">`unique_ptr`</mark><mark style="color:red;">，这样它本身就不再拥有原来指针的所有权了。</mark>例如

```cpp
unique_ptr<T> my_ptr(new T); // 正确
unique_ptr<T> my_other_ptr = std::move(my_ptr); // 正确
unique_ptr<T> ptr = my_ptr; // 报错，不能复制
```

`std::make_shared`是`c++11`的一部分，<mark style="color:red;">但</mark><mark style="color:red;">`std::make_unique`</mark><mark style="color:red;">不是，它是在</mark><mark style="color:red;">`c++14`</mark><mark style="color:red;">里加入标准库的。</mark>

```cpp
auto upw1(std::make_unique<Widget>()); // with make func
std::unique_ptr<Widget> upw2(new Widget); // without make func
```

使用new的版本重复了被创建对象的键入，但是make\_unique函数则没有。重复类型违背了软件工程的 一个重要原则：应该避免代码重复，代码中的重复会引起编译次数增加，导致目标代码膨胀。

除了unique\_ptr的独占性， unique\_ptr和shared\_ptr还有一些区别，比如

* <mark style="color:red;">unique\_ptr可以指向一个数组，代码如下所示</mark>

<pre class="language-cpp"><code class="lang-cpp">std::unique_ptr&#x3C;int []> ptr(new int[10]);
ptr[9] = 9;
<strong>std::shared_ptr&#x3C;int []> ptr2(new int[10]); // 这个是不合法的
</strong></code></pre>

{% hint style="info" %}
<mark style="color:red;">`std::shared_ptr<int []> ptr2(new int[10])`</mark><mark style="color:red;">C++17中支持了此类写法</mark>
{% endhint %}

* unique\_ptr指定删除器和shared\_ptr有区别

```cpp
std::shared_ptr<int> ptr3(new int(1), [](int *p){delete p;}); // 正确
std::unique_ptr<int> ptr4(new int(1), [](int *p){delete p;}); // 错误
```

unique\_ptr需要确定删除器的类型，所以不能像shared\_ptr那样直接指定删除器，可以这样写：

```cpp
std::unique_ptr<int, void(*)(int*)> ptr5(new int(1), [](int *p){delete p;}); //正确
```

{% hint style="info" %}
```cpp
int main()
{
    using func_ptr = void(*)(int*);
    unique_ptr<int, func_ptr> ptr1(new int(10), [](int*p) {delete p; });

    return 0;
}
```

在上面的代码中第7行，`func_ptr`的类型和`lambda表达式`的类型是一致的。在lambda表达式没有捕获任何变量的情况下是正确的，如果捕获了变量，编译时则会报错：

```c
int main()
{
    using func_ptr = void(*)(int*);
    unique_ptr<int, func_ptr> ptr1(new int(10), [&](int*p) {delete p; });	// error
    return 0;
}
```

<mark style="color:red;">上面的代码中错误原因是这样的，在lambda表达式没有捕获任何外部变量时，可以直接转换为函数指针，一旦捕获了就无法转换了(仿函数)，如果想要让编译器成功通过编译，那么需要使用可调用对象包装器来处理声明的函数指针：</mark>

```c
int main()
{
    using func_ptr = void(*)(int*);
    unique_ptr<int, function<void(int*)>> ptr1(new int(10), [&](int*p) {delete p; });
    return 0;
}
```
{% endhint %}

<mark style="color:red;">关于shared\_ptr和unique\_ptr的使用场景是要根据实际应用需求来选择。如果希望只有一个智能指针管理资源或者管理数组就用unique\_ptr，如果希望多个智能指针管理同一个资源就用shared\_ptr。</mark>

```cpp
#include <iostream>
#include <memory>
using namespace std;

struct MyDelete
{
    void operator()(int *p)
    {
        cout << "delete" << endl;
        delete p;
    }
};

int main()
{
    auto upw1(std::make_unique<int[]>(2));     // make_unique需要设置c++14
    std::unique_ptr<int> upw2(new int); // without make func

    upw1[21] = 1;
    std::unique_ptr<int,MyDelete> ptr2(new int(1));
//    auto ptr(std::make_unique<int, MyDelete>(1));
    cout << "main finish! " << upw1[1] << endl;
    return 0;
}

/*
main finish! 0
delete
*/
```

### `weak_ptr`弱引用的智能指针

share\_ptr虽然已经很好用了，但是有一点share\_ptr智能指针还是有内存泄露的情况，当两个对象相互 使用一个shared\_ptr成员变量指向对方，会造成循环引用，使引用计数失效，从而导致内存泄漏。

weak\_ptr <mark style="color:red;">是一种不控制对象生命周期的智能指针</mark>, 它指向一个 shared\_ptr 管理的对象. 进行该对象的内 存管理的是那个强引用的shared\_ptr， weak\_ptr只是提供了对管理对象的一个访问手段。weak\_ptr 设 计的目的是为配合 shared\_ptr 而引入的一种智能指针来协助 shared\_ptr 工作, 它只可以从一个 shared\_ptr 或另一个 weak\_ptr 对象构造, 它的构造和析构不会引起引用记数的增加或减少。weak\_ptr 是用来解决shared\_ptr相互引用时的死锁问题，如果说两个shared\_ptr相互引用，那么这两个指针的引 用计数永远不可能下降为0，资源永远不会释放。<mark style="color:red;">它是对对象的一种弱引用，不会增加对象的引用计数， 和shared\_ptr之间可以相互转化，shared\_ptr可以直接赋值给它，它可以通过调用lock函数来获得 shared\_ptr。</mark>

<mark style="color:red;">weak\_ptr没有重载操作符</mark><mark style="color:red;">`*`</mark><mark style="color:red;">和</mark><mark style="color:red;">`->`</mark><mark style="color:red;">，因为它不共享指针，不能操作资源，主要是为了通过shared\_ptr获得 资源的监测权，它的构造不会增加引用计数，它的析构也不会减少引用计数，纯粹只是作为一个旁观者 来监视shared\_ptr中管理的资源是否存在。weak\_ptr还可以返回this指针和解决循环引用的问题。</mark>

#### weak\_ptr的基本用法

1. 通过use\_count()方法获取当前观察资源的引用计数，如下所示：

```cpp
shared_ptr<int> sp(new int(10));
weak_ptr<int> wp(sp);
cout << wp.use_count() << endl; //结果讲输出1
```

2. <mark style="color:red;">通过expired()方法判断所观察资源是否已经释放，如下所示：</mark>

```cpp
shared_ptr<int> sp(new int(10));
weak_ptr<int> wp(sp);
if(wp.expired())
    cout << "weak_ptr无效,资源已释放";
else
    cout << "weak_ptr有效";
```

3. 通过lock方法获取监视的shared\_ptr，如下所示：

```cpp
std::weak_ptr<int> gw;
void f()
{
    if(gw.expired()) {
        cout << "gw无效,资源已释放";
    }
    else {
        auto spt = gw.lock();
        cout << "gw有效, *spt = " << *spt << endl;
    }
}

int main()
{
    {
        auto sp = atd::make_shared<int>(42);
        gw = sp;
        f();
    }
    f();
    return 0;
}
```

#### weak\_ptr返回this指针

shared\_ptr章节中提到不能直接将this指针返回shared\_ptr，需要通过派生 std::enable\_shared\_from\_this类，并通过其方法`shared_from_this`来返回指针，原因是 std::enable\_shared\_from\_this类中有一个weak\_ptr，这个weak\_ptr用来观察this智能指针，调用 shared\_from\_this()方法是，会调用内部这个weak\_ptr的lock()方法，将所观察的shared\_ptr返回，再看前面的范例

```cpp
#include <iostream>
#include <memory>

using namespace std;

class A: public std::enable_shared_from_this<A>
{
public:
    shared_ptr<A>GetSelf()
    {
        return shared_from_this(); 
    }
    ~A()
    {
        cout << "Deconstruction A" << endl;
    }
};

int main()
{
    // auto spp = make_shared<A>();
    shared_ptr<A> sp1(new A);
    shared_ptr<A> sp2 = sp1->GetSelf();  // ok
    //    shared_ptr<A> sp2;
    //    {
    //        shared_ptr<A> sp1(new A);
    //        sp2 = sp1->GetSelf();  // ok
    //    }
    cout << "sp1.use_count() = " << sp1.use_count()<< endl;
    cout << "sp2.use_count() = " << sp2.use_count()<< endl;

    return 0;
}
/*
sp1.use_count() = 2
sp2.use_count() = 2
Deconstruction A
*/
```

在外面创建A对象的智能指针和通过对象返回this的智能指针都是安全的，因为`shared_from_this()`是内部的`weak_ptr`调用`lock()`方法之后返回的智能指针，在离开作用域之后，`sp1`的引用计数减为0，A对象会被析构，不会出现A对象被析构两次的问题。

<mark style="color:red;">需要注意的是，获取自身智能指针的函数尽量在</mark><mark style="color:red;">`shared_ptr`</mark><mark style="color:red;">的构造函数被调用之后才能使用，因为</mark> <mark style="color:red;"></mark><mark style="color:red;">`enable_shared_from_this`</mark><mark style="color:red;">内部的</mark><mark style="color:red;">`weak_ptr`</mark><mark style="color:red;">只有通过</mark><mark style="color:red;">`shared_ptr`</mark><mark style="color:red;">才能构造。</mark>

#### weak\_ptr解决循环引用问题

在shared\_ptr章节提到智能指针循环引用的问题，因为智能指针的循环引用会导致内存泄漏，可以通过 weak\_ptr解决该问题，只要将A或B的任意一个成员变量改为weak\_ptr

```cpp
#include <iostream>
#include <memory>
using namespace std;


class A;
class B;

class A {
public:
    std::weak_ptr<B> bptr; // 修改为weak_ptr
    int *val;
    A() {
        val = new int(1);
    }
    ~A() {
        cout << "A is deleted" << endl;
        delete  val;
    }
};

class B {
public:
    std::shared_ptr<A> aptr;
    ~B() {
        cout << "B is deleted" << endl;
    }
};

//weak_ptr 是一种不控制对象生命周期的智能指针,
void test()
{
    std::shared_ptr<A> ap(new A);
    std::weak_ptr<A> wp1 = ap;
    std::weak_ptr<A> wp2 = ap;
    cout<< "ap.use_count()" << ap.use_count()<< endl;
}

void test2()
{
    std::weak_ptr<A> wp;
    {
        std::shared_ptr<A> ap(new A);
        wp = ap;
    }
    cout<< "wp.use_count()" << wp.use_count() << ", wp.expired():" << wp.expired() << endl;
    if(!wp.expired()) {
        // wp不能直接操作对象的成员、方法
        std::shared_ptr<A> ptr = wp.lock(); // 需要先lock获取std::shared_ptr<A>
        *(ptr->val) = 20;  
    }
}


int main()
{
    test2();
//    {
//        std::shared_ptr<A> ap(new A);
//        std::shared_ptr<B> bp(new B);
//        ap->bptr = bp;
//        bp->aptr = ap;
//    }
    cout<< "main leave" << endl;
    return 0;
}
/*
A is deleted
wp.use_count()0, wp.expired():1
main leave
*/
```

这样在对B的成员赋值时，即执行bp->aptr=ap;时，由于aptr是weak\_ptr，它并不会增加引用计数，所以ap的引用计数仍然会是1，在离开作用域之后，ap的引用计数为减为0，A指针会被析构，析构后其内部的bptr的引用计数会被减为1，然后在离开作用域后bp引用计数又从1减为0，B对象也被析构，不会发生内存泄漏。

#### weak\_ptr使用注意事项

1. weak\_ptr在使用前需要检查合法性。

<pre class="language-cpp"><code class="lang-cpp">weak_ptr&#x3C;int> wp;
<strong>{
</strong>    shared_ptr&#x3C;int> sp(new int(1)); //sp.use_count()==1
    wp = sp; //wp不会改变引用计数，所以sp.use_count()=1
    shared_ptr&#x3C;int> sp_ok = wp.lock(); //wp没有重载->操作符,只能这样取所指向的对象(sp.use_count()=2)
}
shared_ptr&#x3C;int> sp_null = wp.lock(); //sp_null .use_count()==0;
</code></pre>

因为上述代码中sp和sp\_ok离开了作用域，其容纳的K对象已经被释放了。 得到了一个容纳NULL指针的sp\_null对象。<mark style="color:red;">在使用wp前需要调用wp.expired()函数判断一下</mark>。 因为wp还仍旧存在，虽然引用计数等于0，仍有某处“全局”性的存储块保存着这个计数信息。直到最后 一个weak\_ptr对象被析构，这块“堆”存储块才能被回收。如果`shared_ptr` sp\_ok和`weak_ptr` wp;属于同一个作用域呢？如下所示：

```cpp
weak_ptr<int> wp;
shared_ptr<int> sp_ok;
{
    shared_ptr<int> sp(new int(1)); //sp.use_count()==1
    wp = sp; //wp不会改变引用计数，所以sp.use_count()==1
    sp_ok = wp.lock(); //wp没有重载->操作符,只能这样取所指向的对象(sp.use_count()=2)
}
if(wp.expired()) {
    cout << "shared_ptr is destroy" << endl;
} else {
    cout << "shared_ptr no destroy" << endl;
}

/*
shared_ptr no destroy
*/
```

## reference

* [https://subingwen.cn/cpp/shared\_ptr/](https://subingwen.cn/cpp/shared\_ptr/)
* [https://subingwen.cn/cpp/unique\_ptr/](https://subingwen.cn/cpp/unique\_ptr/)
