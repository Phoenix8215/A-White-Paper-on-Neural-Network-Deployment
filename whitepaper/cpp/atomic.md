# 🥒 原子变量|CAS|memory order|异步操作

### 原子变量

C++11提供了一个原子类型`std::atomic<T>`，通过这个原子类型管理的内部变量就可以称之为原子变量，我们可以给原子类型指定`bool、char、int、long、指针`等类型作为模板参数（`不支持浮点类型和复合类型`）。

原子指的是一系列不可被CPU上下文交换的机器指令，这些指令组合在一起就形成了原子操作。在多核CPU下，当某个CPU核心开始运行原子操作时，会先暂停其它CPU内核对内存的操作，以保证原子操作不会被其它CPU内核所干扰。

<mark style="color:red;">由于原子操作是通过指令提供的支持，因此它的性能相比锁和消息传递会好很多。相比较于锁而言，原子类型不需要开发者处理加锁和释放锁的问题，同时支持修改，读取等操作，还具备较高的并发性能，几乎所有的语言都支持原子类型。</mark>

可以看出原子类型是无锁类型，但是无锁不代表无需等待，因为原子类型内部使用了`CAS`循环，当大量的冲突发生时，该等待还是得等待！但是总归比锁要好。

C++11内置了整形的原子变量，这样就可以更方便的使用原子变量了。在多线程操作中，使用原子变量之后就不需要再使用互斥量来保护该变量了，用起来更简洁。因为对原子变量进行的操作只能是一个原子操作（`atomic operation`），<mark style="color:red;">原子操作指的是不会被线程调度机制打断的操作(</mark><mark style="color:red;">**read-modify-write**</mark><mark style="color:red;">)，这种操作一旦开始，就一直运行到结束，中间不会有任何的上下文切换。</mark>多线程同时访问共享资源造成数据混乱的原因就是因为CPU的上下文切换导致的，使用原子变量解决了这个问题，因此互斥锁的使用也就不再需要了。

#### atomic 类成员 <a href="#id-1atomic-lei-cheng-yuan" id="id-1atomic-lei-cheng-yuan"></a>

<mark style="color:red;">不会存储到缓存中，运算完毕后直接给内存了atomic变量无法保护复合类型的数据(类或者结构体)，对于指针类型的atomic变量来说，只有指针</mark><mark style="color:red;">`++/--`</mark><mark style="color:red;">才是原子操作，指针所指的对象(类或者结构体)并不能保证其原子性</mark>\


<figure><img src="../../.gitbook/assets/1 X-af1R7djTjaQsjB131dkw.png" alt=""><figcaption></figcaption></figure>

**类定义**

```c
// 定义于头文件 <atomic>
template< class T >
struct atomic;
```

通过定义可得知：<mark style="color:red;">在使用这个模板类的时候，一定要指定模板类型。</mark>

#### **构造函数**

```c
// 1
atomic() noexcept = default;
// 2
constexpr atomic( T desired ) noexcept;
// 3
atomic( const atomic& ) = delete;
```

* 构造函数1：默认无参构造函数。
* 构造函数2：使用 `desired` 初始化原子变量的值。
* 构造函数3：使用`=delete`显示删除拷贝构造函数, 不允许进行对象之间的拷贝

```c
/**
 * @brief 初始化atomic变量
 */
void test()
{
    atomic<char> c{'a'};
    atomic<int> b;
    atomic_init(&b, 9);
}
```

#### **公共成员函数**

原子类型在类内部重载了`=`操作符，并且<mark style="color:red;">不允许在类的外部使用</mark> <mark style="color:red;"></mark><mark style="color:red;">`=`</mark><mark style="color:red;">进行对象的拷贝。</mark>

```c
T operator=( T desired ) noexcept;
T operator=( T desired ) volatile noexcept;

atomic& operator=( const atomic& ) = delete;
atomic& operator=( const atomic& ) volatile = delete;
```

原子地以 `desired` 替换当前值。按照 `order` 的值影响内存。

```c
void store( T desired, std::memory_order order = std::memory_order_seq_cst ) noexcept;
void store( T desired, std::memory_order order = std::memory_order_seq_cst ) volatile noexcept;
```

* **desired**：存储到原子变量中的值
* **order**：强制的内存顺序

原子地加载并返回原子变量的当前值。按照 `order` 的值影响内存。直接访问原子对象也可以得到原子变量的当前值。

```c
T load( std::memory_order order = std::memory_order_seq_cst ) const noexcept;
T load( std::memory_order order = std::memory_order_seq_cst ) const volatile noexcept;
```

```c
/**
 * @brief 修改atomic变量的值
 */
void test01()
{
    atomic_char cc('b');
    cc = 'd';
    cout << cc << endl;
    cc.store('a');
    cout << cc << endl;

    char ccc = cc.exchange('e');//返回之前的旧值
    cout << cc.load() << endl;
    cout << ccc << endl;
}

```

```c
class Counter {
   public:
    void increment() {
        for (int i = 0; i < 100; ++i) {
            // mtx.lock();
            number++;
            cout << "+++ increment thread id: " << this_thread::get_id()
                 << ", number: " << number << endl;
            // mtx.unlock();
            this_thread::sleep_for(chrono::milliseconds(100));
        }
    }

    void increment1() {
        for (int i = 0; i < 100; ++i) {
            // mtx.lock();
            number++;
            cout << "*** decrement thread id: " << this_thread::get_id()
                 << ", number: " << number << endl;
            // mtx.unlock();
            this_thread::sleep_for(chrono::milliseconds(50));
        }
    }

   private:
    // int number = 0;
    atomic_int number{0};
    // atomic<Base*> base;
    // mutex mtx;
};
```

#### 特化成员函数

* 复合赋值运算符重载，主要包含以下形式：

| 模板类型T为整形 | <p>T operator+= (T val) volatile noexcept;<br>T operator+= (T val) noexcept;<br>T operator-= (T val) volatile noexcept;<br>T operator-= (T val) noexcept;<br>T operator&#x26;= (T val) volatile noexcept;<br>T operator&#x26;= (T val) noexcept;<br>T operator|= (T val) volatile noexcept;<br>T operator|= (T val) noexcept;<br>T operator^= (T val) volatile noexcept;<br>T operator^= (T val) noexcept;</p> |
| :------: | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 模板类型T为指针 | <p>T operator+= (ptrdiff_t val) volatile noexcept;<br>T operator+= (ptrdiff_t val) noexcept;<br>T operator-= (ptrdiff_t val) volatile noexcept;<br>T operator-= (ptrdiff_t val) noexcept;</p>                                                                                                                                                                                                                  |

* 以上各个 operator 都会有对应的 **fetch\_\*** 操作，详细见下表：

<table><thead><tr><th width="100" align="center">操作符</th><th width="184" align="center">操作符重载函数</th><th width="168" align="center">等级的成员函数</th><th width="84" align="center">整形</th><th width="95" align="center">指针</th><th align="center">其他</th></tr></thead><tbody><tr><td align="center">+</td><td align="center">atomic::operator+=</td><td align="center">atomic::fetch_add</td><td align="center">是</td><td align="center">是</td><td align="center">否</td></tr><tr><td align="center">-</td><td align="center">atomic::operator-=</td><td align="center">atomic::fetch_sub</td><td align="center">是</td><td align="center">是</td><td align="center">否</td></tr><tr><td align="center">&#x26;</td><td align="center">atomic::operator&#x26;=</td><td align="center">atomic::fetch_and</td><td align="center">是</td><td align="center">否</td><td align="center">否</td></tr><tr><td align="center">|</td><td align="center">atomic::operator|=</td><td align="center">atomic::fetch_or</td><td align="center">是</td><td align="center">否</td><td align="center">否</td></tr><tr><td align="center">^</td><td align="center">atomic::operator^=</td><td align="center">atomic::fetch_xor</td><td align="center">是</td><td align="center">否</td><td align="center">否</td></tr></tbody></table>

### CAS(compare and swap)

没有交换操作，CAS 就不完整。交换操作可通过 exchange() 成员函数实现。

```c
std::atomic<int> x(0);
int y = 42;
int x0 = x.exchange(y); // (x0 = x; x = y;) done atomically
```

调用 exchange() 会将原子的当前值与所需值互换，并返回被替换的值。所有操作均以原子方式完成。

CAS 在swap前增加了一个附加条件，可通过 compare\_exchange\_strong() 和 compare\_exchange\_weak()这两个函数使用。

```c
std::atomic<int> x(0);
int expected = x.load();
int desired = 42;

// If x == expected, x = desired and return true
// Otherwise, expected = x and return false
while(!x.compare_exchange_strong(expected, desired));
```

如上代码所示，该方法将一个期望值与原子变量进行比较。如果它们确实相等，原子变量将被交换，并返回 true。

否则，原子变量的当前值将被加载到expected变量，并返回 false。这种机制允许我们创建一个重试循环，而无需再次显式地读取原子值。

CAS 会失败的原因是与其他线程的争用。在我们对原子进行初始读取以获得预期值后，我们必须确保其他线程在我们上次读取原子后没有对其进行更改。假设多个线程都在尝试 CAS，那么重试循环就会一直运行，直到我们的 CAS 比其他线程更快地完成更改。

下面举例说明如何在无锁列表中使用 CAS。

```c
// List node
struct Node
{
    int mValue{0};
    Node* mNext{nullptr};
};

std::atomic<Node*> pHead(nullptr);

// Push a node to the front of the list
void PushFront(const int& value)
{
    Node* newNode = new Node();
    newNode->mValue = value;
    Node* expected = pHead.load();
   
    // Try push the node we have just created to the front of the list
    do
    {
        newNode->mNext = pHead;
    }
    while(!pHead.compare_exchange_strong(expected, newNode));
}
```

#### **Strong vs Weak** <a href="#a3f4" id="a3f4"></a>

std::atomic 为 CAS 提供了强和弱两种选项。

compare\_exchange\_strong() 只有在原子的当前值由于争用而不再是我们所期望的值时，才会失败。

另一方面，compare\_exchange\_weak()允许虚假失败。这意味着，即使预期值等于当前值，它仍可能返回 false。

但为什么我们希望即使当前值等于预期值，CAS 也会失败呢？这是因为在某些平台上，获取独占访问可能代价非常大，但读取原子值却往往代价很小。

因此，弱 CAS 不会要求独占访问，而是执行一定时间的尝试来获取锁（硬件），这可能会超时--类似于网络套接字。

<figure><img src="../../.gitbook/assets/1 ZLd19SxR4tdPYK4fs-KeeQ.png" alt=""><figcaption></figcaption></figure>









### 异步操作

* std::future
* std::aysnc
* std::promise
* std::packaged\_task

参考C++官方手册的范例。

### std::future

std::future期待一个返回，从一个异步调用的角度来说，future更像是执行函数的返回值，C++标准库 使用std::future为一次性事件建模，如果一个事件需要等待特定的一次性事件，那么这线程可以获取一 个future对象来代表这个事件。 异步调用往往不知道何时返回，但是如果异步调用的过程需要同步，或者说后一个异步调用需要使用前 一个异步调用的结果。这个时候就要用到future。 线程可以周期性的在这个future上等待一小段时间，检查future是否已经ready，如果没有，该线程可以 先去做另一个任务，一旦future就绪，该future就无法复位（无法再次使用这个future等待这个事件），所以future代表的是一次性事件。

**std::future的类型**

在库的头文件中声明了两种future，唯一future（std::future）和共享future（std::shared\_future）这 两个是参照std::unique\_ptr和std::shared\_ptr设立的，前者的实例是仅有的一个指向其关联事件的实 例，而后者可以有多个实例指向同一个关联事件，当事件就绪时，所有指向同一事件的 std::shared\_future实例会变成就绪。

**std::future的使用**

std::future是一个模板，例如std::future，模板参数就是期待返回的类型，虽然std::future被用于线程间通 信，但其本身却并不提供同步访问，必须通过互斥量或其他同步机制来保护访问。 std::future使用的时机是当你不需要立刻得到一个结果的时候，你可以开启一个线程帮你去做一项任务，并期待这个任务的返回，但是std::thread并没有提供这样的机制，这就需要用到std::async和std::future （都在头文件中声明） std::async返回一个std::future对象，而不是给你一个确定的值（所以当你不需要立刻使用此值的时候才 需要用到这个机制）。<mark style="color:red;">当你需要使用这个值的时候，对std::future使用get()，线程就会阻塞直到std::future就 绪，然后返回该值。</mark>

**类的定义**

通过类的定义可以得知，`future`是一个模板类，也就是这个类可以存储任意指定类型的数据。

```c
// 定义于头文件 <future>
template< class T > class future;
template< class T > class future<T&>;
template<>          class future<void>;
```

**构造函数**

```c
// 1
future() noexcept;
// 2
future( future&& other ) noexcept;
// 3
future( const future& other ) = delete;
```

* 构造函数 1：默认无参构造函数
* 构造函数 2：移动构造函数，转移资源的所有权
* 构造函数 3：使用`=delete`显示删除拷贝构造函数, 不允许进行对象之间的拷贝

**常用成员函数（public)**

一般情况下使用`=`进行赋值操作就进行对象的拷贝，但是`future`对象不可用复制，因此会根据实际情况进行处理：

* 如果`other`是右值，那么转移资源的所有权
* 如果`other`是非右值，不允许进行对象之间的拷贝（<mark style="color:red;">该函数被显示删除禁止使用</mark>）

```
future& operator=( future&& other ) noexcept;
future& operator=( const future& other ) = delete;
```

取出`future`对象内部保存的数据，其中`void get()`是为`future<void>`准备的，此时对象内部类型就是`void`，<mark style="color:red;">该函数是一个阻塞函数，当子线程的数据就绪后解除阻塞就能得到传出的数值了。</mark>

```cpp
T get();
T& get();
void get();
```

<mark style="color:red;">因为</mark><mark style="color:red;">`future`</mark><mark style="color:red;">对象内部存储的是异步线程任务执行完毕后的结果，是在调用之后的将来得到的，因此可以通过调用</mark><mark style="color:red;">`wait()`</mark><mark style="color:red;">方法，阻塞当前线程，等待这个子线程的任务执行完毕，任务执行完毕当前线程的阻塞也就解除了。</mark>

```cpp
void wait() const;
```

如果当前线程`wait()`方法就会死等，直到子线程任务执行完毕将返回值写入到`future`对象中，调用`wait_for()`只会让线程阻塞一定的时长，但是这样并不能保证对应的那个子线程中的任务已经执行完毕了。

`wait_until()`和`wait_for()`函数功能是差不多，前者是阻塞到某一指定的时间点，后者是阻塞一定的时长。

```cpp
template< class Rep, class Period >
std::future_status wait_for( const std::chrono::duration<Rep,Period>& timeout_duration ) const;

template< class Clock, class Duration >
std::future_status wait_until( const std::chrono::time_point<Clock,Duration>& timeout_time ) const;
```

当`wait_until()`和`wait_for()`函数返回之后，并不能确定子线程当前的状态，因此我们需要判断函数的返回值，这样就能知道子线程当前的状态了：

| 常量                                                                                   | 解释                     |
| ------------------------------------------------------------------------------------ | ---------------------- |
| [`future_status::deferred`](https://zh.cppreference.com/w/cpp/thread/future\_status) | 子线程中的任务函仍未启动           |
| [`future_status::ready`](https://zh.cppreference.com/w/cpp/thread/future\_status)    | 子线程中的任务已经执行完毕，结果已就绪    |
| [`future_status::timeout`](https://zh.cppreference.com/w/cpp/thread/future\_status)  | 子线程中的任务正在执行中，指定等待时长已用完 |

### std::promise

<mark style="color:red;">`std::promise`</mark><mark style="color:red;">是一个协助线程赋值的类，它能够将数据和</mark><mark style="color:red;">`future`</mark><mark style="color:red;">对象绑定起来，为获取线程函数中的某个值提供便利。</mark>

#### 类成员函数 <a href="#id-21-lei-cheng-yuan-han-shu" id="id-21-lei-cheng-yuan-han-shu"></a>

**类定义**

通过`std::promise`类的定义可以得知，这也是一个模板类，我们要在线程中传递什么类型的数据，模板参数就指定为什么类型。

```c
// 定义于头文件 <future>
template< class R > class promise;
template< class R > class promise<R&>;
template<>          class promise<void>;
```

**构造函数**

```c
// 1
promise();
// 2
promise( promise&& other ) noexcept;
// 3
promise( const promise& other ) = delete;
```

* 构造函数1：默认构造函数，得到一个空对象
* 构造函数2：移动构造函数
* 构造函数3：使用`=delete`显示删除拷贝构造函数, 不允许进行对象之间的拷贝

#### **公共成员函数**

在`std::promise`类内部管理着一个`future`类对象，调用`get_future()`就可以得到这个`future`对象了

```c
std::future<T> get_future();
```

存储要传出的 `value` 值，并立即让状态就绪，这样数据被传出其它线程就可以得到这个数据了。重载的第四个函数是为`promise<void>`类型的对象准备的。

```cpp
void set_value( const R& value );
void set_value( R&& value );
void set_value( R& value );
void set_value();
```

存储要传出的 `value` 值，但是不立即令状态就绪。在当前线程退出时，子线程资源被销毁，再令状态就绪。

```c
void set_value_at_thread_exit( const R& value );
void set_value_at_thread_exit( R&& value );
void set_value_at_thread_exit( R& value );
void set_value_at_thread_exit();
```

> 1. `set_value(value_type v)`：此方法用于立即设置 `std::promise` 持有的值。一旦调用了 `set_value`，任何与之关联的 `std::future` 或 `std::shared_future` 对象都将能够获取这个值。如果 `set_value` 被多次调用，或者与 `set_exception` 一起调用，将抛出异常。
> 2. `set_value_at_thread_exit(value_type v)`：此方法与 `set_value` 类似，但它用于设置值，但设置操作会推迟到调用线程退出时才执行。这意味着，如果你在创建 `std::promise` 的线程中调用 `set_value_at_thread_exit`，并且在此线程的 `std::promise` 对象被销毁之前没有退出，那么值将不会被设置。这在某些情况下很有用，特别是当生成值的计算可能涉及线程本身的本地资源，而这些资源只有在线程退出时才可用。

#### promise的使用 <a href="#id-22promise-de-shi-yong" id="id-22promise-de-shi-yong"></a>

<mark style="color:red;">通过</mark><mark style="color:red;">`promise`</mark><mark style="color:red;">传递数据的过程一共分为5步</mark>：

1. 在主线程中创建`std::promise`对象
2. 将这个`std::promise`对象通过引用的方式传递给子线程的任务函数
3. 在子线程任务函数中给`std::promise`对象赋值
4. 在主线程中通过`std::promise`对象取出绑定的`future`实例对象
5. 通过得到的`future`对象取出子线程任务函数中返回的值。

```c
#include <iostream>
#include <thread>
#include <future>
using namespace std;

void func(promise<string>& p)
{
    this_thread::sleep_for(chrono::milliseconds(3));
    p.set_value("我是路飞，要成为海贼王。。。");
    this_thread::sleep_for(chrono::milliseconds(1));
}

/**
 * @brief 异步操作和同步等待的结合
 */
int main()
{
    promise<string> pro;
    // thread t1(func, ref(pro));
    thread t2([](promise<string>& p) {
        this_thread::sleep_for(chrono::seconds(3));
        p.set_value("我是路飞，要成为海贼王。。。");
        this_thread::sleep_for(chrono::seconds(1));
    },ref(pro));
    future<string> f = pro.get_future();
    cout << "主线程" << endl;
    string str = f.get();
    cout << "主线程阻塞结束" << endl;

    cout << "子线程返回的数据: " << str << endl;
    t2.join();
    return 0;
}

```

```c
#include <iostream>
#include <thread>
#include <future>
#include <chrono>

using namespace std;


int main()
{
    promise<int> pr;
    thread t1([](promise<int> &p){
        this_thread::sleep_for(chrono::milliseconds(10));
        p.set_value(10);
        cout << "子线程结束了" <<endl;
    }, std::ref(pr));

    // future<int> ft = pr.get_future();
    auto ft = pr.get_future();
    
    //循环直到future的状态是就绪的，主线程可以乘机做其他的事情
    while(ft.wait_for(chrono::seconds(0)) != future_status::ready )
    {
        cout << "在异步等待中, 主线程在做其他的事情" << endl;
        // this_thread::sleep_for(chrono::seconds(3));  
    }

    //future已经就绪了，可以获得取值
    int val = ft.get();
    cout << "val = " << val <<endl;
    t1.join();
    return 0;

}
```

### std::packaged\_task

如果说std::async和std::feature还是分开看的关系的话，那么std::packaged\_task就是将任务和feature 绑定在一起的模板，是一种封装，对任务的封装。 `The class template std::packaged_task wraps any Callable target (function, lambda expression, bind expression, or another function object) so that it can be invoked asynchronously. Its return value or exception thrown is stored in a shared state which can be accessed through std::future objects.`

可以通过std::packaged\_task对象获取任务相关联的feature，调用get\_future()方法可以获得 std::packaged\_task对象绑定的函数的返回值类型的future。std::packaged\_task的模板参数是函数签 名。 PS：例如int add(int a, intb)的函数签名就是int(int, int)

```cpp
#include <future>
#include <iostream>
#include <thread>
#include <functional>
using namespace std;


string myFunc() {
    this_thread::sleep_for(chrono::seconds(3));
    return "我是路飞，要成为海贼王。。。";
}

using funcPtr = string (*)(string, int);

class Base {
   public:
    string operator()(string msg) {
        string str = "operator() function msg: " + msg;
        return str;
    }

    operator funcPtr() {
        return showMsg;
    }

    int getNumber(int num) {
        int number = num + 100;
        return number;
    }

    static string showMsg(string msg, int num) {
        string str = "showMsg() function msg: " + msg + ", " + to_string(num);
        return str;
    }
};

int main() {

    // 普通函数
    packaged_task<string(void)> task1(myFunc);
    // 匿名函数
    packaged_task<int(int)> task2([](int arg) {
        return 100;
        });
    // 仿函数
    Base ba;
    packaged_task<string(string)> task3(ba);

    // 将类对象进行转换得到的函数指针
    Base bb;
    packaged_task<string(string, int)> task4(bb);
    // 静态函数
    packaged_task<string(string, int)> task5(&Base::showMsg);
    // 非静态函数
    Base bc;
    auto obj = bind(&Base::getNumber, &bc, placeholders::_1);
    packaged_task<int(int)> task6(obj);

    thread t1(ref(task6), 200);
    future<int> f = task6.get_future();
    int num = f.get();
    cout << "子线程返回值: " << num << endl;
    t1.join();

    return 0;
}

```















