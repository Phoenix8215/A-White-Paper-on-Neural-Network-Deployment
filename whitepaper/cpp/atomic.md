# 🥒 异步操作|C++11

### 异步操作

* `std::future`
* `std::aysnc`
* `std::promise`
* `std::packaged_task`

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

{% hint style="info" %}
`get()` 既有等待又有获取结果的功能，而 `wait()` 只有等待的功能。
{% endhint %}

`wait_until()`和`wait_for()`函数功能是差不多，前者是阻塞到某一指定的时间点，后者是阻塞一定的时长。

```cpp
template< class Rep, class Period >
std::future_status wait_for( const std::chrono::duration<Rep,Period>& timeout_duration ) const;

template< class Clock, class Duration >
std::future_status wait_until( const std::chrono::time_point<Clock,Duration>& timeout_time ) const;
```

当`wait_until()`和`wait_for()`函数返回之后，并不能确定子线程当前的状态，因此我们需要判断函数的返回值，这样就能知道子线程当前的状态了：

<table><thead><tr><th width="283">常量</th><th>解释</th></tr></thead><tbody><tr><td><a href="https://zh.cppreference.com/w/cpp/thread/future_status"><code>future_status::deferred</code></a></td><td>子线程中的任务函仍未启动</td></tr><tr><td><a href="https://zh.cppreference.com/w/cpp/thread/future_status"><code>future_status::ready</code></a></td><td>子线程中的任务已经执行完毕，结果已就绪</td></tr><tr><td><a href="https://zh.cppreference.com/w/cpp/thread/future_status"><code>future_status::timeout</code></a></td><td>子线程中的任务正在执行中，指定等待时长已用完</td></tr></tbody></table>

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

```c
do {
    status = f.wait_for(chrono::seconds(1));
    if (status == future_status::deferred) {
        cout << "线程还没有执行..." << endl;
        f.wait();
    } else if (status == future_status::ready) {
        cout << "子线程返回值: " << f.get() << endl;
    } else if (status == future_status::timeout) {
        cout << "任务还未执行完毕, 继续等待..." << endl;
    }
} while (status != future_status::ready);
```

### std::packaged\_task

&#x20;`The class template std::packaged_task wraps any Callable target (function, lambda expression, bind expression, or another function object) so that it can be invoked asynchronously. Its return value or exception thrown is stored in a shared state which can be accessed through std::future objects.`

可以通过std::packaged\_task对象获取任务相关联的future，调用get\_future()方法可以获得 std::packaged\_task对象绑定的函数的返回值类型的future。std::packaged\_task的模板参数是函数签 名。 PS：例如int add(int a, intb)的函数签名就是int(int, int)

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

### `std::promise`、`std::packaged_task`和`std::future`的关系

至此, 我们介绍了std::async相关的几个对象std::future、std::promise和std::packaged\_task，其中 std::promise和std::packaged\_task的结果最终都是通过其内部的future返回出来的，不知道读者有没有搞糊涂，为什么有 这么多东西出来，他们之间的关系到底是怎样的？且听我慢慢道来，<mark style="color:red;">std::future提供了一个访问异步操作结果的机制，它和线程是一个级别的，属于低层次的对象，在它之上高一层的是std::packaged\_task和std::promise，他们内部都有future以便访问异步操作结果，std::packaged\_task包装的是一个异步操作，而std::promise包装的是一个值，都是为了方便异步操作的，因为有时我需要获取线程中的某个值，这时就用std::promise，而有时我需要获一个异步操作的返回值，这时就用std::packaged\_task。</mark>

### **`std::async`**

std::async是为了让用户的少费点脑子的，它让std::future、std::promise和std::packaged\_task这三个对象默契的工作。大概的工作过程是这样的：std::async先将异步操作用std::packaged\_task包装起来，然后将异步操作的结果放到std::promise中，这个过程就是创造future()的过程。外面再通过future.get()/wait()来获取future()的结果，你不用再想到底该怎么用std::future、std::promise和 std::packaged\_task了，std::async已经帮你搞定一切了！

现在来看看std::async的原型async(std::launch::async | std::launch::deferred, f, args...)，第一个参数是线程的创建策略，有两种策略，默认的策略是立即创建线程：

* `std::launch::async`：在调用async就开始创建线程。
* `std::launch::deferred`：延迟加载方式创建线程。调用async时不创建线程，直到调用了future()的get()或者wait()时才创建线程。

第二个参数是线程函数，第三个参数是线程函数的参数，函数返回值是一个`future`对象。

> <mark style="color:red;">`get()`</mark> <mark style="color:red;"></mark><mark style="color:red;">既有等待又有获取结果的功能，而</mark> <mark style="color:red;"></mark><mark style="color:red;">`wait()`</mark> <mark style="color:red;"></mark><mark style="color:red;">只有等待的功能。</mark>

关于`std::async()`函数的使用，对应的示例代码如下：

* **调用async()函数直接创建线程执行任务**

```c
#include <iostream>
#include <thread>
#include <future>
using namespace std;

int main()
{
    cout << "主线程ID: " << this_thread::get_id() << endl;
    // 调用函数直接创建线程执行任务
    future<int> f = async([](int x) {
        cout << "子线程ID: " << this_thread::get_id() << endl;
        this_thread::sleep_for(chrono::seconds(5));
        return x += 100;
    }, 100);

    future_status status;
    do {
        status = f.wait_for(chrono::seconds(1));
        if (status == future_status::deferred)
        {
            cout << "线程还没有执行..." << endl;
            f.wait();
        }
        else if (status == future_status::ready)
        {
            cout << "子线程返回值: " << f.get() << endl;
        }
        else if (status == future_status::timeout)
        {
            cout << "任务还未执行完毕, 继续等待..." << endl;
        }
    } while (status != future_status::ready);

    return 0;
}
```

示例程序输出的结果为：

```bash
主线程ID: 8904
子线程ID: 25036
任务还未执行完毕, 继续等待...
任务还未执行完毕, 继续等待...
任务还未执行完毕, 继续等待...
任务还未执行完毕, 继续等待...
任务还未执行完毕, 继续等待...
子线程返回值: 200
```

调用`async()`函数时不指定策略就是直接创建线程并执行任务，示例代码的主线程中做了如下操作`status = f.wait_for(chrono::seconds(1));`其实直接调用`f.get()`(get本身就有等待，没必要多此一举)就能得到子线程的返回值。这里为了给大家演示`wait_for()`的使用，所以写的复杂了些。

* **调用async()函数不创建线程执行任务**

```c
#include <iostream>
#include <thread>
#include <future>
using namespace std;

int main()
{
    cout << "主线程ID: " << this_thread::get_id() << endl;
    // 调用函数直接创建线程执行任务
    future<int> f = async(launch::deferred, [](int x) {
        cout << "子线程ID: " << this_thread::get_id() << endl;
        return x += 100;
    }, 100);

    this_thread::sleep_for(chrono::seconds(5));
    cout << f.get();

    return 0;
}
```

示例程序输出的结果：

```bash
主线程ID: 24760
主线程开始休眠5秒...
子线程ID: 24760
200
```

由于指定了`launch::deferred` 策略，因此调用`async()`函数并不会创建新的线程执行任务，当使用`future`类对象调用了`get()`或者`wait()`方法后才开始执行任务（此处一定要注意调用wait\_for()函数是不行的）。

通过测试程序输出的结果可以看到，两次输出的线程ID是相同的，任务函数是在主线程中被延迟（主线程休眠了5秒）调用了。

### reference

* [https://subingwen.cn/cpp/atomic/](https://subingwen.cn/cpp/atomic/)
* [https://subingwen.cn/cpp/async/](https://subingwen.cn/cpp/async/)
* [https://www.cnblogs.com/chengyuanchun/p/5394843.html](https://www.cnblogs.com/chengyuanchun/p/5394843.html)
* [https://ryonaldteofilo.medium.com/atomics-in-c-compare-and-swap-and-memory-order-part-2-64e127847e00](https://ryonaldteofilo.medium.com/atomics-in-c-compare-and-swap-and-memory-order-part-2-64e127847e00)
