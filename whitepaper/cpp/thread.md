# 🥬 多线程|互斥锁|条件变量

## 线程thread

### std::thread

在 `#include<thread>`头文件中声明，因此使用 std::thread 时需要包含`#include<thread>` 头文件。

### 基本语法

#### 构造函数

* **默认构造函数**

```cpp
//创建一个空的 thread 执行对象。
thread() _NOEXCEPT
{ // construct with no thread
    _Thr_set_null(_Thr);
}
```

* **初始化构造函数**

```cpp
//创建std::thread执行对象，该thread对象可被joinable，新产生的线程会调用threadFun函数，该函数的参数由 args 给出
template<class Fn, class... Args>
explicit thread(Fn&& fn, Args&&... args);
```

* **拷贝构造函数**

```cpp
// 拷贝构造函数（被禁用），意味着 thread 不可被拷贝构造。
thread(const thread&) = delete;
```

* **Move构造函数**

```cpp
//move 构造函数，调用成功之后 x 不代表任何 thread 执行对象。
//注意：可被 joinable 的 thread 对象必须在他们销毁之前被主线程 join 或者将其设置为detached。
thread(thread&& x)noexcept
```

#### 主要成员函数

* **get\_id**()
  * 获取线程ID，返回类型std::thread::id对象。
  * http://www.cplusplus.com/reference/thread/thread/get\_id/
* **joinable**()
  * 判断线程是否可以加入等待
  * http://www.cplusplus.com/reference/thread/thread/joinable/
* **join**()
  * 等该线程执行完成后才返回。
  * http://www.cplusplus.com/reference/thread/thread/join/
* **detach**()
  * 将本线程从调用线程中分离出来，允许本线程独立执行。(但是当主进程结束的时候，即便是 detach()出去的子线程不管有没有完成都会被强制杀死)
  * http://www.cplusplus.com/reference/thread/thread/detach/

#### 简单线程的创建

简单线程的创建 使用std::thread创建线程，提供线程函数或者函数对象，并可以同时指定线程函数的参数。

```cpp
#include <iostream>
#include <thread>
using namespace std;

void func1()
{
    cout << "func1 into" << endl;
}

void func2(int a, int b)
{
    cout << "func2 a + b = " << a+b << endl;
}

class A
{
public:
    static void fun3(int a)
    {
        cout << "a = " << a << endl;
    }
};

class A
{
public:
    void showMsg(string name, int age)
    {
        cout << "name: " << name << ", age: " << age << endl;
    }

    static void fun3(int a)
    {
        cout << "a = " << a << endl;
    }
};

int main()
{
    std::thread t1(func1); // 只传递函数
    t1.join(); // 阻塞等待线程函数执行结束
    int a =10;
    int b =20;
    std::thread t2(func2, a, b); // 加上参数传递
    t2.join();
    std::thread t3(&A::fun3, 1); // 绑定类静态函数
    t3.join();
    A a;
    thread t4(&A::showMsg, a, "Mike1", 19); // 绑定类的成员函数
    t4.join();
    return 0;
}

```

#### 线程封装

**zero\_thread.h**

```cpp
#ifndef ZERO_THREAD_H
#define ZERO_THREAD_H
#include <thread>

class ZERO_Thread
{
public:
    ZERO_Thread(); // 构造函数
    virtual ~ZERO_Thread(); // 析构函数
    bool start();
    void stop();
    bool isAlive() const; // 线程是否存活.
    std::thread::id id() { return _th->get_id(); }
    std::thread* getThread() { return _th; }
    void join(); // 等待当前线程结束, 不能在当前线程上调用
    void detach(); //能在当前线程上调用
    static size_t CURRENT_THREADID();
protected:
    static void threadEntry(ZERO_Thread *pThread); // 静态函数, 线程入口
    virtual void run() = 0; // 运行
protected:
    bool _running; //是否在运行
    std::thread *_th;
};
#endif // ZERO_THREAD_H
```

**zero\_thread.cpp**

```cpp
#include "zero_thread.h"
#include <sstream>
#include <iostream>
#include <exception>

ZERO_Thread::ZERO_Thread():_running(false), _th(NULL)
{
}
ZERO_Thread::~ZERO_Thread()
{
    if(_th != NULL)
    {
        //如果资源没有被detach或者被join，则自己释放
        if (_th->joinable())
        {
            _th->detach();
        }
        delete _th;
        _th = NULL;
    }
    std::cout << "~ZERO_Thread()" << std::endl;
}
bool ZERO_Thread::start()
{
    if (_running)
    {
        return false;
    }
    try
    {
        _th = new std::thread(&ZERO_Thread::threadEntry, this);
    }
    catch(...)
    {
        throw "[ZERO_Thread::start] thread start error";
    }
    return true;
}
void ZERO_Thread::stop()
{
    _running = false;
}
bool ZERO_Thread::isAlive() const
{
    return _running;
}
void ZERO_Thread::join()
{
    if (_th->joinable())
    {
        _th->join();
    }
}
void ZERO_Thread::detach()
{
    _th->detach();
}
size_t ZERO_Thread::CURRENT_THREADID()
{
    // 声明为thread_local的本地变量在线程中是持续存在的，不同于普通临时变量的生命周期，
    // 它具有static变量一样的初始化特征和生命周期，即使它不被声明为static。
    static thread_local size_t threadId = 0;
    if(threadId == 0 )
    {
        std::stringstream ss;
        ss << std::this_thread::get_id();
        threadId = strtol(ss.str().c_str(), NULL, 0);
    }
    return threadId;
}
void ZERO_Thread::threadEntry(ZERO_Thread *pThread)
{
    pThread->_running = true;
    try
    {
        pThread->run(); // 函数运行所在
    }
    catch (std::exception &ex)
    {
        pThread->_running = false;
        throw ex;
    }
    catch (...)
    {
        pThread->_running = false;
        throw;
    }
    pThread->_running = false;
}
```

**main.cpp**

```cpp
#include <iostream>
#include <chrono>
#include "zero_thread.h"

using namespace std;
class A: public ZERO_Thread
{
    public:
    void run()
    {
        while (_running)
        {
            cout << "print A " << endl;
            std::this_thread::sleep_for(std::chrono::seconds(5));
        }
        cout << "----- leave A " << endl;
    }
};
class B: public ZERO_Thread
{
    public:
    void run()
    {
        while (_running)
        {
            cout << "print B " << endl;
            std::this_thread::sleep_for(std::chrono::seconds(2));
        }
        cout << "----- leave B " << endl;
    }
};
int main()
{
    {
        A a;
        a.start();
        B b;
        b.start();
        std::this_thread::sleep_for(std::chrono::seconds(10));
        a.stop();
        a.join();
        b.stop();
        b.join();
    }
    cout << "Hello World!" << endl;
    return 0;
}

```

`thread`线程类还提供了一个静态方法，用于获取当前计算机的CPU核心数，根据这个结果在程序中创建出数量相等的线程，每个线程独自占有一个CPU核心，这些线程就不用分时复用CPU时间片，此时程序的并发效率是最高的。

```cpp
#include <iostream>
#include <thread>
using namespace std;

int main()
{
    int num = thread::hardware_concurrency();
    cout << "CPU number: " << num << endl;
}
```

### 命名空间 ： `this_thread`

在C++11中不仅添加了线程类，还添加了一个关于线程的命名空间`std::this_thread`，在这个命名空间中提供了四个公共的成员函数，通过这些成员函数就可以对当前线程进行相关的操作了。

#### get\_id() <a href="#id-1-get-id" id="id-1-get-id"></a>

```c
#include <iostream>
#include <thread>
using namespace std;

void func()
{
    cout << "子线程: " << this_thread::get_id() << endl;
}

int main()
{
    cout << "主线程: " << this_thread::get_id() << endl;
    thread t(func);
    t.join();
}
```

程序启动，开始执行`main()`函数，此时只有一个线程也就是主线程。当创建了子线程对象`t`之后，指定的函数`func()`会在子线程中执行，这时通过调用`this_thread::get_id()`就可以得到当前线程的线程ID了。

#### sleep\_for() <a href="#id-2-sleep-for" id="id-2-sleep-for"></a>

同样地线程被创建后也有这五种状态：`创建态`，`就绪态`，`运行态`，`阻塞态(挂起态)`，`退出态(终止态)` ，关于状态之间的转换是一样的，请参考进程，在此不再过多的赘述。

线程和进程的执行有很多相似之处，在计算机中启动的多个线程都需要占用CPU资源，但是CPU的个数是有限的并且每个CPU在同一时间点不能同时处理多个任务。`为了能够实现并发处理，多个线程都是分时复用CPU时间片，快速的交替处理各个线程中的任务。因此多个线程之间需要争抢CPU时间片，抢到了就执行，抢不到则无法执行`（因为默认所有的线程优先级都相同，内核也会从中调度，不会出现某个线程永远抢不到CPU时间片的情况）。

命名空间`this_thread`中提供了一个休眠函数`sleep_for()`，调用这个函数的线程会马上从`运行态`变成`阻塞态`并在这种状态下休眠一定的时长，因为阻塞态的线程已经让出了CPU资源，代码也不会被执行，所以线程休眠过程中对CPU来说没有任何负担。

```cpp
#include <iostream>
#include <thread>
#include <chrono>
using namespace std;

void func()
{
    for (int i = 0; i < 10; ++i)
    {
        this_thread::sleep_for(chrono::seconds(1));
        cout << "子线程: " << this_thread::get_id() << ", i = " << i << endl;
    }
}

int main()
{
    thread t(func);
    t.join();
}
```

在`func()`函数的`for`循环中使用了`this_thread::sleep_for(chrono::seconds(1));`之后，每循环一次程序都会阻塞1秒钟，也就是说每隔1秒才会进行一次输出。<mark style="color:red;">需要注意的是：程序休眠完成之后，会从阻塞态重新变成就绪态，就绪态的线程需要再次争抢CPU时间片，抢到之后才会变成运行态，这时候程序才会继续向下运行。</mark>

#### sleep\_until() <a href="#id-3-sleep-until" id="id-3-sleep-until"></a>

命名空间`this_thread`中提供了另一个休眠函数`sleep_until()`，和`sleep_for()`不同的是它的参数类型不一样

* `sleep_until()`：指定线程阻塞到某一个指定的时间点`time_point`类型，之后解除阻塞
* `sleep_for()`：指定线程阻塞一定的时间长度`duration`类型，之后解除阻塞

```cpp
#include <iostream>
#include <thread>
#include <chrono>
using namespace std;

void func()
{
    for (int i = 0; i < 10; ++i)
    {
        // 获取当前系统时间点
        auto now = chrono::system_clock::now();
        // 时间间隔为2s
        chrono::seconds sec(2);
        // 当前时间点之后休眠两秒
        this_thread::sleep_until(now + sec);
        cout << "子线程: " << this_thread::get_id() << ", i = " << i << endl;
    }
}

int main()
{
    thread t(func);
    t.join();
}
```

`sleep_until()`和`sleep_for()`函数的功能是一样的，只不过前者是基于时间点去阻塞线程，后者是基于时间段去阻塞线程，项目开发过程中根据实际情况选择最优的解决方案即可。

#### yield() <a href="#id-4-yield" id="id-4-yield"></a>

命名空间`this_thread`中提供了一个非常绅士的函数`yield()`，在线程中调用这个函数之后，处于运行态的线程会主动让出自己已经抢到的CPU时间片，最终变为就绪态，这样其它的线程就有更大的概率能够抢到CPU时间片了。<mark style="color:red;">使用这个函数的时候需要注意一点，线程调用了yield()之后会主动放弃CPU资源，但是这个变为就绪态的线程会马上参与到下一轮CPU的抢夺战中，不排除它能继续抢到CPU时间片的情况，这是概率问题。</mark>

```cpp
#include <iostream>
#include <thread>
using namespace std;

void func()
{
    for (int i = 0; i < 100000000000; ++i)
    {
        cout << "子线程: " << this_thread::get_id() << ", i = " << i << endl;
        this_thread::yield();
    }
}

int main()
{
    thread t(func);
    thread t1(func);
    t.join();
    t1.join();
}
```

在上面的程序中，执行`func()`中的`for`循环会占用大量的时间，在极端情况下，如果当前线程占用CPU资源不释放就会导致其他线程中的任务无法被处理，或者该线程每次都能抢到CPU时间片，导致其他线程中的任务没有机会被执行。<mark style="color:red;">解决方案就是每执行一次循环，让该线程主动放弃CPU资源，重新和其他线程再次抢夺CPU时间片，如果其他线程抢到了CPU时间片就可以执行相应的任务了。</mark>

### 互斥量

mutex又称互斥量，C++ 11中与 mutex相关的类（包括锁类型）和函数都声明在 头文件中，所以如果 你需要使用 std::mutex，就必须包含`#include<mutex>`头文件。

C++11提供如下4种语义的互斥量（mutex）

* std::mutex，独占的互斥量，不能递归使用。
* std::time\_mutex，带超时的独占互斥量，不能递归使用。
* std::recursive\_mutex，递归互斥量，不带超时功能。
* std::recursive\_timed\_mutex，带超时的递归互斥量。

{% hint style="info" %}
<mark style="color:red;">多线程中有多少个共享资源就申请多少个互斥量</mark>
{% endhint %}

#### 独占互斥量std::mutex

**std::mutex 介绍**

下面以 std::mutex 为例介绍 C++11 中的互斥量用法。 std::mutex 是C++11 中最基本的互斥量，std::mutex 对象提供了独占所有权的特性——即不支持递归地 对 std::mutex 对象上锁，而 std::recursive\_lock 则可以递归地对互斥量对象上锁。

**std::mutex 的成员函数**

* <mark style="color:red;">构造函数，std::mutex不允许拷贝构造，也不允许 move 拷贝，最初产生的 mutex 对象是处于 unlocked 状态的。</mark>
* lock()，调用线程将锁住该互斥量。线程调用该函数会发生下面 3 种情况：
  * 如果该互斥量当前没 有被锁住，则调用线程将该互斥量锁住，直到调用 unlock之前，该线程一直拥有该锁。
  * <mark style="color:red;">如果当 前互斥量被其他线程锁住，则当前的调用线程被阻塞住。</mark>
  * 如果当前互斥量被当前调用线程锁 住，则会产生死锁(deadlock)。
* unlock()， 解锁，释放对互斥量的所有权。
* try\_lock()，尝试锁住互斥量，如果互斥量被其他线程占有，则当前线程也不会被阻塞。线程调用该 函数也会出现下面 3 种情况，
  * 如果当前互斥量没有被其他线程占有，则该线程锁住互斥量，直 到该线程调用 unlock 释放互斥量。
  * 如果当前互斥量被其他线程锁住，则当前调用线程返回 false，而并不会被阻塞掉。
  * 如果当前互斥量被当前调用线程锁住，则会产生死锁(deadlock)。

```cpp
#include <iostream> // std::cout
#include <thread> // std::thread
#include <mutex> // std::mutex

volatile int counter(0); // non-atomic counter
std::mutex mtx; // locks access to counter
void increases_10k()
{
    for (int i=0; i<10000; ++i) {
        // 1. 使用try_lock的情况
        // if (mtx.try_lock()) { // only increase if currently not locked:
        // ++counter;
        // mtx.unlock();
        // }
        // 2. 使用lock的情况
        {
            mtx.lock();
            ++counter;
            mtx.unlock();
        }
    }
}

int main()
{
    std::thread threads[10];
    for (int i=0; i<10; ++i)
        threads[i] = std::thread(increases_10k);
    for (auto& th : threads) th.join();
    std::cout << " successful increases of the counter " << counter << std::endl;
    return 0;
}
```

#### 递归互斥量std::recursive\_mutex

递归锁允许同一个线程多次获取该互斥锁，可以用来解决同一线程需要多次获取互斥量时死锁的问题。

```cpp
#include <iostream>
#include <thread>
#include <mutex>

struct Complex
{
    std::mutex mutex;
    int i;
    Complex() : i(0){}
    void mul(int x)
    {
        std::lock_guard<std::mutex> lock(mutex);
        i *= x;
    }
    void div(int x)
    {
        std::lock_guard<std::mutex> lock(mutex);
        i /= x;
    }
    void both(int x, int y)
    {
        std::lock_guard<std::mutex> lock(mutex);
        mul(x);
        div(y);
    }
};
int main(void)
{
    Complex complex;
    complex.both(32, 23);
    return 0;
}

```

使用递归锁

```cpp
#include <iostream>
#include <thread>
#include <mutex>
struct Complex
{
    std::recursive_mutex mutex;
    int i;
    Complex() : i(0){}
    void mul(int x)
    {
        std::lock_guard<std::recursive_mutex> lock(mutex);
        i *= x;
    }
    void div(int x)
    {
        std::lock_guard<std::recursive_mutex> lock(mutex);
        i /= x;
    }
    void both(int x, int y)
    {
        std::lock_guard<std::recursive_mutex> lock(mutex);
        mul(x);
        div(y);
    }
};
int main(void)
{
    Complex complex;
    complex.both(32, 23); //因为同一线程可以多次获取同一互斥量，不会发生死锁
    std::cout << "main finish\n";
    return 0;
}

```

虽然递归锁能解决这种情况的死锁问题，但是尽量不要使用递归锁，主要原因如下：

1. 需要用到递归锁的多线程互斥处理本身就是可以简化的，允许递归很容易放纵复杂逻辑的产生，并 且产生晦涩，当要使用递归锁的时候应该重新审视自己的代码是否一定要使用递归锁；
2. 递归锁比起非递归锁，效率会低；
3. 递归锁虽然允许同一个线程多次获得同一个互斥量，但可重复获得的最大次数并未具体说明，一旦 超过一定的次数，再对lock进行调用就会抛出std::system错误。

#### 带超时的互斥量std::timed\_mutex和 std::recursive\_timed\_mutex

std::timed\_mutex比std::mutex多了两个超时获取锁的接口：try\_lock\_for和try\_lock\_until

```cpp
#include <iostream>
#include <thread>
#include <mutex>
#include <chrono>

std::timed_mutex mutex;
void work()
{
    std::chrono::milliseconds timeout(100);
    while (true)
    {
        if (mutex.try_lock_for(timeout))
        {
            std::cout << std::this_thread::get_id() << ": do work with the
                mutex" << std::endl;
                std::chrono::milliseconds sleepDuration(250);
            std::this_thread::sleep_for(sleepDuration);
            mutex.unlock();
            std::this_thread::sleep_for(sleepDuration);
        }
        else
        {
            std::cout << std::this_thread::get_id() << ": do work without the
                mutex" << std::endl;
                std::chrono::milliseconds sleepDuration(100);
            std::this_thread::sleep_for(sleepDuration);
        }
    }
}
int main(void)
{
    std::thread t1(work);
    std::thread t2(work);
    t1.join();
    t2.join();
    std::cout << "main finish\n";
    return 0;
}

```

#### <mark style="color:red;">lock\_guard和unique\_lock的使用和区别</mark>

相对于手动lock和unlock，我们可以使用RAII(通过类的构造析构)来实现更好的编码方式。 这里涉及到unique\_lock,lock\_guard的使用。 ps: C++相较于C引入了很多新的特性, 比如可以在代码中抛出异常, 如果还是按照以前的加锁解锁的话，代码会极为复杂繁琐

```cpp
#include <iostream> // std::cout
#include <thread> // std::thread
#include <mutex> // std::mutex, std::lock_guard
#include <stdexcept> // std::logic_error

std::mutex mtx;
void print_even (int x) {
    if (x%2==0) std::cout << x << " is even\n";
    else throw (std::logic_error("not even"));
}

void print_thread_id (int id) {
    try {
        // using a local lock_guard to lock mtx guarantees unlocking on destruction / exception:
        std::lock_guard<std::mutex> lck (mtx);
        print_even(id);
    }
    catch (std::logic_error&) {
        std::cout << "[exception caught]\n";
    }
}

int main ()
{
    std::thread threads[10];
    // spawn 10 threads:
    for (int i=0; i<10; ++i)
        threads[i] = std::thread(print_thread_id,i+1);
    for (auto& th : threads) th.join();
    return 0;
}

```

这里的lock\_guard换成unique\_lock是一样的。 unique\_lock,lock\_guard的区别

* <mark style="color:red;">unique\_lock与lock\_guard都能实现自动加锁和解锁，但是前者更加灵活，能实现更多的功能。</mark>
* <mark style="color:red;">unique\_lock可以进行临时解锁和再上锁，如在构造对象之后使用lck.unlock()就可以进行解锁， lck.lock()进行上锁，而不必等到析构时自动解锁。</mark>

```cpp
#include <iostream>
#include <deque>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <unistd.h>

std::deque<int> q;
std::mutex mu;
std::condition_variable cond;

void fun1() {
    while (true) {
        std::unique_lock<std::mutex> locker(mu);
        q.push_front(count);
        locker.unlock();
        cond.notify_one();
        sleep(10);
    }
}

void fun2() {
    while (true) {
        std::unique_lock<std::mutex> locker(mu);
        cond.wait(locker, [](){return !q.empty();});
        data = q.back();
        q.pop_back();
        locker.unlock();
        std::cout << "thread2 get value form thread1: " << data << std::endl;
    }
}

int main() {
    std::thread t1(fun1);
    std::thread t2(fun2);
    t1.join();
    t2.join();
    return 0;
}
```

条件变量的目的就是为了，在没有获得某种提醒时长时间休眠; 如果正常情况下, 我们需要一直循环 (`++sleep`), 这样的问题就是CPU消耗+时延问题，条件变量的意思是在cond.wait这里一直休眠直到 cond.notify\_one唤醒才开始执行下一句; 还有cond.notify\_all接口用于唤醒所有等待的线程。

**那么为什么必须使用unique\_lock呢?**

> 原因: 条件变量在wait时会进行unlock再进入休眠, lock\_guard并无该操作接口

* wait: 如果线程被唤醒或者超时那么会先进行lock获取锁, 再判断条件(传入的参数)是否成立, 如果成立则 wait函数返回否则释放锁继续休眠
* notify: 进行notify动作并不需要获取锁

#### 总结

**lock\_guard**

1. std::lock\_guard 在构造函数中进行加锁，析构函数中进行解锁。
2. 锁在多线程编程中，使用较多，因此c++11提供了lock\_guard模板类；在实际编程中，我们也可以根 据自己的场景编写resource\_guard RAII类，避免忘掉释放资源。

**std::unique\_lock**

1. unique\_lock 是通用互斥包装器，允许延迟锁定、锁定的有时限尝试、递归锁定、所有权转移和与 条件变量一同使用。
2. unique\_lock比lock\_guard使用更加灵活，功能更加强大。
3. 使用unique\_lock需要付出更多的时间、性能成本。

### 条件变量

互斥量是多线程间同时访问某一共享变量时，保证变量可被安全访问的手段。但单靠互斥量无法实现线 程的同步。线程同步是指线程间需要按照预定的先后次序顺序进行的行为。C++11对这种行为也提供了 有力的支持，这就是条件变量。条件变量位于头文件`condition_variable`下。 [http://www.cplusplus.com/reference/condition\_variable/condition\_variable](http://www.cplusplus.com/reference/condition\_variable/condition\_variable)

条件变量使用过程：

1. 拥有条件变量的线程获取互斥量；
2. 循环检查某个条件，如果条件不满足则阻塞直到条件满足；如果条件满足则向下执行；
3. 某个线程满足条件执行完之后调用notify\_one或notify\_all唤醒一个或者所有等待线程。 条件变量提供了两类操作：wait和notify。这两类操作构成了多线程同步的基础。

> * 条件变量存放了被阻塞线程的线程ID
> * condition\_variable：需要配合std::unique\_lock\<std::mutex>进行wait操作，也就是阻塞线程的操作。
> * &#x20;condition\_variable\_any：可以和任意带有lock()、unlock()语义的mutex搭配使用，也就是说有四种：&#x20;
>   * std::mutex：独占的非递归互斥锁&#x20;
>   * std::timed\_mutex：带超时的独占非递归互斥锁&#x20;
>   * std::recursive\_mutex：不带超时功能的递归互斥锁&#x20;
>   * std::recursive\_timed\_mutex：带超时的递归互斥锁

#### 成员函数

**wait函数**

**函数原型**

```cpp
void wait (unique_lock<mutex>& lck);
template <class Predicate>
void wait (unique_lock<mutex>& lck, Predicate pred);
```

包含两种重载，第一种只包含unique\_lock对象，另外一个Predicate 对象（等待条件），这里必须使用 unique\_lock，因为wait函数的工作原理：

* 当前线程调用wait()后将被阻塞并且函数会解锁互斥量，直到另外某个线程调用notify\_one或者 notify\_all唤醒当前线程；<mark style="color:red;">一旦当前线程获得通知(notify)，wait()函数也是自动调用lock()，同理不能使用lock\_guard对象。</mark>
* 如果wait没有第二个参数，<mark style="color:red;">第一次调用默认条件不成立，直接解锁互斥量并阻塞到本行</mark>，直到某一 个线程调用notify\_one或notify\_all为止，被唤醒后，wait重新尝试获取互斥量，如果得不到，线程会卡在这里，直到获取到互斥量，然后无条件地继续进行后面的操作。
* 如果wait包含第二个参数，如果第二个参数不满足，那么wait将解锁互斥量并堵塞到本行，直到某 一个线程调用notify\_one或notify\_all为止，被唤醒后，wait重新尝试获取互斥量，如果得不到，线程会卡在这里，直到获取到互斥量，然后继续判断第二个参数，如果表达式为false，wait对互斥量解锁，然后休眠，如果为true，则进行后面的操作。

{% hint style="info" %}
<mark style="color:red;">wait阻塞之前会解锁，解除阻塞之后加锁</mark>
{% endhint %}

**wait\_for函数**

函数原型：

```cpp
template <class Rep, class Period>
cv_status wait_for (unique_lock<mutex>& lck,
         const chrono::duration<Rep,Period>& rel_time);
         
template <class Rep, class Period, class Predicate>
bool wait_for (unique_lock<mutex>& lck,
    const chrono::duration<Rep,Period>& rel_time, Predicate
    pred);

```

和wait不同的是，wait\_for可以执行一个时间段，在线程收到唤醒通知或者时间超时之前，该线程都会 处于阻塞状态，如果收到唤醒通知或者时间超时，wait\_for返回，剩下操作和wait类似。

**wait\_until函数**

函数原型：

```cpp
template <class Clock, class Duration>
cv_status wait_until (unique_lock<mutex>& lck,
    const chrono::time_point<Clock,Duration>& abs_time);
                          
template <class Clock, class Duration, class Predicate>
bool wait_until (unique_lock<mutex>& lck,
    const chrono::time_point<Clock,Duration>& abs_time,
    Predicate pred);

```

**notify\_one函数**

函数原型：

<pre class="language-cpp"><code class="lang-cpp"><strong>void notify_one() noexcept;
</strong></code></pre>

解锁正在等待当前条件的线程中的一个，如果没有线程在等待，则函数不执行任何操作，如果正在等待 的线程多余一个，则唤醒的线程是不确定的。

**notify\_all函数**

函数原型：

```cpp
void notify_all() noexcept;
```

解锁正在等待当前条件的所有线程，如果没有正在等待的线程，则函数不执行任何操作。

{% hint style="info" %}
<mark style="color:red;">两个方法所产生的结果是一样的，每次都只会有一个线程工作</mark>
{% endhint %}

#### 范例

使用条件变量实现一个同步队列，同步队列作为一个线程安全的数据共享区，经常用于线程之间数据读 取。

`sync_queue.h`

```cpp
#ifndef SYNC_QUEUE_H
#define SYNC_QUEUE_H
#include<list>
#include<mutex>
#include<thread>
#include<condition_variable>
#include <iostream>

template<typename T>
class SyncQueue
{
private:
    bool IsFull() const
    {
        return _queue.size() == _maxSize;
    }
    bool IsEmpty() const
    {
        return _queue.empty();
    }
public:
    SyncQueue(int maxSize) : _maxSize(maxSize)
    {
    }
    void Put(const T& x)
    {
        std::lock_guard<std::mutex> locker(_mutex);
        while (IsFull())
        {
            std::cout << "full wait..." << std::endl;
            _notFull.wait(_mutex);
        }
        _queue.push_back(x);
        _notFull.notify_one();
    }
    void Take(T& x)
    {
        std::lock_guard<std::mutex> locker(_mutex);
        //如果只有一个任务，但是唤醒了多个消费者线程，
        //则需要消费者线程wait后判断队列是不是空的，解决方法就是将if empty改为while empty
        while (IsEmpty())
        {
            std::cout << "empty wait.." << std::endl;
            _notEmpty.wait(_mutex);
        }
        x = _queue.front();
        _queue.pop_front();
        _notFull.notify_one();
    }
    bool Empty()
    {
        std::lock_guard<std::mutex> locker(_mutex);
        return _queue.empty();
    }
    bool Full()
    {
        std::lock_guard<std::mutex> locker(_mutex);
        return _queue.size() == _maxSize;
    }
    size_t Size()
    {
        std::lock_guard<std::mutex> locker(_mutex);
        return _queue.size();
    }
    int Count()
    {
        return _queue.size();
    }
private:
    std::list<T> _queue; //缓冲区
    std::mutex _mutex; //互斥量和条件变量结合起来使用
    std::condition_variable_any _notEmpty;//不为空的条件变量
    std::condition_variable_any _notFull; //没有满的条件变量
    int _maxSize; //同步队列最大的size
};
#endif // SYNC_QUEUE_H
```

`main.cpp`

```cpp
#include <iostream>
#include "sync_queue.h"
#include <thread>
#include <iostream>
#include <mutex>

using namespace std;
SyncQueue<int> syncQueue(5);
void PutDatas()
{
    for (int i = 0; i < 20; ++i)
    {
        syncQueue.Put(888);
    }
    std::cout << "PutDatas finish\n";
}
void TakeDatas()
{
    int x = 0;
    for (int i = 0; i < 20; ++i)
    {
        syncQueue.Take(x);
        std::cout << x << std::endl;
    }
    std::cout << "TakeDatas finish\n";
}
int main(void)
{
    std::thread t1(PutDatas);
    std::thread t2(TakeDatas);
    t1.join();
    t2.join();
    std::cout << "main finish\n";
    return 0;
}
```

代码中用到了std::lock\_guard，它利用RAII机制可以保证安全释放mutex。

```cpp
std::lock_guard<std::mutex> locker(_mutex);
while (IsFull())
{
    std::cout << "full wait..." << std::endl;
    _notFull.wait(_mutex);
}
```

可以改成

```cpp
std::lock_guard<std::mutex> locker(_mutex);
_notFull.wait(_mutex， [this] {return !IsFull();});
```

两种写法效果是一样的，但是后者更简洁，条件变量会先检查判断式是否满足条件，如果满足条件则重 新获取mutex，然后结束wait继续往下执行；如果不满足条件则释放mutex，然后将线程置为waiting状 态继续等待。

<mark style="color:red;">这里需要注意的是，wait函数中会释放mutex，而lock\_guard这时还拥有mutex，它只会在出了作用域 之后才会释放mutex，所以这时它并不会释放，但执行wait时会提取释放mutex。 从语义上看这里使用lock\_guard会产生矛盾，但是实际上并不会出问题，因为wait提前释放锁之后会处 于等待状态，在被notify\_one或者notify\_all唤醒后会先获取mutex，这相当于lock\_guard的mutex在 释放之后又获取到了，因此，在出了作用域之后lock\_guard自动释放mutex不会有问题。 这里应该用unique\_lock，因为unique\_lock不像lock\_guard一样只能在析构时才释放锁，它可以随时释 放锁，因此在wait时让unique\_lock释放锁从语义上更加准确。</mark>

使用unique\_lock和condition\_variable\_variable改写上面的代码，用等待一个判 断式的方法来实现一个简单的队列。&#x20;

```cpp
#ifndef SIMPLE_SYNC_QUEUE_H
#define SIMPLE_SYNC_QUEUE_H
#include <thread>
#include <condition_variable>
#include <mutex>
#include <list>
#include <iostream>

template<typename T>
class SimpleSyncQueue
{
public:
    SimpleSyncQueue(){}
    void Put(const T& x)
    {
        std::lock_guard<std::mutex> locker(_mutex);
        _queue.push_back(x);
        _notEmpty.notify_one();
    }
    void Take(T& x)
    {
        std::unique_lock<std::mutex> locker(_mutex);
        _notEmpty.wait(locker, [this]{return !_queue.empty(); });
        x = _queue.front();
        _queue.pop_front();
    }
    bool Empty()
    {
        std::lock_guard<std::mutex> locker(_mutex);
        return _queue.empty();
    }
    size_t Size()
    {
        std::lock_guard<std::mutex> locker(_mutex);
        return _queue.size();
    }
private:
    std::list<T> _queue;
    std::mutex _mutex;
    std::condition_variable _notEmpty;
};
#endif // SIMPLE_SYNC_QUEUE_H
```

```cpp
#include <iostream>
#include "sync_queue.h"
#include <thread>
#include <iostream>
#include <mutex>
using namespace std;
SimpleSyncQueue<int> syncQueue;
void PutDatas()
{
    for (int i = 0; i < 20; ++i)
    {
        syncQueue.Put(888);
    }
}
void TakeDatas()
{
    int x = 0;
    for (int i = 0; i < 20; ++i)
    {
        syncQueue.Take(x);
        std::cout << x << std::endl;
    }
}
int main(void)
{
    std::thread t1(PutDatas);
    std::thread t2(TakeDatas);
    t1.join();
    t2.join();
    std::cout << "main finish\n";
    return 0;
}

```

> 在多线程环境中，当使用固定大小的队列（如基于数组的队列）时，我们通常需要两个条件变量：一个用于同步队列非空（`not_empty`），另一个用于同步队列未满（`not_full`）。这是因为固定大小的队列在满了之后不能再添加新元素，否则会发生溢出；同样，空了之后就不能移除元素，否则会发生下标越界。
>
> 然而，`std::list` 是一个动态数据结构，它不基于连续的内存分配，而是通过指针链接各个元素。这意味着 `std::list` 可以动态地增长和缩减，没有固定的最大容量限制（除了系统内存的大小）。因此，除非自己实现了某种形式的容量限制逻辑，否则 `std::list` 本身不会因添加元素而“溢出”。

### call\_once和once\_flag使用

具体：https://www.apiref.com/cpp-zh/cpp/thread/call\_once.html 在多线程中，有一种场景是某个任务只需要执行一次，可以用C++11中的`std::call_once`函数配合 `std::once_flag`来实现。多个线程同时调用某个函数，`std::call_once`可以保证多个线程对该函数只调用一 次。

```cpp
#include <iostream>
#include <thread>
#include <mutex>

std::once_flag flag1, flag2;
void simple_do_once()
{
    std::cout << "simple_do_once\n" ;
    std::call_once(flag1, [](){ std::cout << "Simple example: called once\n";
                              });
}
void may_throw_function(bool do_throw)
{
    if (do_throw) {
        std::cout << "throw: call_once will retry\n"; //
        throw std::exception();
    }
    std::cout << "Didn't throw, call_once will not attempt again\n"; // 保证一次
}
void do_once(bool do_throw)
{
    try {
        std::call_once(flag2, may_throw_function, do_throw);
    }
    catch (...) {
    }
}
int main()
{
    std::thread st1(simple_do_once);
    std::thread st2(simple_do_once);
    std::thread st3(simple_do_once);
    std::thread st4(simple_do_once);
    st1.join();
    st2.join();
    st3.join();
    st4.join();
    std::thread t1(do_once, false);
    std::thread t2(do_once, false);
    std::thread t3(do_once, false);
    std::thread t4(do_once, true);
    t1.join();
    t2.join();
    t3.join();
    t4.join();
}

```

再看看单例模式下的应用：

```cpp
#include <iostream>
#include <mutex>
#include <thread>
using namespace std;


once_flag g_flag;
// 编写一个单例模式的类-->懒汉模式：在第一次调用getInstance()方法时，才会创建对象
// 多线程环境下，懒汉模式是线程不安全的，多个线程同时调用getInstance()方法时，会创建多个对象
// 解决方案：加锁，但是效率低
// 解决方案：C++11之后，使用call_once()函数，保证线程安全
class Base
{
public:
    Base(const Base& obj) = delete;
    Base& operator=(const Base& obj) = delete;
    static Base* getInstance()
    {
        call_once(g_flag, [&]()
            {
                obj = new Base;
                cout << "Base实例来也!!!" << endl;
            });
        return obj;
    }

    void setName(string name)
    {
        this->name = name;
    }

    string getName()
    {
        return name;
    }
private:
    Base() {};
    static Base* obj;
    string name;
};
Base* Base::obj = nullptr;

void myFunc(string name)
{
    Base::getInstance()->setName(name);
    cout << "my name is: " << Base::getInstance()->getName() << endl;
}

int main()
{
    thread t1(myFunc, "Mike");
    thread t2(myFunc, "Conan");
    thread t3(myFunc, "Like");
    thread t4(myFunc, "Nini");
    t1.join();
    t2.join();
    t3.join();
    t4.join();
    return 0;
}
```

### Reference

* [参考网页](https://blog.csdn.net/li1615882553/article/details/86179781)
* [https://subingwen.cn/cpp/thread/](https://subingwen.cn/cpp/thread/)
* [https://subingwen.cn/cpp/this\_thread/](https://subingwen.cn/cpp/this\_thread/)
* [https://subingwen.cn/cpp/call\_once/](https://subingwen.cn/cpp/call\_once/)
* [https://subingwen.cn/cpp/call\_once/](https://subingwen.cn/cpp/call\_once/)
* [https://subingwen.cn/cpp/mutex/](https://subingwen.cn/cpp/mutex/)
* [https://subingwen.cn/cpp/condition/](https://subingwen.cn/cpp/condition/)
