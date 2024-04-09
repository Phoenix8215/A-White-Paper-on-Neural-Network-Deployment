# C++11多线程

## 线程thread

### std::thread

在 `#include<thread>`头文件中声明，因此使用 std::thread 时需要包含\*\*`#include<thread>`\*\* 头文件。

### 语法

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
//创建std::thread执行对象，该thread对象可被joinable，新产生的线程会调用threadFun函数，该函
数的参数由 args 给出
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
//注意：可被 joinable 的 thread 对象必须在他们销毁之前被主线程 join 或者将其设置为
detached。
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
    return 0;
}

```

#### 线程封装

见范例1-thread2-pack zero\_thread.h

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
ZERO_Thread::ZERO_Thread():
_running(false), _th(NULL)
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

### 互斥量

mutex又称互斥量，C++ 11中与 mutex相关的类（包括锁类型）和函数都声明在 头文件中，所以如果 你需要使用 std::mutex，就必须包含\*\*`#include<mutex>`\*\*头文件。

C++11提供如下4种语义的互斥量（mutex）

* std::mutex，独占的互斥量，不能递归使用。
* std::time\_mutex，带超时的独占互斥量，不能递归使用。
* std::recursive\_mutex，递归互斥量，不带超时功能。
* std::recursive\_timed\_mutex，带超时的递归互斥量。

#### 独占互斥量std::mutex

**std::mutex 介绍**

下面以 std::mutex 为例介绍 C++11 中的互斥量用法。 std::mutex 是C++11 中最基本的互斥量，std::mutex 对象提供了独占所有权的特性——即不支持递归地 对 std::mutex 对象上锁，而 std::recursive\_lock 则可以递归地对互斥量对象上锁。

**std::mutex 的成员函数**

* 构造函数，std::mutex不允许拷贝构造，也不允许 move 拷贝，最初产生的 mutex 对象是处于 unlocked 状态的。
* lock()，调用线程将锁住该互斥量。线程调用该函数会发生下面 3 种情况：
  * 如果该互斥量当前没 有被锁住，则调用线程将该互斥量锁住，直到调用 unlock之前，该线程一直拥有该锁。
  * 如果当 前互斥量被其他线程锁住，则当前的调用线程被阻塞住。
  * 如果当前互斥量被当前调用线程锁 住，则会产生死锁(deadlock)。
* unlock()， 解锁，释放对互斥量的所有权。
* try\_lock()，尝试锁住互斥量，如果互斥量被其他线程占有，则当前线程也不会被阻塞。线程调用该 函数也会出现下面 3 种情况，
  * 如果当前互斥量没有被其他线程占有，则该线程锁住互斥量，直 到该线程调用 unlock 释放互斥量。
  * 如果当前互斥量被其他线程锁住，则当前调用线程返回 false，而并不会被阻塞掉。
  * 如果当前互斥量被当前调用线程锁住，则会产生死锁(deadlock)。

```cpp
//1-2-mutex1
#include <iostream> // std::cout
#include <thread> // std::thread
#include <mutex> // std::mutex
volatile int counter(0); // non-atomic counter
std::mutex mtx; // locks access to counter
void increases_10k()
{
    for (int i=0; i<10000; ++i) {
        // 1. 使用try_lock的情况
        // if (mtx.try_lock()) { // only increase if currently not
        locked:
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
    std::cout << " successful increases of the counter " << counter <<
        std::endl;
    return 0;
}
```

#### 递归互斥量std::recursive\_mutex

递归锁允许同一个线程多次获取该互斥锁，可以用来解决同一线程需要多次获取互斥量时死锁的问题。

```cpp
//死锁范例1-2-mutex2-dead-lock
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
//递归锁1-2-recursive_mutex1
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
//1-2-timed_mutex
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

#### lock\_guard和unique\_lock的使用和区别

相对于手动lock和unlock，我们可以使用RAII(通过类的构造析构)来实现更好的编码方式。 这里涉及到unique\_lock,lock\_guard的使用。 ps: C++相较于C引入了很多新的特性, 比如可以在代码中抛出异常, 如果还是按照以前的加锁解锁的话代 码会极为复杂繁琐

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
        // using a local lock_guard to lock mtx guarantees unlocking on
        destruction / exception:
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

* unique\_lock与lock\_guard都能实现自动加锁和解锁，但是前者更加灵活，能实现更多的功能。
* unique\_lock可以进行临时解锁和再上锁，如在构造对象之后使用lck.unlock()就可以进行解锁， lck.lock()进行上锁，而不必等到析构时自动解锁。

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

条件变量的目的就是为了，在没有获得某种提醒时长时间休眠; 如果正常情况下, 我们需要一直循环 (+sleep), 这样的问题就是CPU消耗+时延问题，条件变量的意思是在cond.wait这里一直休眠直到 cond.notify\_one唤醒才开始执行下一句; 还有cond.notify\_all()接口用于唤醒所有等待的线程。

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

互斥量是多线程间同时访问某一共享变量时，保证变量可被安全访问的手段。但单靠互斥量无法实现线 程的同步。线程同步是指线程间需要按照预定的先后次序顺序进行的行为。C++11对这种行为也提供了 有力的支持，这就是条件变量。条件变量位于\*\*头文件`condition_variable`\*\*下。 http://www.cplusplus.com/reference/condition\_variable/condition\_variable

条件变量使用过程：

1. 拥有条件变量的线程获取互斥量；
2. 循环检查某个条件，如果条件不满足则阻塞直到条件满足；如果条件满足则向下执行；
3. 某个线程满足条件执行完之后调用notify\_one或notify\_all唤醒一个或者所有等待线程。 条件变量提供了两类操作：wait和notify。这两类操作构成了多线程同步的基础。

条件变量提供了两类操作：wait和notify。这两类操作构成了多线程同步的基础。

#### 成员函数

**wait函数**

**函数原型**

```cpp
void wait (unique_lock<mutex>& lck);
template <class Predicate>
    void wait (unique_lock<mutex>& lck, Predicate pred);
```

包含两种重载，第一种只包含unique\_lock对象，另外一个Predicate 对象（等待条件），这里必须使用 unique\_lock，因为wait函数的工作原理：

* 当前线程调用wait()后将被阻塞并且函数会解锁互斥量，直到另外某个线程调用notify\_one或者 notify\_all唤醒当前线程；一旦当前线程获得通知(notify)，wait()函数也是自动调用lock()，同理不 能使用lock\_guard对象。
* 如果wait没有第二个参数，第一次调用默认条件不成立，直接解锁互斥量并阻塞到本行，直到某一 个线程调用notify\_one或notify\_all为止，被唤醒后，wait重新尝试获取互斥量，如果得不到，线程 会卡在这里，直到获取到互斥量，然后无条件地继续进行后面的操作。
* 如果wait包含第二个参数，如果第二个参数不满足，那么wait将解锁互斥量并堵塞到本行，直到某 一个线程调用notify\_one或notify\_all为止，被唤醒后，wait重新尝试获取互斥量，如果得不到，线程会卡在这里，直到获取到互斥量，然后继续判断第二个参数，如果表达式为false，wait对互斥 量解锁，然后休眠，如果为true，则进行后面的操作。

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

```cpp
void notify_one() noexcept;
```

解锁正在等待当前条件的线程中的一个，如果没有线程在等待，则函数不执行任何操作，如果正在等待 的线程多余一个，则唤醒的线程是不确定的。

**notify\_all函数**

函数原型：

```cpp
void notify_all() noexcept;
```

解锁正在等待当前条件的所有线程，如果没有正在等待的线程，则函数不执行任何操作。

#### 范例

使用条件变量实现一个同步队列，同步队列作为一个线程安全的数据共享区，经常用于线程之间数据读 取。 代码范例：同步队列的实现1-3-condition-sync-queue sync\_queue.h

```cpp
//同步队列的实现1-3-condition-sync-queue
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

main.cpp

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

这里需要注意的是，wait函数中会释放mutex，而lock\_guard这时还拥有mutex，它只会在出了作用域 之后才会释放mutex，所以这时它并不会释放，但执行wait时会提取释放mutex。 从语义上看这里使用lock\_guard会产生矛盾，但是实际上并不会出问题，因为wait提前释放锁之后会处 于等待状态，在被notify\_one或者notify\_all唤醒后会先获取mutex，这相当于lock\_guard的mutex在 释放之后又获取到了，因此，在出了作用域之后lock\_guard自动释放mutex不会有问题。 这里应该用unique\_lock，因为unique\_lock不像lock\_guard一样只能在析构时才释放锁，它可以随时释 放锁，因此在wait时让unique\_lock释放锁从语义上更加准确。

使用unique\_lock和condition\_variable\_variable改写1-3-condition-sync-queue，改写为用等待一个判 断式的方法来实现一个简单的队列。 范例：1-3-condition-sync-queue2

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

### 原子变量

具体参考：http://www.cplusplus.com/reference/atomic/atomic/ 范例：1-4-atomic

```cpp
// atomic::load/store example
#include <iostream> // std::cout
#include <atomic> // std::atomic, std::memory_order_relaxed
#include <thread> // std::thread
//std::atomic<int> foo = 0;//错误初始化
std::atomic<int> foo(0); // 准确初始化
void set_foo(int x)
{
    foo.store(x,std::memory_order_relaxed); // set value atomically
}
void print_foo()
{
    int x;
    do {
        x = foo.load(std::memory_order_relaxed); // get value atomically
    } while (x==0);
    std::cout << "foo: " << x << '\n';
}
int main ()
{
    std::thread first (print_foo);
    std::thread second (set_foo,10);
    first.join();
    second.join();
    std::cout << "main finish\n";
    return 0;
}

```

### call\_once和once\_flag使用

具体：https://www.apiref.com/cpp-zh/cpp/thread/call\_once.html 在多线程中，有一种场景是某个任务只需要执行一次，可以用C++11中的std::call\_once函数配合 std::once\_flag来实现。多个线程同时调用某个函数，std::call\_once可以保证多个线程对该函数只调用一 次。

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

### 异步操作

* std::future
* std::aysnc
* std::promise
* std::packaged\_task

参考C++官方手册的范例。

#### std::future

std::future期待一个返回，从一个异步调用的角度来说，future更像是执行函数的返回值，C++标准库 使用std::future为一次性事件建模，如果一个事件需要等待特定的一次性事件，那么这线程可以获取一 个future对象来代表这个事件。 异步调用往往不知道何时返回，但是如果异步调用的过程需要同步，或者说后一个异步调用需要使用前 一个异步调用的结果。这个时候就要用到future。 线程可以周期性的在这个future上等待一小段时间，检查future是否已经ready，如果没有，该线程可以 先去做另一个任务，一旦future就绪，该future就无法复位（无法再次使用这个future等待这个事 件），所以future代表的是一次性事件。

**future的类型**

在库的头文件中声明了两种future，唯一future（std::future）和共享future（std::shared\_future）这 两个是参照std::unique\_ptr和std::shared\_ptr设立的，前者的实例是仅有的一个指向其关联事件的实 例，而后者可以有多个实例指向同一个关联事件，当事件就绪时，所有指向同一事件的 std::shared\_future实例会变成就绪。

**future的使用**

std::future是一个模板，例如std::future，模板参数就是期待返回的类型，虽然future被用于线程间通 信，但其本身却并不提供同步访问，热门必须通过互斥元或其他同步机制来保护访问。 future使用的时机是当你不需要立刻得到一个结果的时候，你可以开启一个线程帮你去做一项任务，并 期待这个任务的返回，但是std::thread并没有提供这样的机制，这就需要用到std::async和std::future （都在头文件中声明） std::async返回一个std::future对象，而不是给你一个确定的值（所以当你不需要立刻使用此值的时候才 需要用到这个机制）。当你需要使用这个值的时候，对future使用get()，线程就会阻塞直到future就 绪，然后返回该值。

```cpp
//1-6-future
#include <iostream>
#include <future>
#include <thread>
using namespace std;
int find_result_to_add()
{
    // std::this_thread::sleep_for(std::chrono::seconds(5)); // 用来测试异步延迟的影
    响
        return 1 + 1;
}
int find_result_to_add2(int a, int b)
{
    // std::this_thread::sleep_for(std::chrono::seconds(5)); // 用来测试异步延迟的影
    响
        return a + b;
}
void do_other_things()
{
    std::cout << "Hello World" << std::endl;
    // std::this_thread::sleep_for(std::chrono::seconds(5));
}
int main()
{
    // std::future<int> result = std::async(find_result_to_add);
    std::future<decltype (find_result_to_add())> result =
        std::async(find_result_to_add);
    do_other_things();
    std::cout << "result: " << result.get() << std::endl; // 延迟是否有影响？
    // std::future<decltype (find_result_to_add2(int, int))> result2 =
    std::async(find_result_to_add2, 10, 20); //错误
    std::future<decltype (find_result_to_add2(0, 0))> result2 =
        std::async(find_result_to_add2, 10, 20);
    std::cout << "result2: " << result2.get() << std::endl; // 延迟是否有影响？
    std::cout << "main finish" << endl;
    return 0;
}
```

跟thread类似，async允许你通过将额外的参数添加到调用中，来将附加参数传递给函数。如果传入的 函数指针是某个类的成员函数，则还需要将类对象指针传入（直接传入，传入指针，或者是std::ref封 装）。 默认情况下，std::async是否启动一个新线程，或者在等待future时，任务是否同步运行都取决于你给的 参数。这个参数为std::launch类型

* std::launch::defered表明该函数会被延迟调用，直到在future上调用get()或者wait()为止
* std::launch::async，表明函数会在自己创建的线程上运行
* std::launch::any = std::launch::defered |std::launch::async
* std::launch::sync = std::launch::defered

```cpp
enum class launch
{
    async,deferred,sync=deferred,any=async|deferred
};

```

PS：默认选项参数被设置为std::launch::any。如果函数被延迟运行可能永远都不会运行。

#### std::packaged\_task

如果说std::async和std::feature还是分开看的关系的话，那么std::packaged\_task就是将任务和feature 绑定在一起的模板，是一种封装对任务的封装。 `The class template std::packaged_task wraps any Callable target (function, lambda expression, bind expression, or another function object) so that it can be invoked asynchronously. Its return value or exception thrown is stored in a shared state which can be accessed through std::future objects.`

可以通过std::packaged\_task对象获取任务相关联的feature，调用get\_future()方法可以获得 std::packaged\_task对象绑定的函数的返回值类型的future。std::packaged\_task的模板参数是函数签 名。 PS：例如int add(int a, intb)的函数签名就是int(int, int)

```cpp
//1-6-package_task
#include <iostream>
#include <future>
using namespace std;
int add(int a, int b)
{
    return a + b;
}
void do_other_things()
{
    std::cout << "Hello World" << std::endl;
}
int main()
{
    std::packaged_task<int(int, int)> task(add);
    do_other_things();
    std::future<int> result = task.get_future();
    task(1, 1); //必须要让任务执行，否则在get()获取future的值时会一直阻塞
    std::cout << result.get() << std::endl;
    return 0;
}

```

#### std::promise

从字面意思上理解promise代表一个承诺。promise比std::packaged\_task抽象层次低。 std::promise提供了一种设置值的方式，它可以在这之后通过相关联的std::future对象进行读取。换种 说法，之前已经说过std::future可以读取一个异步函数的返回值了，那么这个std::promise就提供一种 方式手动让future就绪。

```cpp
//1-6-promise
#include <future>
#include <string>
#include <thread>
#include <iostream>
using namespace std;
void print(std::promise<std::string>& p)
{
    p.set_value("There is the result whitch you want.");
}
void do_some_other_things()
{
    std::cout << "Hello World" << std::endl;
}
int main()
{
    std::promise<std::string> promise;
    std::future<std::string> result = promise.get_future();
    std::thread t(print, std::ref(promise));
    do_some_other_things();
    std::cout << result.get() << std::endl;
    t.join();
    return 0;
}


```

由此可以看出在promise创建好的时候future也已经创建好了 线程在创建promise的同时会获得一个future，然后将promise传递给设置他的线程，当前线程则持有 future，以便随时检查是否可以取值。

#### 总结

future的表现为期望，当前线程持有future时，期望从future获取到想要的结果和返回，可以把future当 做异步函数的返回值。而promise是一个承诺，当线程创建了promise对象后，这个promise对象向线程 承诺他必定会被人设置一个值，和promise相关联的future就是获取其返回的手段。

### Reference

* [参考网页](https://blog.csdn.net/li1615882553/article/details/86179781)
