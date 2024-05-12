# 🥒 原子变量|异步操作

### 原子变量

具体参考：http://www.cplusplus.com/reference/atomic/atomic/&#x20;

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
