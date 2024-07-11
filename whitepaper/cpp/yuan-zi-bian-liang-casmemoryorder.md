# 🍆 原子变量|CAS|memory\_order

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

<figure><img src="../../.gitbook/assets/1 ZLd19SxR4tdPYK4fs-KeeQ (1).png" alt=""><figcaption></figcaption></figure>
