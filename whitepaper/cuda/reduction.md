---
description: >-
  source from 《Professional CUDA C Programming》By Max Grossman&Ty McKercher.I
  made some changes according to my own understanding.
---

# 🤭 Reduction|并行规约

### 并行归约问题&#x20;

假设要对一个有N个元素的整数数组求和。使用如下的串行代码很容易实现算法：

```c
int sum = 0;
 for (int i = 0; i < N; i++)
   sum += array[i];
```

如果有大量的数据元素会怎么样呢？如何通过并行计算快速求和呢？鉴于加法的结合 律和交换律，数组元素可以以任何顺序求和。所以可以用以下的方法执行并行加法运算：&#x20;

1. 将输入向量划分到更小的数据块中。&#x20;
2. 用一个线程计算一个数据块的部分和。
3. 对每个数据块的部分和再求和得出最终结果。&#x20;

并行加法的一个常用方法是使用迭代成对实现。一个数据块只包含一对元素，并且一 个线程对这两个元素求和产生一个局部结果。然后，这些局部结果在最初的输入向量中就 地保存。这些新值被作为下一次迭代求和的输入值。<mark style="color:red;">因为输入值的数量在每一次迭代后会 减半，当输出向量的长度达到1时，最终的和就已经被计算出来了。</mark>

根据每次迭代后输出元素就地存储的位置，成对的并行求和实现可以被进一步分为以 下两种类型：

* <mark style="color:red;">相邻配对：元素与它们直接相邻的元素配对</mark>
* <mark style="color:red;">交错配对：根据给定的跨度配对元素</mark>

下图所示为相邻配对的实现。在每一步实现中，一个线程对两个相邻元素进行操 作，产生部分和。对于有N个元素的数组，这种实现方式需要N―1次求和，进行log2N 步，时间复杂度为O(log2N)。

<figure><img src="../../.gitbook/assets/图片 (165).png" alt="" width="563"><figcaption></figcaption></figure>

下图所示为交错配对的实现。值得注意的是，在这种实现方法的每一步中，一个线 程的输入是输入数组长度的一半。

<figure><img src="../../.gitbook/assets/图片 (166).png" alt="" width="563"><figcaption></figcaption></figure>

下列的C语言函数是一个交错配对方法的递归实现：

```c
 int recursiveReduce(int *data, int const size) {
   // terminate check
   if (size == 1) return data[0];
   // renew the stride
   int const stride = size / 2;
   // in-place reduction
   for (int i = 0; i < stride; i++) {
      data[i] += data[i + stride];
   }
   // call recursively
   return recursiveReduce(data, stride);
 }
```

<mark style="color:red;">尽管以上代码实现的是加法，但任何满足交换律和结合律的运算都可以代替加法。例 如，通过调用max代替求和运算，就可以计算输入向量中的最大值。其他有效运算的例子 有最小值、平均值和乘积。</mark>&#x20;

在向量中执行满足交换律和结合律的运算，被称为归约问题。并行归约问题是这种运 算的并行执行。并行归约是一种最常见的并行模式，并且是许多并行算法中的一个关键运 算。

### 并行归约中的分化

下图所示的是相邻配对方法的内核实现流程。每个线程将相邻的两个元素相加产生 部分和。 在这个内核里，有两个全局内存数组：一个大数组用来存放整个数组，进行归约；另 一个小数组用来存放每个线程块的部分和。

每个线程块在数组的一部分上独立地执行操 作。循环中迭代一次执行一个归约步骤。归约是在就地完成的，这意味着在每一步，全局 内存里的值都被部分和替代。`__syncthreads`语句可以保证，线程块中的任一线程在进入下 一次迭代之前，在当前迭代里每个线程的所有部分和都被保存在了全局内存中。进入下一 次迭代的所有线程都使用上一步产生的数值。在最后一个循环以后，整个线程块的和被保 存进全局内存中。

<figure><img src="../../.gitbook/assets/图片 (167).png" alt=""><figcaption></figcaption></figure>

```c
 // Neighbored Pair Implementation with divergence
 // n是array数组·的总长度
__global__ void reduceNeighbored (int *g_idata, int *g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;

    // boundary check
    if (idx >= n) return;

    // in-place reduction in global memory
    // 每一个block负责计算blockDim.x个数据
    for (int stride = 1; stride < blockDim.x; stride *= 2)// 控制stride不变，让tid变化
    {
        if ((tid % (2 * stride)) == 0)// tid为奇数的情况会被过滤掉，会发生线程束分化
        {
            // idata为每一个block的首地址
            idata[tid] += idata[tid + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

```

{% hint style="info" %}
1. 第一次看代码可能有些迷糊，可以结合我的注释，自己在画画图
2. 可以从线程束是否分化，时间局部性和空间局部性思考下这段代码可以怎么改进
{% endhint %}

两个相邻元素间的距离被称为跨度，初始化均为1。在每一次归约循环结束后，这个 间隔就被乘以2。在第一次循环结束后，idata（全局数据指针）的偶数元素将会被部分和 替代。在第二次循环结束后，idata的每四个元素将会被新产生的部分和替代。因为线程块 间无法同步，所以每个线程块产生的部分和被复制回了主机，并且在那儿进行串行求和， 如下图所示。

<figure><img src="../../.gitbook/assets/图片 (168).png" alt=""><figcaption></figcaption></figure>

### 改善并行归约的分化&#x20;

测试核函数reduceNeighbored，并注意以下条件表达式：

```c
 if ((tid % (2 * stride)) == 0)
```

因为上述语句只对偶数ID的线程为true，所以这会导致很高的线程束分化。在并行归 约的第一次迭代中，只有ID为偶数的线程执行这个条件语句的主体，但是所有的线程都必 须被调度。在第二次迭代中，只有四分之一的线程是活跃的，但是所有的线程仍然都必须 被调度。通过重新组织每个线程的数组索引来强制ID相邻的线程执行求和操作，线程束分 化就能被归约了。下图展示了这种实现。

<figure><img src="../../.gitbook/assets/图片 (169).png" alt=""><figcaption></figcaption></figure>

修改之后的内核代码如下：

```c
 __global__ void reduceNeighboredLess (int *g_idata, int *g_odata, unsigned int n) {
   // set thread ID
   unsigned int tid = threadIdx.x;
   unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
   // convert global data pointer to the local pointer of this block
   int *idata = g_idata + blockIdx.x*blockDim.x;
   // boundary check
   if(idx >= n) return;
   // in-place reduction in global memory
   for (int stride = 1; stride < blockDim.x; stride *= 2) {
        // convert tid into local array index
      int index = 2 * stride * tid;
      if (index < blockDim.x) {
         idata[index] += idata[index + stride];
      }
      // synchronize within threadblock
      __syncthreads();
   }
   // write result for this block to global mem
   if (tid == 0) g_odata[blockIdx.x] = idata[0];
 }
```

注意内核中的下述语句，它为每个线程设置数组访问索引：

```c
int index = 2 * stride * tid;
```

因为跨度乘以了2，所以下面的语句使用线程块的前半部分来执行求和操作：

```c
if (index < blockDim.x)
```

> 在归约操作中，使用邻近归约方法时，每个线程负责处理相邻的两个元素。通过乘以 2，可以使每个线程在数组中跳过一个元素，从而处理相邻的两个元素。考虑一个简单的示例，假设有一个包含 8 个元素的数组，编号为 0 到 7。使用邻近归约方法时，第一次迭代中，线程 0 负责处理元素 0 和元素 1，线程 1 负责处理元素 2 和元素 3，以此类推。通过将 tid 乘以 2，可以使每个线程的索引跳过一个元素。对于线程 0，tid 为 0，stride 为 1，因此 index = 2 \* stride \* tid = 0，线程 0 处理元素 0 和元素 1。对于线程 1，tid 为 1，stride 为 1，因此 index = 2 \* stride \* tid = 2，线程 1 处理元素 2 和元素 3。同样的方式继续下去，每个线程处理相邻的两个元素，直到完成归约操作。乘以 2 是为了确保每个线程处理相邻的元素，这是邻近归约方法的要求。

对于一个有512个线程的块来说，前8个线程束执行第一轮归约，剩下8个线程束什么 也不做。在第二轮里，前4个线程束执行归约，剩下12个线程束什么也不做。因此，这样 就彻底不存在分化了。在最后五轮中，当每一轮的线程总数小于线程束的大小时，分化就 会出现。

### 交错配对的归约&#x20;

与相邻配对方法相比，交错配对方法颠倒了元素的跨度。初始跨度是线程块大小的一 半，然后在每次迭代中减少一半。在每次循环中，每个线程对两个被当 前跨度隔开的元素进行求和，以产生一个部分和。与相邻规约，交错归约的工作线程没 有变化。但是，每个线程在全局内存中的加载/存储位置是不同的。

<figure><img src="../../.gitbook/assets/图片 (170).png" alt="" width="563"><figcaption></figcaption></figure>

交错归约的内核代码如下所示：

```c
// Interleaved Pair Implementation with less divergence
__global__ void reduceInterleaved (int *g_idata, int *g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;

    // boundary check
    if(idx >= n) return;

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }

        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}
```

注意核函数中的下述语句，两个元素间的跨度被初始化为线程块大小的一半，然后在 每次循环中减少一半：

```c
for (int stride = blockDim.x / 2; stride > 0; sstride >>= 1)
```

下面的语句在第一次迭代时强制线程块中的前半部分线程执行求和操作，第二次迭代 时是线程块的前四分之一，以此类推：

```c
if(tid < stride)
```

### 展开的规约&#x20;

<mark style="color:red;">循环展开是一个尝试通过减少分支出现的频率和循环维护指令来优化循环的技术。</mark>在 循环展开中，循环主体在代码中要多次被编写，而不是只编写一次循环主体再使用另一个 循环来反复执行的。任何的封闭循环可将它的迭代次数减少或完全删除。循环体的复制数 量被称为循环展开因子，迭代次数就变为了原始循环迭代次数除以循环展开因子。在顺序 数组中，当循环的迭代次数在循环执行之前就已经知道时，循环展开是最有效提升性能的 方法。考虑下面的代码：

```c
 for (int i = 0; i < 100; i++) {
   a[i] = b[i] + c[i];
 }
```

如果重复操作一次循环体，迭代次数能减少到原始循环的一半：

```c
 for (int i = 0; i < 100; i += 2) {
   a[i]   = b[i]   + c[i];
   a[i+1] = b[i+1] + c[i+1];
 }
```

<mark style="color:red;">从高级语言层面上来看，循环展开使性能提高的原因可能不是显而易见的。这种提升 来自于编译器执行循环展开时低级指令的改进和优化。例如，在前面循环展开的例子中， 条件i<100只检查了50次，而在原来的循环中则检查了100次。另外，因为在每个循环中每 个语句的读和写都是独立的，所以CPU可以同时发出内存操作。</mark>&#x20;

在CUDA中，循环展开的意义非常重大。我们的目标仍然是相同的：通过减少指令消 耗和增加更多的独立调度指令来提高性能。因此，更多的并发操作被添加到流水线上，以 产生更高的指令和内存带宽。这为线程束调度器提供更多符合条件的线程束，它们可以帮 助隐藏指令或内存延迟。

### 展开的归约&#x20;

你可能会注意到，在reduceInterleaved核函数中每个线程块只处理一部分数据，这些 数据可以被认为是一个数据块。如果用一个线程块手动展开两个数据块的处理，会怎么 样？以下的核函数是reduceInterleaved核函数的修正版：<mark style="color:red;">每个线程块汇总了来自两个数据 块的数据。这是一个循环分区的例子，每个线程作用于多个数据 块，并处理每个数据块的一个元素：</mark>

```c
 __global__ void reduceUnrolling2 (int *g_idata, int *g_odata, unsigned int n) {
   // set thread ID
   unsigned int tid = threadIdx.x;
   unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
   // convert global data pointer to the local pointer of this block
   // 偶数号block的首地址
   int *idata = g_idata + blockIdx.x * blockDim.x * 2;
   // unrolling 2 data blocks
   if (idx + blockDim.x < n) g_idata[idx] += g_idata[idx + blockDim.x];
   __syncthreads();
   // in-place reduction in global memory
   for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
         idata[tid] += idata[tid + stride];
      } 
     // synchronize within threadblock
      __syncthreads();
   }
   // write result for this block to global mem
   if (tid == 0) g_odata[blockIdx.x] = idata[0];
 }
```

注意要在核函数的开头添加的下述语句。在这里，每个线程都添加一个来自于相邻数 据块的元素。从概念上来讲，可以把它作为归约循环的一个迭代，此循环可在数据块间归 约：

```c
if (idx + blockDim.x <n) g_idata[idx] += gidata[idx+blockDim.x];
```

如下所示，全局数组索引被相应地调整，因为只需要一半的线程块来处理相同的数据 集。请注意，这也意味着对于相同大小的数据集，向设备显示的线程束和线程块级别的并 行性更低。

```c
 unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
 int *idata = g_idata + blockIdx.x * blockDim.x * 2;
```

<figure><img src="../../.gitbook/assets/图片 (171).png" alt=""><figcaption></figcaption></figure>

{% hint style="info" %}
因为现在每个线程块处理两个数据块，所以需要调整内核的执行配置，将网格大小减 小至一半
{% endhint %}

### 展开线程规约(Reducing with Unrolled Warps)

`__syncthreads`是用于块内同步的。在归约核函数中，它用来确保在线程进入下一轮之 前，每一轮中所有线程已经将局部结果写入全局内存中了。&#x20;

然而，要细想一下只剩下32个或更少线程（即一个线程束）的情况。因为线程束的执 行是SIMT（单指令多线程）的，每条指令之后有隐式的线程束内同步过程。因此，归约 循环的最后6个迭代可以用下述语句来展开：

```c
 if (tid < 32) {
   volatile int *vmem = idata;
   vmem[tid] += vmem[tid + 32];
   vmem[tid] += vmem[tid + 16];
   vmem[tid] += vmem[tid +  8];
   vmem[tid] += vmem[tid +  4];
   vmem[tid] += vmem[tid +  2];
   vmem[tid] += vmem[tid +  1];
 }
```

注意变量vmem是和volatile修饰符一起被声明的，它告诉编译器每次赋值时必须将 vmem\[tid]的值存回全局内存中。如果省略了volatile修饰符，这段代码将不能正常工作， 因为编译器或缓存可能对全局或共享内存优化读写。如果位于全局或共享内存中的变量有 volatile修饰符，编译器会假定其值可以被其他线程在任何时间修改或使用。因此，任何引用volatile修饰符的变量强制直接读或写内存，而不是简单地读写缓存或寄存器。

{% hint style="info" %}
如果省略了`volatile`修饰符，编译器可能会对共享变量的读写操作进行优化，例如将其缓存到寄存器或者CPU缓存中，而不是每次都直接读写全局内存。这种优化可能会导致以下问题：

1. **可见性问题：** 在多线程环境下，如果一个线程修改了共享变量的值，但是这个修改没有立即写回到全局内存中，其他线程就无法感知到这个修改，导致了可见性问题。
2. **数据不一致性：** 如果多个线程都缓存了共享变量的值，但是它们各自的缓存值可能与全局内存中的值不一致，这会导致数据不一致性问题。
3. **竞态条件：** 缺乏同步措施的情况下，多个线程对共享变量进行读写操作可能导致竞态条件的发生，从而产生不确定的结果。

通过使用`volatile`修饰符，可以告诉编译器不要对这些变量的读写操作进行优化，确保每次操作都直接反映到全局内存中，从而避免了上述问题的发生。
{% endhint %}

基于reduceUnrolling8，线程束的展开可以添加到归约核函数中，如下所示：

```c
__global__ void reduceUnrollWarps8 (int *g_idata, int *g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    // unrolling 8
    if (idx + 7 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int b1 = g_idata[idx + 4 * blockDim.x];
        int b2 = g_idata[idx + 5 * blockDim.x];
        int b3 = g_idata[idx + 6 * blockDim.x];
        int b4 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }

    __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1)// 注意此处的stride循环条件发生了变化(相较于上面的函数)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }
    
    // unrolling warp
    if (tid < 32)
    {
        volatile int *vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid +  8];
        vmem[tid] += vmem[tid +  4];
        vmem[tid] += vmem[tid +  2];
        vmem[tid] += vmem[tid +  1];
    }
    /*
    上面这段代码可以看成
    for (int stride = 32; stride > 0; stride >>= 1)// 注意此处的stride循环条件发生了变化(相较于上面的函数)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }
    实际上就是把上面的循环展开了
    */
    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}
```

{% hint style="info" %}
因为在这个实现中，每个线程处理8个数据块，调用这个内核的同时它的网格尺寸减 小到1/8
{% endhint %}

### 完全展开的归约&#x20;

如果编译时已知一个循环中的迭代次数，就可以把循环完全展开。因为在Fermi或 Kepler架构中，每个块的最大线程数都是1024，并且在这些归约核函数中 循环迭代次数是基于一个线程块维度的，所以完全展开归约循环是完全可以的：

```c
__global__ void reduceCompleteUnrollWarps8 (int *g_idata, int *g_odata,
        unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    // unrolling 8
    if (idx + 7 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int b1 = g_idata[idx + 4 * blockDim.x];
        int b2 = g_idata[idx + 5 * blockDim.x];
        int b3 = g_idata[idx + 6 * blockDim.x];
        int b4 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }

    __syncthreads();

    // in-place reduction and complete unroll
    if (blockDim.x >= 1024 && tid < 512) idata[tid] += idata[tid + 512];

    __syncthreads();

    if (blockDim.x >= 512 && tid < 256) idata[tid] += idata[tid + 256];

    __syncthreads();

    if (blockDim.x >= 256 && tid < 128) idata[tid] += idata[tid + 128];

    __syncthreads();

    if (blockDim.x >= 128 && tid < 64) idata[tid] += idata[tid + 64];

    __syncthreads();

    // unrolling warp
    if (tid < 32)
    {
        volatile int *vsmem = idata;
        vsmem[tid] += vsmem[tid + 32]; // tid = 0,1,2,3....
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid +  8];
        vsmem[tid] += vsmem[tid +  4];
        vsmem[tid] += vsmem[tid +  2];
        vsmem[tid] += vsmem[tid +  1];
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}
```

### 模板函数的归约&#x20;

虽然可以手动展开循环，但是使用模板函数有助于进一步减少分支消耗。在设备函数 上CUDA支持模板参数。如下所示，可以指定块的大小作为模板函数的参数：

```c
template <unsigned int iBlockSize>
__global__ void reduceCompleteUnroll(int *g_idata, int *g_odata,
                                     unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    // unrolling 8
    if (idx + 7 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int b1 = g_idata[idx + 4 * blockDim.x];
        int b2 = g_idata[idx + 5 * blockDim.x];
        int b3 = g_idata[idx + 6 * blockDim.x];
        int b4 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }

    __syncthreads();

    // in-place reduction and complete unroll
    if (iBlockSize >= 1024 && tid < 512) idata[tid] += idata[tid + 512];

    __syncthreads();

    if (iBlockSize >= 512 && tid < 256)  idata[tid] += idata[tid + 256];

    __syncthreads();

    if (iBlockSize >= 256 && tid < 128)  idata[tid] += idata[tid + 128];

    __syncthreads();

    if (iBlockSize >= 128 && tid < 64)   idata[tid] += idata[tid + 64];

    __syncthreads();

    // unrolling warp
    if (tid < 32)
    {
        volatile int *vsmem = idata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid +  8];
        vsmem[tid] += vsmem[tid +  4];
        vsmem[tid] += vsmem[tid +  2];
        vsmem[tid] += vsmem[tid +  1];
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}
```

相比reduceCompleteUnrollWarps8，唯一的区别是使用了模板参数替换了块大小。检查 块大小的if语句将在编译时被评估，如果这一条件为false，那么编译时它将会被删除，使 得内循环更有效率。例如，在线程块大小为256的情况下调用这个核函数，下述语句将永 远是false：

```c
iBlockSize>=1024 && tid < 512
```

编译器会自动从执行内核中移除它。 该核函数一定要在switch-case结构中被调用。这允许编译器为特定的线程块大小自动 优化代码，但这也意味着它只对在特定块大小下启动reduceCompleteUnroll有效：

```c
switch (blocksize) {
      case 1024:
         reduceCompleteUnroll<1024><<<grid.x/8, block>>>(d_idata, d_odata, size);
         break;
      case 512:
         reduceCompleteUnroll<512><<<grid.x/8, block>>>(d_idata, d_odata, size);
         break;
      case 256:
         reduceCompleteUnroll<256><<<grid.x/8, block>>>(d_idata, d_odata, size);
         break;
      case 128:
         reduceCompleteUnroll<128><<<grid.x/8, block>>>(d_idata, d_odata, size);
         break;
      case 64:
         reduceCompleteUnroll<64><<<grid.x/8, block>>>(d_idata, d_odata, size); 
         break;
   } 
```

下表概括了本节提到的所有并行归约实现的效果

<figure><img src="../../.gitbook/assets/图片 (172).png" alt=""><figcaption></figcaption></figure>

下表总结了所有核函数加载/存储数据的效率

<figure><img src="../../.gitbook/assets/图片 (173).png" alt=""><figcaption></figcaption></figure>

完整代码如下：

```c
#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

/*
 * This code implements the interleaved and neighbor-paired approaches to
 * parallel reduction in CUDA. For this example, the sum operation is used. A
 * variety of optimizations on parallel reduction aimed at reducing divergence
 * are also demonstrated, such as unrolling.
 */

// Recursive Implementation of Interleaved Pair Approach
int recursiveReduce(int *data, int const size)
{
    // terminate check
    if (size == 1) return data[0];

    // renew the stride
    int const stride = size / 2;

    // in-place reduction
    for (int i = 0; i < stride; i++)
    {
        data[i] += data[i + stride];
    }

    // call recursively
    return recursiveReduce(data, stride);
}

// Neighbored Pair Implementation with divergence
__global__ void reduceNeighbored (int *g_idata, int *g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;

    // boundary check
    if (idx >= n) return;

    // in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *= 2)// 控制stride不变，让tid变化
    {
        if ((tid % (2 * stride)) == 0) // tid为奇数的情况会被过滤掉，会发生线程束分化
        {
            idata[tid] += idata[tid + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

// Neighbored Pair Implementation with less divergence
__global__ void reduceNeighboredLess (int *g_idata, int *g_odata,
                                      unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;

    // boundary check
    if(idx >= n) return;

    // in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        // convert tid into local array index
        int index = 2 * stride * tid;

        if (index < blockDim.x)
        {
            idata[index] += idata[index + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

// Interleaved Pair Implementation with less divergence
__global__ void reduceInterleaved (int *g_idata, int *g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;

    // boundary check
    if(idx >= n) return;

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }

        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceUnrolling2 (int *g_idata, int *g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 2;

    // unrolling 2
    if (idx + blockDim.x < n) g_idata[idx] += g_idata[idx + blockDim.x];

    __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];// idata是偶数
        }

        // synchronize within threadblock
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceUnrolling4 (int *g_idata, int *g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 4;

    // unrolling 4
    if (idx + 3 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4;
    }

    __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceUnrolling8 (int *g_idata, int *g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    // unrolling 8
    if (idx + 7 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int b1 = g_idata[idx + 4 * blockDim.x];
        int b2 = g_idata[idx + 5 * blockDim.x];
        int b3 = g_idata[idx + 6 * blockDim.x];
        int b4 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }

    __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceUnrollWarps8 (int *g_idata, int *g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    // unrolling 8
    if (idx + 7 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int b1 = g_idata[idx + 4 * blockDim.x];
        int b2 = g_idata[idx + 5 * blockDim.x];
        int b3 = g_idata[idx + 6 * blockDim.x];
        int b4 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }

    __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1)// 注意此处的stride循环条件发生了变化(相较于上面的函数)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }
    
    // unrolling warp
    if (tid < 32)
    {
        volatile int *vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid +  8];
        vmem[tid] += vmem[tid +  4];
        vmem[tid] += vmem[tid +  2];
        vmem[tid] += vmem[tid +  1];
    }
    /*
    上面这段代码可以看成
    for (int stride = 32; stride > 0; stride >>= 1)// 注意此处的stride循环条件发生了变化(相较于上面的函数)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }
    实际上就是把上面的循环展开了
    */
    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceCompleteUnrollWarps8 (int *g_idata, int *g_odata,
        unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    // unrolling 8
    if (idx + 7 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int b1 = g_idata[idx + 4 * blockDim.x];
        int b2 = g_idata[idx + 5 * blockDim.x];
        int b3 = g_idata[idx + 6 * blockDim.x];
        int b4 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }

    __syncthreads();

    // in-place reduction and complete unroll
    if (blockDim.x >= 1024 && tid < 512) idata[tid] += idata[tid + 512];

    __syncthreads();

    if (blockDim.x >= 512 && tid < 256) idata[tid] += idata[tid + 256];

    __syncthreads();

    if (blockDim.x >= 256 && tid < 128) idata[tid] += idata[tid + 128];

    __syncthreads();

    if (blockDim.x >= 128 && tid < 64) idata[tid] += idata[tid + 64];

    __syncthreads();

    // unrolling warp
    if (tid < 32)
    {
        volatile int *vsmem = idata;
        vsmem[tid] += vsmem[tid + 32]; // tid = 0,1,2,3....
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid +  8];
        vsmem[tid] += vsmem[tid +  4];
        vsmem[tid] += vsmem[tid +  2];
        vsmem[tid] += vsmem[tid +  1];
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

template <unsigned int iBlockSize>
__global__ void reduceCompleteUnroll(int *g_idata, int *g_odata,
                                     unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    // unrolling 8
    if (idx + 7 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int b1 = g_idata[idx + 4 * blockDim.x];
        int b2 = g_idata[idx + 5 * blockDim.x];
        int b3 = g_idata[idx + 6 * blockDim.x];
        int b4 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }

    __syncthreads();

    // in-place reduction and complete unroll
    if (iBlockSize >= 1024 && tid < 512) idata[tid] += idata[tid + 512];

    __syncthreads();

    if (iBlockSize >= 512 && tid < 256)  idata[tid] += idata[tid + 256];

    __syncthreads();

    if (iBlockSize >= 256 && tid < 128)  idata[tid] += idata[tid + 128];

    __syncthreads();

    if (iBlockSize >= 128 && tid < 64)   idata[tid] += idata[tid + 64];

    __syncthreads();

    // unrolling warp
    if (tid < 32)
    {
        volatile int *vsmem = idata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid +  8];
        vsmem[tid] += vsmem[tid +  4];
        vsmem[tid] += vsmem[tid +  2];
        vsmem[tid] += vsmem[tid +  1];
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceUnrollWarps (int *g_idata, int *g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 2;

    // unrolling 2
    if (idx + blockDim.x < n) g_idata[idx] += g_idata[idx + blockDim.x];

    __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }

    // unrolling last warp
    if (tid < 32)
    {
        volatile int *vsmem = idata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid +  8];
        vsmem[tid] += vsmem[tid +  4];
        vsmem[tid] += vsmem[tid +  2];
        vsmem[tid] += vsmem[tid +  1];
    }

    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

int main(int argc, char **argv)
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s starting reduction at ", argv[0]);
    printf("device %d: %s ", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    bool bResult = false;

    // initialization
    int size = 1 << 24; // total number of elements to reduce
    printf("    with array size %d  ", size);

    // execution configuration
    int blocksize = 512;   // initial block size

    if(argc > 1)
    {
        blocksize = atoi(argv[1]);   // block size from command line argument
    }

    dim3 block (blocksize, 1);
    dim3 grid  ((size + block.x - 1) / block.x, 1);
    printf("grid %d block %d\n", grid.x, block.x);

    // allocate host memory
    size_t bytes = size * sizeof(int);
    int *h_idata = (int *) malloc(bytes);
    int *h_odata = (int *) malloc(grid.x * sizeof(int));
    int *tmp     = (int *) malloc(bytes);

    // initialize the array
    for (int i = 0; i < size; i++)
    {
        // mask off high 2 bytes to force max number to 255
        h_idata[i] = (int)( rand() & 0xFF );
    }

    memcpy (tmp, h_idata, bytes);

    double iStart, iElaps;
    int gpu_sum = 0;

    // allocate device memory
    int *d_idata = NULL;
    int *d_odata = NULL;
    CHECK(cudaMalloc((void **) &d_idata, bytes));
    CHECK(cudaMalloc((void **) &d_odata, grid.x * sizeof(int)));

    // cpu reduction
    iStart = seconds();
    int cpu_sum = recursiveReduce (tmp, size);
    iElaps = seconds() - iStart;
    printf("cpu reduce      elapsed %f sec cpu_sum: %d\n", iElaps, cpu_sum);

    // kernel 1: reduceNeighbored
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceNeighbored<<<grid, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];

    printf("gpu Neighbored  elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x, block.x);

    // kernel 2: reduceNeighbored with less divergence
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceNeighboredLess<<<grid, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];

    printf("gpu Neighbored2 elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x, block.x);

    // kernel 3: reduceInterleaved
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceInterleaved<<<grid, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];

    printf("gpu Interleaved elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x, block.x);

    // kernel 4: reduceUnrolling2
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceUnrolling2<<<grid.x / 2, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 2 * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x / 2; i++) gpu_sum += h_odata[i];

    printf("gpu Unrolling2  elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x / 2, block.x);

    // kernel 5: reduceUnrolling4
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceUnrolling4<<<grid.x / 4, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 4 * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x / 4; i++) gpu_sum += h_odata[i];

    printf("gpu Unrolling4  elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x / 4, block.x);

    // kernel 6: reduceUnrolling8
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceUnrolling8<<<grid.x / 8, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x / 8; i++) gpu_sum += h_odata[i];

    printf("gpu Unrolling8  elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x / 8, block.x);

    for (int i = 0; i < grid.x / 16; i++) gpu_sum += h_odata[i];

    // kernel 8: reduceUnrollWarps8
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceUnrollWarps8<<<grid.x / 8, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x / 8; i++) gpu_sum += h_odata[i];

    printf("gpu UnrollWarp8 elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x / 8, block.x);


    // kernel 9: reduceCompleteUnrollWarsp8
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceCompleteUnrollWarps8<<<grid.x / 8, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x / 8; i++) gpu_sum += h_odata[i];

    printf("gpu Cmptnroll8  elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x / 8, block.x);

    // kernel 9: reduceCompleteUnroll
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();

    switch (blocksize)
    {
    case 1024:
        reduceCompleteUnroll<1024><<<grid.x / 8, block>>>(d_idata, d_odata,
                size);
        break;

    case 512:
        reduceCompleteUnroll<512><<<grid.x / 8, block>>>(d_idata, d_odata,
                size);
        break;

    case 256:
        reduceCompleteUnroll<256><<<grid.x / 8, block>>>(d_idata, d_odata,
                size);
        break;

    case 128:
        reduceCompleteUnroll<128><<<grid.x / 8, block>>>(d_idata, d_odata,
                size);
        break;

    case 64:
        reduceCompleteUnroll<64><<<grid.x / 8, block>>>(d_idata, d_odata, size);
        break;
    }

    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(int),
                     cudaMemcpyDeviceToHost));

    gpu_sum = 0;

    for (int i = 0; i < grid.x / 8; i++) gpu_sum += h_odata[i];

    printf("gpu Cmptnroll   elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x / 8, block.x);

    // free host memory
    free(h_idata);
    free(h_odata);

    // free device memory
    CHECK(cudaFree(d_idata));
    CHECK(cudaFree(d_odata));

    // reset device
    CHECK(cudaDeviceReset());

    // check the results
    bResult = (gpu_sum == cpu_sum);

    if(!bResult) printf("Test failed!\n");

    return EXIT_SUCCESS;
}
```
