# 😷 CUDA流和事件

### 流和事件的概述

CUDA流是一系列异步的CUDA操作，这些操作按照主机代码确定的顺序在设备上执行。流能封装这些操作，保持操作的顺序，允许操作在流中排队，并使它们在先前的所有 操作之后执行，并且可以查询排队操作的状态。这些操作包括在主机与设备间进行数据传 输，内核启动以及大多数由主机发起但由设备处理的其他命令。流中操作的执行相对于主 机总是异步的。`CUDA runtime`决定何时可以在设备上执行操作。<mark style="color:red;">我们的任务是使用CUDA 的API来确保一个异步操作在运行结果被使用之前可以完成。在同一个CUDA流中的操作 有严格的执行顺序，而在不同CUDA流中的操作在执行顺序上不受限制。</mark>使用多个流同时 启动多个内核，可以实现网格级并发。

因为所有在CUDA流中排队的操作都是异步的，所以在主机与设备系统中可以重叠执 行其他操作。在同一时间内将流中排队的操作与其他有用的操作一起执行，可以隐藏执行 那些操作的开销。

**CUDA编程的一个典型模式：**

1. 将输入数据从主机移到设备上。&#x20;
2. 在设备上执行一个内核。&#x20;
3. 将结果从设备移回主机中。

{% hint style="info" %}
从软件的角度来看，CUDA操作在不同的流中并发运行；而从硬件上来看，不一定总 是如此。根据PCIe总线争用或每个SM资源的可用性，完成不同的CUDA流仍然需要互相等待。
{% endhint %}

### CUDA流

所有的CUDA操作都在一个流中显式或隐式地运行。流分为 两种类型：

* 隐式声明的流（空流）

如果没有显式地指定一个流，那么内核启动和数据传输将默认使用空流。

* 显式声明的流（非空流）&#x20;

非空流可以被显式地创建和管理。<mark style="color:red;">如果想要重叠不同的CUDA操作，必须使用非空流。</mark>

<mark style="color:red;">基于流的异步的内核启动和数据传输支持以下类型的粗粒度并发：</mark>&#x20;

* 重叠主机计算和设备计算&#x20;
* 重叠主机计算和主机与设备间的数据传输&#x20;
* 重叠主机与设备间的数据传输和设备计算&#x20;
* 并发设备计算 &#x20;

思考下面使用默认流的代码：

```c
cudaMemcpy(..., cudaMemcpyHostToDevice);
kernel<<<grid, block>>>(...);
cudaMemcpy(..., cudaMemcpyDeviceToHost);
```

要想理解一个CUDA程序，应该从设备和主机两个角度去考虑。**从设备的角度来看， 上述代码中所有的3个操作都被发布到默认的流中，并且按发布顺序执行。**设备不知道其 他被执行的主机操作。从主机的角度来看，每个数据传输都是同步的，在等待它们完成时，将强制主机处于空闲状态。内核启动是异步的，所以无论内核是否完成，主机的应用程序 几乎都立即恢复执行。**这种由内核启动导致的异步行为可以用来重叠主机和设备的计算。**

数据传输也可以被异步发布，但是必须显式地设置一个CUDA流来装载它们。`CUDA runtime API`提供了以下`cudaMemcpy`函数的异步版本：

```c
cudaError_t cudaMemcpyAsync(void* dst, constvoid* src, size_t count, cudaMemcpyKind kind, cudaStream t stream =0)
```

请注意附加的流标识符作为第五个参数。默认情况下，流标识符被设置为默认流。这 个函数与主机是异步的，所以调用后，控制权将立即返回到主机。将数据传输操作和非空流进行关联是很容易的，但是首先需要使用如下代码创建一个非空流：

```c
cudaError_t cudaStreamCreate(cudaStream_t* pStream)
```

cudaStreamCreate创建了一个可以显式管理的非空流。之后，返回到pStream中的流就 可以被当作流参数供cudaMemcpyAsync和其他异步CUDA的API来使用。<mark style="color:red;">在使用异步 CUDA函数时，它们可能会从先前启动的异步操作中返回错误代码。</mark>

当执行异步数据传输时，必须使用固定（或非分页的）主机内存。可以使用`cudaMallocHost`函数或`cudaHostAlloc`函数分配固定内存：

```c
cudaError t cudaMallocHost(void **ptr, sizet size);
cudaError_t cudaHostAlloc(void **pHost, size_t ssize, unsigned int flags);
```

<mark style="color:red;">在主机虚拟内存中进行固定分配，可以确保其在CPU内存中的物理位置在应用程序的整个 生命周期中保持不变。否则，操作系统可以随时自由改变主机虚拟内存的物理位置。在没有固定主机内存的情况下执行一个异步CUDA转移操作，操作系统可能会在物理层面 上移动数组，这样会导致未定义的行为。</mark>

{% hint style="info" %}
* `cudaMallocHost`分配的是页锁定内存，也称为固定内存。
* 页锁定内存不会被分页到磁盘，因此对于GPU访问非常高效。
* 在某些情况下，人们更喜欢直接使用`cudaMallocHost`来分配页锁定内存，因为它更容易使用。
{% endhint %}

在非默认流中启动内核，必须在内核执行配置中提供一个流标识符作为第四个参数：

```c
kernel name<<<grid, block, sharedMemSize, stream>>>(argument list);
```

一个非默认流声明如下：

```c
cudaStream_t stream;
```

非默认流可以使用如下方式进行创建：

```c
cudaStreamCreate (&stream);
```

可以使用如下代码释放流中的资源：

```c
cudaError t cudaStreamDestroy(cudaStream_t stream);
```

在一个流中，当cudaStreamDestroy函数被调用时，如果该流中仍有未完成的工作， cudaStreamDestroy函数将立即返回，当流中所有的工作都已完成时，与流相关的资源将被 自动释放。&#x20;

因为所有的CUDA流操作都是异步的，所以CUDA的API提供了两个函数来检查流中所 有操作是否都已经完成：

```c
cudaError_t cudaStreamSynchronize(cudaStream_(t stream)
cudaError t cudaStreamQuery(cudaStream_t sttream);
```

<mark style="color:red;">**cudaStreamSynchronize强制阻塞主机，直到在给定流中所有的操作都完成了。cudaStreamQuery会检查流中所有操作是否都已经完成，但在它们完成前不会阻塞主机。**</mark><mark style="color:red;">当所有操作都完成时cudaStreamQuery函数会返回cudaSuccess，当一个或多个操作仍在执行或等待执行时返回cudaErrorNotReady。</mark>

来看几个简单的编程示例：

```c
#include <cstdio>

using namespace std;

__global__ void
foo_kernel(int step)
{
    printf("loop: %d\n", step);
}

int main()
{
    int n_loop = 5;

    // 使用默认流
    for (int i = 0; i < n_loop; i++)
        foo_kernel<<< 1, 1, 0, 0 >>>(i);

    cudaDeviceSynchronize();

    return 0;
}
```

<figure><img src="../../.gitbook/assets/图片 (4) (1) (1) (1) (1) (1) (1) (1).png" alt=""><figcaption></figcaption></figure>

```c
#include <cstdio>

using namespace std;

__global__ void
foo_kernel(int step)
{
    printf("loop: %d\n", step);
}

int main()
{
    int n_stream = 5;
    cudaStream_t *ls_stream;
    ls_stream = (cudaStream_t*) new cudaStream_t[n_stream];

    // create multiple streams
    for (int i = 0; i < n_stream; i++)
        cudaStreamCreate(&ls_stream[i]);

    // execute kernels with the CUDA stream each
    for (int i = 0; i < n_stream; i++)
        foo_kernel<<< 1, 1, 0, ls_stream[i] >>>(i);

    // synchronize the host and GPU
    cudaDeviceSynchronize();

    // terminates all the created CUDA streams
    for (int i = 0; i < n_stream; i++)
        cudaStreamDestroy(ls_stream[i]);
    delete [] ls_stream;

    return 0;
}
```

<figure><img src="../../.gitbook/assets/图片 (2) (1) (1) (1) (1) (1) (1) (1).png" alt=""><figcaption></figcaption></figure>

{% hint style="info" %}
<mark style="color:red;">从上图可以看到，数据传输操作虽然分布在不同的流中，但是并没有并发执行。这是由一 个共享资源导致的：PCIe总线。虽然从编程模型的角度来看这些操作是独立的，但是因为它们共享一个相同的硬件资源，所以它们的执行必须是串行的。</mark>
{% endhint %}

### 流同步

CUDA 流通过 cudaStreamSynchronize() 函数提供流级同步。使用该函数可强制主机等待某个流操作的结束。下面的代码展示了在内核执行结束时使用流同步的示例：

```c
#include <cstdio>

using namespace std;

__global__ void
foo_kernel(int step)
{
    printf("loop: %d\n", step);
}

int main()
{
    int n_stream = 5;
    cudaStream_t *ls_stream;
    ls_stream = (cudaStream_t*) new cudaStream_t[n_stream];

    // create multiple streams
    for (int i = 0; i < n_stream; i++)
        cudaStreamCreate(&ls_stream[i]);

    // execute kernels with the CUDA stream each
    for (int i = 0; i < n_stream; i++) {
       foo_kernel<<< 1, 1, 0, ls_stream[i] >>>(i);
       cudaStreamSynchronize(ls_stream[i]);
    }

    // synchronize the host and GPU
    cudaDeviceSynchronize();

    // terminates all the created CUDA streams
    for (int i = 0; i < n_stream; i++)
        cudaStreamDestroy(ls_stream[i]);
    delete [] ls_stream;

    return 0;
}
```

内核操作的并发性将因同步而消失

<figure><img src="../../.gitbook/assets/图片 (3) (1) (1) (1) (1) (1) (1) (1).png" alt=""><figcaption></figcaption></figure>

可以看到，所有内核执行都没有重叠点，尽管它们是以不同的流执行的。利用这一特性，我们可以让主机等待某一流操作返回的结果。

### 默认流的使用

要同时运行多个流，我们应该使用显式创建的流，因为_<mark style="color:red;">**所有流操作都与默认流同步**</mark>_：

> In general, when an operation is issued to the NULL stream, the CUDA context waits on all operations previously issued to all blocking streams before starting that operation. Also, any operations issued to blocking streams will wait on preceding operations in the NULL stream to complete before executing.

<figure><img src="../../.gitbook/assets/图片 (4) (1) (1) (1) (1) (1) (1).png" alt=""><figcaption></figcaption></figure>

相关代码如下(只需要改循环启动核函数那块代码)：

```c
//....省略
// execute kernels with the CUDA stream each
for (int i = 0; i < n_stream; i++)
    if (i == 3)
        foo_kernel<<< 1, 1, 0, 0 >>>(i);
    else
        foo_kernel<<< 1, 1, 0, ls_stream[i] >>>(i);
//....省略
```

因此，我们可以看到，最后一次操作不能与之前的内核执行重叠，而必须等到第四次内核执行结束。

### GPU流水线执行

多数据流的主要优势之一是数据传输与内核执行重叠。通过重叠内核操作和数据传输，我们可以隐藏数据传输开销，提高整体性能。

> 这里的重叠具体说的是：将大的数据块拆分成小块，将多个H2D->Kernel->D2H操作放到多个非默认流中执行。

#### GPU 流水线概念

当我们执行内核函数时，需要将数据从主机传输到 GPU。 然后，再将结果从 GPU 传输回主机。下图显示了在主机和内核之间传输数据的过程：

<figure><img src="../../.gitbook/assets/图片 (1) (1) (1) (1) (1) (1) (1) (1) (1).png" alt=""><figcaption></figcaption></figure>

然而，内核执行基本上是异步的，主机和 GPU 可以同时运行。如果主机和 GPU 之间的数据传输具有相同的特性，我们就可以重叠执行。

<figure><img src="../../.gitbook/assets/图片 (86).png" alt=""><figcaption></figcaption></figure>

从图中我们可以看到，主机和设备之间的数据传输可以与内核执行重叠。那么，这种重叠操作的好处就是缩短了应用程序的执行时间。通过比较两幅图的长度，你就能确定哪种操作的吞吐量更高。

关于 CUDA 数据流，所有 CUDA 操作（数据传输和内核执行）在同一数据流中都是顺序进行的。不过，这些操作可以与不同的数据流同时进行。下图显示了多个数据流中重叠的数据传输和内核操作：

<figure><img src="../../.gitbook/assets/图片 (87).png" alt="" width="563"><figcaption></figcaption></figure>

<mark style="color:red;">要实现这种流水线操作，有三个条件：</mark>

1. 主机内存应被分配为`pinned memory`(cudaMallocHost() 函数和 cudaFreeHost() 函数)。
2. 在主机和 GPU 之间传输数据而不阻塞主机进程(cudaMemcpyAsync() 函数)。
3. 将每个操作放到不同的 CUDA 流中管理，以实现并发操作。

#### 构建流水线执行

现在，让我们构建一个在数据传输和内核执行中进行流水线操作的应用程序。在此应用程序中，我们将使用一个内核函数，通过对数据流进行分片(分成4段)，将两个向量相加，然后输出结果。 每个核函数中将重复加法操作 500 次，以延长内核执行时间。因此，实现的代码如下：

```c
__global__ void
vecAdd_kernel(float *c, const float* a, const float* b)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = 0; i < 500; i++)
        c[idx] = a[idx] + b[idx];
}
```

为了处理每个流的操作，我们将创建一个管理 CUDA 流和 CUDA 操作的类。使用该类，我们可以通过索引来管理 CUDA 流。以下代码显示了该类的基本架构：

```c
class Operator
{
private:
    int index;
    cudaStream_t stream;

public:
    Operator() {
        cudaStreamCreate(&stream);
    }

    ~Operator() {
        cudaStreamDestroy(stream);
    }

    void set_index(int idx) { index = idx; }
    void async_operation(float *h_c, const float *h_a, const float *h_b,
                          float *d_c, float *d_a, float *d_b,
                          const int size, const int bufsize);

}; // Operator

void Operator::async_operation(float *h_c, const float *h_a, const float *h_b,
                          float *d_c, float *d_a, float *d_b,
                          const int size, const int bufsize)
{
    // copy host -> device
    cudaMemcpyAsync(d_a, h_a, bufsize, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, h_b, bufsize, cudaMemcpyHostToDevice, stream);

    // launch cuda kernel
    dim3 dimBlock(256);
    dim3 dimGrid(size / dimBlock.x);
    vecAdd_kernel<<< dimGrid, dimBlock, 0, stream >>>(d_c, d_a, d_b);

    // copy device -> host
    cudaMemcpyAsync(h_c, d_c, bufsize, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    printf("Launched GPU task %d\n", index);
}
```

完整的代码如下：

```c
#include <cstdio>
#include <helper_timer.h>

using namespace std;

__global__ void vecAdd_kernel(float *c, const float* a, const float* b);
void init_buffer(float *data, const int size);

class Operator
{
private:
    int index;
    cudaStream_t stream;

public:
    Operator() {
        cudaStreamCreate(&stream);
    }

    ~Operator() {
        cudaStreamDestroy(stream);
    }

    void set_index(int idx) { index = idx; }
    void async_operation(float *h_c, const float *h_a, const float *h_b,
                          float *d_c, float *d_a, float *d_b,
                          const int size, const int bufsize);

}; // Operator

void Operator::async_operation(float *h_c, const float *h_a, const float *h_b,
                          float *d_c, float *d_a, float *d_b,
                          const int size, const int bufsize)
{
    // copy host -> device
    cudaMemcpyAsync(d_a, h_a, bufsize, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, h_b, bufsize, cudaMemcpyHostToDevice, stream);

    // launch cuda kernel
    dim3 dimBlock(256);
    dim3 dimGrid(size / dimBlock.x);
    vecAdd_kernel<<< dimGrid, dimBlock, 0, stream >>>(d_c, d_a, d_b);

    // copy device -> host
    cudaMemcpyAsync(h_c, d_c, bufsize, cudaMemcpyDeviceToHost, stream);

    printf("Launched GPU task %d\n", index);
}

int main(int argc, char* argv[])
{
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    int size = 1 << 24;
    int bufsize = size * sizeof(float);
    int num_operator = 4;

    if (argc != 1)
        num_operator = atoi(argv[1]);
    
    // allocate host memories
    cudaMallocHost((void**)&h_a, bufsize);
    cudaMallocHost((void**)&h_b, bufsize);
    cudaMallocHost((void**)&h_c, bufsize);

    // initialize host values
    srand(8215);
    init_buffer(h_a, size);
    init_buffer(h_b, size);
    init_buffer(h_c, size);

    // allocate device memories
    cudaMalloc((void**)&d_a, bufsize);
    cudaMalloc((void**)&d_b, bufsize);
    cudaMalloc((void**)&d_c, bufsize);

    // create list of operation elements
    Operator *ls_operator = new Operator[num_operator];

    // initialize & start timer
    StopWatchInterface *timer;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    // execute each operator collesponding data
    for (int i = 0; i < num_operator; i++) {
        int offset = i * size / num_operator;
        ls_operator[i].set_index(i);
        ls_operator[i].async_operation(&h_c[offset], &h_a[offset], &h_b[offset],
                                       &d_c[offset], &d_a[offset], &d_b[offset],
                                       size / num_operator, bufsize / num_operator);
    }

    // synchronize until all the stream operation is finished
    cudaDeviceSynchronize();

    sdkStopTimer(&timer);

    // print out the result
    int print_idx = 256;
    printf("compared a sample result...\n");
    printf("host: %.6f, device: %.6f\n",  h_a[print_idx] + h_b[print_idx], h_c[print_idx]);

    // Compute and print the performance
    float elapsed_time_msed = sdkGetTimerValue(&timer);
    float bandwidth = 3 * bufsize * sizeof(float) / elapsed_time_msed / 1e6;
    printf("Time= %.3f msec, bandwidth= %f GB/s\n", elapsed_time_msed, bandwidth);

    sdkDeleteTimer(&timer);

    // terminate operators
    delete [] ls_operator;

    // terminate device memories
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // terminate host memories
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
    
    return 0;
}

void init_buffer(float *data, const int size)
{
    for (int i = 0; i < size; i++) 
        data[i] = rand() / (float)RAND_MAX;
}

__global__ void
vecAdd_kernel(float *c, const float* a, const float* b)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = 0; i < 500; i++)
        c[idx] = a[idx] + b[idx];
}
```

输出结果如下：

```bash
 Launched GPU task 0
 Launched GPU task 1
 Launched GPU task 2
 Launched GPU task 3
 compared a sample result...
 host: 1.523750, device: 1.523750
 Time= 29.508 msec, bandwidth= 27.291121 GB/s
```

下面的截图显示了四个 CUDA 数据流通过重叠数据传输和内核执行进行的操作：

<figure><img src="../../.gitbook/assets/图片 (88).png" alt=""><figcaption></figcaption></figure>

因此，GPU 可以一直忙到最后一个内核执行完毕，而且我们可以隐藏大部分数据传输。这不仅提高了 GPU 的利用率，还缩短了应用程序总的执行时间。

在内核函数执行之间，我们可以发现，虽然它们属于不同的 CUDA 流，但是存在征用窗口期。这是因为 GPU 调度器首先为第一个请求提供服务。不过，当任务完成后，流式多处理器就为另一个 CUDA 流中的内核提供服务。

在所有的 CUDA 流操作结束后，我们需要同步主机和 GPU设备，以确认 GPU 上的所有 CUDA 操作都已完成。为此，我们在循环之后使用了 `cudaDeviceSynchronize()`。该函数可以在函数调用处同步所有的 GPU 操作。

<mark style="color:red;">对于同步任务，我们可以用下面的代码替换</mark> <mark style="color:red;"></mark><mark style="color:red;">`cudaDeviceSynchronize()`</mark><mark style="color:red;">函数。为此，我们还必须将私有成员</mark> <mark style="color:red;"></mark><mark style="color:red;">`_stream`</mark> <mark style="color:red;"></mark><mark style="color:red;">改为公有</mark>：

```c
 for (int i = 0; i < num_operator; i++) {
    cudaStreamSynchronize(ls_operator[i]._stream);
 }
```

<mark style="color:red;">如果每个流操作完成后立即执行一些特定的CPU操作，那么这种设计可能会导致不必要的同步点，从而影响程序的整体性能。这是因为即使某些流的操作已经完成，CPU上的后续操作可能仍然需要等待其他流的操作完成才能开始。(这个问题可以通过流回调函数的方式比较完美的解决)</mark>

如果在函数`async_operation`中使用 `cudaStreamSynchronize()` 这将无法重叠核函数的执行与数据的传输。

```c
void Operator::async_operation(float *h_c, const float *h_a, const float *h_b,
                          float *d_c, float *d_a, float *d_b,
                          const int size, const int bufsize)
{
    // copy host -> device
    cudaMemcpyAsync(d_a, h_a, bufsize, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, h_b, bufsize, cudaMemcpyHostToDevice, stream);

    // launch cuda kernel
    dim3 dimBlock(256);
    dim3 dimGrid(size / dimBlock.x);
    vecAdd_kernel<<< dimGrid, dimBlock, 0, stream >>>(d_c, d_a, d_b);

    // copy device -> host
    cudaMemcpyAsync(h_c, d_c, bufsize, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    printf("Launched GPU task %d\n", index);
}
```

<figure><img src="../../.gitbook/assets/图片 (89).png" alt=""><figcaption></figcaption></figure>

在这种情况下，测得的执行时间为 41.521 毫秒，比重叠执行时间慢了约 40%。

### 流回调函数

<mark style="color:red;">CUDA 回调函数是由 GPU 上下文执行的可调用主机函数。利用它，程序员可以在 GPU 操作结束之后调用所需的主机操作。</mark>

CUDA 回调函数有一个名为 `CUDART_CB` 的特殊数据类型。使用这种类型，程序员可以指定由哪个 CUDA 流启动该函数、传递 GPU 错误状态并提供用户数据。

为注册回调函数，CUDA 提供了 `cudaStreamAddCallback()`。该函数接受 CUDA 流、CUDA 回调函数及其参数，这样就可以从指定的 CUDA 流调用指定的 CUDA 回调函数，获取用户数据。该函数有四个输入参数，但最后一个是保留参数。因此，我们不使用该参数，其值为 0。

现在，让我们改进代码，使用回调函数并输出单个数据流的性能。首先，将下面这些声明放入到`Operator`类的`private`区域：

```c
private:
    int _index;
    cudaStream_t stream;
    StopWatchInterface *p_timer;

    static void CUDART_CB Callback(cudaStream_t stream, cudaError_t status, void* userData);
    void print_time();
```

`Callback()` 函数将在每个数据流操作完成后调用，而 `print_time()` 函数将使用主机端(`host side`)计时器 `_p_timer` 报告估计的性能。这些函数的实现如下：

```c
void Operator::CUDART_CB Callback(cudaStream_t stream, cudaError_t status, void* userData) {
    Operator* this_ = (Operator*) userData;
    this_->print_time();
}

void Operator::print_time() {
    sdkStopTimer(&p_timer);    // end timer
    float elapsed_time_msed = sdkGetTimerValue(&p_timer);
    printf("stream %2d - elapsed %.3f ms \n", _index, elapsed_time_msed);
}
```

完整的程序如下：

```c
#include <cstdio>
#include <helper_timer.h>

using namespace std;

__global__ void vecAdd_kernel(float *c, const float* a, const float* b);
void init_buffer(float *data, const int size);

class Operator
{
private:
    int _index;
    cudaStream_t stream;
    StopWatchInterface *p_timer;

    static void CUDART_CB Callback(cudaStream_t stream, cudaError_t status, void* userData);
    void print_time();

public:
    Operator() {
        cudaStreamCreate(&stream);
        sdkCreateTimer(&p_timer);
    }

    ~Operator() {
        cudaStreamDestroy(stream);
        sdkDeleteTimer(&p_timer);
    }

    void set_index(int index) { _index = index; }
    void async_operation(float *h_c, const float *h_a, const float *h_b,
                          float *d_c, float *d_a, float *d_b,
                          const int size, const int bufsize);

}; // Operator

void Operator::CUDART_CB Callback(cudaStream_t stream, cudaError_t status, void* userData) {
    Operator* this_ = (Operator*) userData;
    this_->print_time();
}

void Operator::print_time() {
    sdkStopTimer(&p_timer);    // end timer
    float elapsed_time_msed = sdkGetTimerValue(&p_timer);
    printf("stream %2d - elapsed %.3f ms \n", _index, elapsed_time_msed);
}

void Operator::async_operation(float *h_c, const float *h_a, const float *h_b,
                          float *d_c, float *d_a, float *d_b,
                          const int size, const int bufsize)
{
    // start timer
    sdkStartTimer(&p_timer);

    // copy host -> device
    cudaMemcpyAsync(d_a, h_a, bufsize, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, h_b, bufsize, cudaMemcpyHostToDevice, stream);

    // launch cuda kernel
    dim3 dimBlock(256);
    dim3 dimGrid(size / dimBlock.x);
    vecAdd_kernel<<< dimGrid, dimBlock, 0, stream >>>(d_c, d_a, d_b);

    // copy device -> host
    cudaMemcpyAsync(h_c, d_c, bufsize, cudaMemcpyDeviceToHost, stream);

    // register callback function
    cudaStreamAddCallback(stream, Operator::Callback, this, 0);
}

int main(int argc, char* argv[])
{
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    int size = 1 << 24;
    int bufsize = size * sizeof(float);
    int num_operator = 4;

    if (argc != 1)
        num_operator = atoi(argv[1]);

    // initialize timer
    StopWatchInterface *timer;
    sdkCreateTimer(&timer);
    
    // allocate host memories
    cudaMallocHost((void**)&h_a, bufsize);
    cudaMallocHost((void**)&h_b, bufsize);
    cudaMallocHost((void**)&h_c, bufsize);

    // initialize host values
    srand(2019);
    init_buffer(h_a, size);
    init_buffer(h_b, size);
    init_buffer(h_c, size);

    // allocate device memories
    cudaMalloc((void**)&d_a, bufsize);
    cudaMalloc((void**)&d_b, bufsize);
    cudaMalloc((void**)&d_c, bufsize);

    // create list of operation elements
    Operator *ls_operator = new Operator[num_operator];

    sdkStartTimer(&timer);
    
    // execute each operator collesponding data
    for (int i = 0; i < num_operator; i++) {
        int offset = i * size / num_operator;
        ls_operator[i].set_index(i);
        ls_operator[i].async_operation(&h_c[offset], &h_a[offset], &h_b[offset],
                                       &d_c[offset], &d_a[offset], &d_b[offset],
                                       size / num_operator, bufsize / num_operator);
    }

    // synchronize all the stream operation
    cudaDeviceSynchronize();

    sdkStopTimer(&timer);

    // print out the result
    int print_idx = 256;
    printf("compared a sample result...\n");
    printf("host: %.6f, device: %.6f\n",  h_a[print_idx] + h_b[print_idx], h_c[print_idx]);

    // Compute and print the performance
    float elapsed_time_msed = sdkGetTimerValue(&timer);
    float bandwidth = 3 * bufsize * sizeof(float) / elapsed_time_msed / 1e6;
    printf("Time= %.3f msec, bandwidth= %f GB/s\n", elapsed_time_msed, bandwidth);

    // delete timer
    sdkDeleteTimer(&timer);

    // terminate operators
    delete [] ls_operator;

    // terminate device memories
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // terminate host memories
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
    
    return 0;
}

__global__ void
vecAdd_kernel(float *c, const float* a, const float* b)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = 0; i < 500; i++)
        c[idx] = a[idx] + b[idx];
}

void init_buffer(float *data, const int size)
{
    for (int i = 0; i < size; i++) 
        data[i] = rand() / (float)RAND_MAX;
}
```

程序输出结果如下：

```bash
stream 0 - elapsed 11.136 ms
stream 1 - elapsed 16.998 ms
stream 2 - elapsed 23.283 ms
stream 3 - elapsed 29.487 ms
compared a sample result...
host: 1.523750, device: 1.523750
Time= 29.771 msec, bandwidth= 27.050028 GB/
```

可以看出流回调函数可以直接预测核函数的执行时间，但是由于存在`overlap`，会导致靠后的CUDA流执行时间比较长

<figure><img src="../../.gitbook/assets/图片 (90).png" alt=""><figcaption></figcaption></figure>

### 流的优先级

我们从 Operator 类中创建一个派生类，它将处理流的优先级。因此，我们将成员变量 stream 的保护级别从私有成员改为受保护成员。此外，构造函数可以选择性地创建流，因为这可以由派生类完成。完整代码如下：

```c
#include <cstdio>
#include <helper_timer.h>

using namespace std;

__global__ void vecAdd_kernel(float *c, const float* a, const float* b);
void init_buffer(float *data, const int size);

class Operator
{
private:
    int _index;
    StopWatchInterface *p_timer;

    static void CUDART_CB Callback(cudaStream_t stream, cudaError_t status, void* userData);
    void print_time();

protected:
    cudaStream_t stream = nullptr;

public:
    Operator(bool create_stream = true) {
        if (create_stream)
            cudaStreamCreate(&stream);
        sdkCreateTimer(&p_timer);
    }

    ~Operator() {
        if (stream != nullptr)
            cudaStreamDestroy(stream);
        sdkDeleteTimer(&p_timer);
    }

    void set_index(int index) { _index = index; }
    void async_operation(float *h_c, const float *h_a, const float *h_b,
                          float *d_c, float *d_a, float *d_b,
                          const int size, const int bufsize);

}; // Operator

void Operator::CUDART_CB Callback(cudaStream_t stream, cudaError_t status, void* userData) {
    Operator* this_ = (Operator*) userData;
    this_->print_time();
}

void Operator::print_time() {
    // end timer
    sdkStopTimer(&p_timer);
    float elapsed_time_msed = sdkGetTimerValue(&p_timer);
    printf("stream %2d - elapsed %.3f ms \n", _index, elapsed_time_msed);
}

void Operator::async_operation(float *h_c, const float *h_a, const float *h_b,
                          float *d_c, float *d_a, float *d_b,
                          const int size, const int bufsize)
{
    // start timer
    sdkStartTimer(&p_timer);

    // copy host -> device
    cudaMemcpyAsync(d_a, h_a, bufsize, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, h_b, bufsize, cudaMemcpyHostToDevice, stream);

    // launch cuda kernel
    dim3 dimBlock(256);
    dim3 dimGrid(size / dimBlock.x);
    vecAdd_kernel<<< dimGrid, dimBlock, 0, stream >>>(d_c, d_a, d_b);

    // copy device -> host
    cudaMemcpyAsync(h_c, d_c, bufsize, cudaMemcpyDeviceToHost, stream);

    // register callback function
    cudaStreamAddCallback(stream, Operator::Callback, this, 0);
}

class Operator_with_priority: public Operator {
public:
    Operator_with_priority() : Operator(false) {}

    void set_priority(int priority) {
        // 用于创建一个新的 CUDA 流（stream），并且为这个流设置一个优先级
        cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, priority);
    }
};

int main(int argc, char* argv[])
{
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    int size = 1 << 24;
    int bufsize = size * sizeof(float);
    int num_operator = 4;

    if (argc != 1)
        num_operator = atoi(argv[1]);

    // check the current device supports CUDA stream's prority
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); 
    if (prop.streamPrioritiesSupported == 0) {
        printf("This device does not support priority streams");
        return 1;
    }

    // initialize timer
    StopWatchInterface *timer;
    sdkCreateTimer(&timer);

    // allocate host memories
    cudaMallocHost((void**)&h_a, bufsize);
    cudaMallocHost((void**)&h_b, bufsize);
    cudaMallocHost((void**)&h_c, bufsize);

    // initialize host values
    srand(2019);
    init_buffer(h_a, size);
    init_buffer(h_b, size);
    init_buffer(h_c, size);

    // allocate device memories
    cudaMalloc((void**)&d_a, bufsize);
    cudaMalloc((void**)&d_b, bufsize);
    cudaMalloc((void**)&d_c, bufsize);

    // create list of operation elements
    Operator_with_priority *ls_operator = new Operator_with_priority[num_operator];

    // Get Priority range
    int priority_low, priority_high;
    cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);
    printf("Priority Range: low(%d), high(%d)\n", priority_low, priority_high);

    // start to measure the execution time
    sdkStartTimer(&timer);
    
    // execute each operator corresponding data
    // priority setting for each CUDA stream
    for (int i = 0; i < num_operator; i++) {
        // int offset = i * size / num_operator;
        ls_operator[i].set_index(i);
        if (i + 1 == num_operator)
            ls_operator[i].set_priority(priority_high);
        else
            ls_operator[i].set_priority(priority_low);
    }

    // operation (copy(H2D), kernel execution, copy(D2H))
    for (int i = 0; i < num_operator; i++) {
        int offset = i * size / num_operator;
        ls_operator[i].async_operation(&h_c[offset], &h_a[offset], &h_b[offset],
                                       &d_c[offset], &d_a[offset], &d_b[offset],
                                       size / num_operator, bufsize / num_operator);
    }

    // synchronize all the stream operation
    cudaDeviceSynchronize();

    // stop to measure the execution time    
    sdkStopTimer(&timer);

    // print out the result
    int print_idx = 256;
    printf("compared a sample result...\n");
    printf("host: %.6f, device: %.6f\n",  h_a[print_idx] + h_b[print_idx], h_c[print_idx]);

    // Compute and print the performance
    float elapsed_time_msed = sdkGetTimerValue(&timer);
    float bandwidth = 3 * bufsize * sizeof(float) / elapsed_time_msed / 1e6;
    printf("Time= %.3f msec, bandwidth= %f GB/s\n", elapsed_time_msed, bandwidth);

    // delete timer
    sdkDeleteTimer(&timer);

    // terminate operators
    delete [] ls_operator;

    // terminate device memories
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // terminate host memories
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
    
    return 0;
}

__global__ void
vecAdd_kernel(float *c, const float* a, const float* b)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = 0; i < 500; i++)
        c[idx] = a[idx] + b[idx];
}

void init_buffer(float *data, const int size)
{
    for (int i = 0; i < size; i++) 
        data[i] = rand() / (float)RAND_MAX;
}
```

* 程序输出结果如下：

```bash
Priority Range: low(0), high(-1)
stream 0 - elapsed 11.119 ms
stream 3 - elapsed 19.126 ms
stream 1 - elapsed 23.327 ms
stream 2 - elapsed 29.422 ms
compared a sample result...
host: 1.523750, device: 1.523750
Time= 29.730 msec, bandwidth= 27.087332 GB/s
```

<figure><img src="../../.gitbook/assets/图片 (91).png" alt=""><figcaption></figcaption></figure>

可以看到，优先级最高的 CUDA 流（流 21）抢占了第二个 CUDA 流（本例中为流 19）的位置，因此流 21 可以在流 19 执行完毕前完成工作。<mark style="color:red;">请注意，数据传输顺序不会因优先级而改变。</mark>

### 使用事件估计核函数执行的时间

之前的 GPU 运行时间估算有一个局限性，那就是无法测量内核的执行时间。这是因为我们在主机端使用了定时 API。因此，我们需要主机和 GPU 同步才能测量内核执行时间，考虑到时间开销和对程序性能的影响，这种做法在实际工作中是不现实的。

这可以使用 CUDA 事件来解决。CUDA 事件与 CUDA 数据流一起记录 GPU 端事件。CUDA 事件可以是基于 GPU 状态的事件，并记录相关时间。利用这一点，我们可以估算内核执行时间

CUDA 事件由 `cudaEvent_t` 句柄管理。我们可以使用 `cudaEventCreate()`创建一个 CUDA 事件句柄，并使用 `cudaEventDestroy()` 终止它。要记录事件时间，可以使用 `cudaEventRecord()`。然后，CUDA 事件句柄会记录 GPU 的事件时间。该函数还接受 CUDA 流，这样我们就能枚举出特定 CUDA 流的事件时间。获得内核执行的开始和结束事件后，就可以使用<mark style="color:red;">以毫秒为单位</mark>的 `cudaEventElapsedTime()`来获得经过时间。

### 使用CUDA事件

修改之前的代码：

```c
#include <cstdio>
#include <helper_timer.h>

using namespace std;

__global__ void vecAdd_kernel(float *c, const float* a, const float* b);
void init_buffer(float *data, const int size);

int main(int argc, char* argv[])
{
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    int size = 1 << 24;
    int bufsize = size * sizeof(float);

    // allocate host memories
    cudaMallocHost((void**)&h_a, bufsize);
    cudaMallocHost((void**)&h_b, bufsize);
    cudaMallocHost((void**)&h_c, bufsize);

    // initialize host values
    srand(2019);
    init_buffer(h_a, size);
    init_buffer(h_b, size);
    init_buffer(h_c, size);

    // allocate device memories
    cudaMalloc((void**)&d_a, bufsize);
    cudaMalloc((void**)&d_b, bufsize);
    cudaMalloc((void**)&d_c, bufsize);

    // copy host -> device
    cudaMemcpyAsync(d_a, h_a, bufsize, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_b, h_b, bufsize, cudaMemcpyHostToDevice);

    // initialize the host timer
    StopWatchInterface *timer;
    sdkCreateTimer(&timer);

    cudaEvent_t start, stop;
    // create CUDA events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // start to measure the execution time
    sdkStartTimer(&timer);
    cudaEventRecord(start);

    // launch cuda kernel
    dim3 dimBlock(256);
    dim3 dimGrid(size / dimBlock.x);
    vecAdd_kernel<<< dimGrid, dimBlock >>>(d_c, d_a, d_b);

    // record the event right after the kernel execution finished
    cudaEventRecord(stop);

    // Synchronize the device to measure the execution time from the host side
    cudaEventSynchronize(stop); // we also can make synchronization based on CUDA event
    sdkStopTimer(&timer);
    
    // copy device -> host
    cudaMemcpyAsync(h_c, d_c, bufsize, cudaMemcpyDeviceToHost);

    // print out the result
    int print_idx = 256;
    printf("compared a sample result...\n");
    printf("host: %.6f, device: %.6f\n",  h_a[print_idx] + h_b[print_idx], h_c[print_idx]);

    // print estimated kernel execution time
    float elapsed_time_msed = 0.f;
    cudaEventElapsedTime(&elapsed_time_msed, start, stop);
    printf("CUDA event estimated - elapsed %.3f ms \n", elapsed_time_msed);

    // Compute and print the performance
    elapsed_time_msed = sdkGetTimerValue(&timer);
    printf("Host measured time= %.3f msec/s\n", elapsed_time_msed);

    // terminate device memories
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // terminate host memories
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);

    // delete timer
    sdkDeleteTimer(&timer);

    // terminate CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}

__global__ void
vecAdd_kernel(float *c, const float* a, const float* b)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = 0; i < 500; i++)
        c[idx] = a[idx] + b[idx];
}

void init_buffer(float *data, const int size)
{
    for (int i = 0; i < size; i++) 
        data[i] = rand() / (float)RAND_MAX;
}
```

上述代码中定时器需要 GPU 和主机同步， 为了同步，我们使用了 `cudaEventSynchronize(stop)` 函数，让主机线程与事件同步。&#x20;

分别用 CUDA 事件和计时器测量核函数运行时间，我们可以发现两者是不同的。 可以使用 NVIDIA Profiler 来验证两种方法谁更准确一些，使用 `# nvprof ./cuda_event` 命令，输出如下：

<figure><img src="../../.gitbook/assets/图片 (92).png" alt=""><figcaption></figcaption></figure>

由上图可知CUDA 事件能提供准确的结果。

<mark style="color:red;">使用 CUDA 事件的另一个好处是，我们可以在多个 CUDA 流中同时测量多个内核的执行时间。</mark>

### 多流估计

<mark style="color:red;">`cudaEventRecord()`</mark><mark style="color:red;">函数与主机是异步的。要实现事件与主机的同步，我们需要使用</mark> <mark style="color:red;"></mark><mark style="color:red;">`cudaEventSynchronize()`</mark><mark style="color:red;">。</mark>

```c
#include <cstdio>
#include <helper_timer.h>

using namespace std;

__global__ void vecAdd_kernel(float *c, const float* a, const float* b);
void init_buffer(float *data, const int size);

class Operator
{
private:
    int index;
    StopWatchInterface *p_timer;

    static void CUDART_CB Callback(cudaStream_t stream, cudaError_t status, void* userData);
    void print_time();

    cudaEvent_t start, stop;

protected:
    cudaStream_t stream = nullptr;

public:
    Operator(bool create_stream = true) {
        if (create_stream)
            cudaStreamCreate(&stream);
        sdkCreateTimer(&p_timer);

        // create CUDA events
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~Operator() {
        if (stream != nullptr)
            cudaStreamDestroy(stream);
        sdkDeleteTimer(&p_timer);

        // terminate CUDA events
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void set_index(int idx) { index = idx; }
    void async_operation(float *h_c, const float *h_a, const float *h_b,
                          float *d_c, float *d_a, float *d_b,
                          const int size, const int bufsize);
    void print_kernel_time();

}; // Operator

void Operator::CUDART_CB Callback(cudaStream_t stream, cudaError_t status, void* userData) {
    Operator* this_ = (Operator*) userData;
    this_->print_time();
}

void Operator::print_time() {
    // end timer
    sdkStopTimer(&p_timer);
    float elapsed_time_msed = sdkGetTimerValue(&p_timer);
    printf("stream %2d - elapsed %.3f ms \n", index, elapsed_time_msed);
}

void Operator::print_kernel_time() {
    float elapsed_time_msed = 0.f;
    cudaEventElapsedTime(&elapsed_time_msed, start, stop);
    printf("kernel in stream %2d - elapsed %.3f ms \n", index, elapsed_time_msed);
}

void Operator::async_operation(float *h_c, const float *h_a, const float *h_b,
                          float *d_c, float *d_a, float *d_b,
                          const int size, const int bufsize)
{
    // start timer
    sdkStartTimer(&p_timer);

    // copy host -> device
    cudaMemcpyAsync(d_a, h_a, bufsize, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, h_b, bufsize, cudaMemcpyHostToDevice, stream);

    // record the event before the kernel execution
    cudaEventRecord(start, stream);

    // launch cuda kernel
    dim3 dimBlock(256);
    dim3 dimGrid(size / dimBlock.x);
    vecAdd_kernel<<< dimGrid, dimBlock, 0, stream >>>(d_c, d_a, d_b);

    // record the event right after the kernel execution finished
    cudaEventRecord(stop, stream);

    // copy device -> host
    cudaMemcpyAsync(h_c, d_c, bufsize, cudaMemcpyDeviceToHost, stream);

    // what happen if we include CUDA event synchronize?
    // QUIZ: cudaEventSynchronize(stop);

    // register callback function
    cudaStreamAddCallback(stream, Operator::Callback, this, 0);
}

class Operator_with_priority: public Operator {
public:
    Operator_with_priority() : Operator(false) {}

    void set_priority(int priority) {
        cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, priority);
    }
};

int main(int argc, char* argv[])
{
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    int size = 1 << 24;
    int bufsize = size * sizeof(float);
    int num_operator = 4;

    if (argc != 1)
        num_operator = atoi(argv[1]);

    // check the current device supports CUDA stream's prority
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); 
    if (prop.streamPrioritiesSupported == 0) {
        printf("This device does not support priority streams");
        return 1;
    }

    // initialize timer
    StopWatchInterface *timer;
    sdkCreateTimer(&timer);

    // allocate host memories
    cudaMallocHost((void**)&h_a, bufsize);
    cudaMallocHost((void**)&h_b, bufsize);
    cudaMallocHost((void**)&h_c, bufsize);

    // initialize host values
    srand(2019);
    init_buffer(h_a, size);
    init_buffer(h_b, size);
    init_buffer(h_c, size);

    // allocate device memories
    cudaMalloc((void**)&d_a, bufsize);
    cudaMalloc((void**)&d_b, bufsize);
    cudaMalloc((void**)&d_c, bufsize);

    // create list of operation elements
    Operator_with_priority *ls_operator = new Operator_with_priority[num_operator];

    // Get Priority range
    int priority_low, priority_high;
    cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);
    printf("Priority Range: low(%d), high(%d)\n", priority_low, priority_high);

    // start to measure the execution time
    sdkStartTimer(&timer);
    
    // execute each operator collesponding data
    // priority setting for each CUDA stream
    for (int i = 0; i < num_operator; i++) {
        // int offset = i * size / num_operator;
        ls_operator[i].set_index(i);
        if (i + 1 == num_operator)
            ls_operator[i].set_priority(priority_high);
        else
            ls_operator[i].set_priority(priority_low);
    }

    // operation (copy(H2D), kernel execution, copy(D2H))
    for (int i = 0; i < num_operator; i++) {
        int offset = i * size / num_operator;
        ls_operator[i].async_operation(&h_c[offset], &h_a[offset], &h_b[offset],
                                       &d_c[offset], &d_a[offset], &d_b[offset],
                                       size / num_operator, bufsize / num_operator);
    }

    // synchronize all the stream operation
    cudaDeviceSynchronize();

    // stop to measure the execution time    
    sdkStopTimer(&timer);

    // print each cuda stream execution time
    for (int i = 0; i < num_operator; i++)
        ls_operator[i].print_kernel_time(); 

    // print out the result
    int print_idx = 256;
    printf("compared a sample result...\n");
    printf("host: %.6f, device: %.6f\n",  h_a[print_idx] + h_b[print_idx], h_c[print_idx]);

    // Compute and print the performance
    float elapsed_time_msed = sdkGetTimerValue(&timer);
    float bandwidth = 3 * bufsize * sizeof(float) / elapsed_time_msed / 1e6;
    printf("Time= %.3f msec, bandwidth= %f GB/s\n", elapsed_time_msed, bandwidth);

    // delete timer
    sdkDeleteTimer(&timer);

    // terminate operators
    delete [] ls_operator;

    // terminate device memories
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // terminate host memories
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
    
    return 0;
}

__global__ void
vecAdd_kernel(float *c, const float* a, const float* b)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = 0; i < 500; i++)
        c[idx] = a[idx] + b[idx];
}

void init_buffer(float *data, const int size)
{
    for (int i = 0; i < size; i++) 
        data[i] = rand() / (float)RAND_MAX;
}
```

* 程序输出结果：

```sh
Priority Range: low(0), high(-1)
stream 0 - elapsed 11.348 ms
stream
3 - elapsed 19.435 ms
stream 1 - elapsed 22.707 ms
stream 2 - elapsed 35.768 ms
kernel in stream 0 - elapsed 6.052 ms
kernel in stream 1 - elapsed 14.820 ms
kernel in stream 2 - elapsed 17.461 ms
kernel in stream 3 - elapsed 6.190 ms
compared a sample result...
host: 1.523750, device: 1.523750
Time= 35.993 msec, bandwidth= 22.373972 GB/s
```

{% hint style="info" %}
如果在代码中包含了`cudaEventSynchronize(stop);`，那么在该语句处会发生同步操作，即等待与`stop`事件相关联的CUDA流中的所有操作完成。这意味着在主机端的代码将会等待直到与`stop`事件相关联的CUDA流中的所有操作都完成后才会继续执行。因此，在此处包含`cudaEventSynchronize(stop);`将会导致主机端的代码在确认与CUDA流相关联的所有操作都已完成后才会继续执行，而不会提前继续执行。
{% endhint %}

### 使用OpenMP的调度操作

在使用OpenMP的同时使用CUDA，不仅可以提高便携性和生产效率，而且还可以提 高主机代码的性能。在之前的例子中，我们使用了一个循环调度操作，与此不同， 我们使用了OpenMP线程调度操作到不同的流中，具体方法如下所示：

```c
omp_set_num_threads(num_operator);
    #pragma omp parallel
    {
        int i = omp_get_thread_num();
        int offset = i * size / num_operator;
        ls_operator[i].set_index(i);
        ls_operator[i].async_operation(&h_c[offset], &h_a[offset], &h_b[offset],
                                    &d_c[offset], &d_a[offset], &d_b[offset],
                                    size / num_operator, bufsize / num_operator);
    }
```

程序输出结果如下：

```bash
stream 0 - elapsed 10.734 ms
stream 2 - elapsed 16.153 ms
stream 3 - elapsed 21.968 ms
stream 1 - elapsed 27.668 ms
compared a sample result...
host: 1.523750, device: 1.523750
Time= 27.836 msec, bandwidth= 28.930389GB/s
```

每当执行该程序时，你会发现每个数据流完成工作的顺序都不一样。此外，每个流显示的时间也不同。这是因为 OpenMP 可以创建多个线程，而操作是在运行时确定的。

<figure><img src="../../.gitbook/assets/图片 (93).png" alt=""><figcaption></figcaption></figure>
