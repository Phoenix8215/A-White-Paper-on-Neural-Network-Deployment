# 😷 CUDA流和事件

### 流和事件的概述

CUDA流是一系列异步的CUDA操作，这些操作按照主机代码确定的顺序在设备上执 行。流能封装这些操作，保持操作的顺序，允许操作在流中排队，并使它们在先前的所有 操作之后执行，并且可以查询排队操作的状态。这些操作包括在主机与设备间进行数据传 输，内核启动以及大多数由主机发起但由设备处理的其他命令。流中操作的执行相对于主 机总是异步的。CUDA运行时决定何时可以在设备上执行操作。<mark style="color:red;">我们的任务是使用CUDA 的API来确保一个异步操作在运行结果被使用之前可以完成。在同一个CUDA流中的操作 有严格的执行顺序，而在不同CUDA流中的操作在执行顺序上不受限制。</mark>使用多个流同时 启动多个内核，可以实现网格级并发。

因为所有在CUDA流中排队的操作都是异步的，所以在主机与设备系统中可以重叠执 行其他操作。在同一时间内将流中排队的操作与其他有用的操作一起执行，可以隐藏执行 那些操作的开销。

**CUDA编程的一个典型模式是以下形式：**

1. 将输入数据从主机移到设备上。&#x20;
2. 在设备上执行一个内核。&#x20;
3. 将结果从设备移回主机中。

在许多情况下，执行内核比传输数据耗时更多。在这些情况下，可以完全隐藏CPU和 GPU之间的通信延迟。<mark style="color:red;">通过将内核执行和数据传输调度到不同的流中，这些操作可以重叠，程序的总运行时间将被缩短</mark>。

{% hint style="info" %}
从软件的角度来看，CUDA操作在不同的流中并发运行；而从硬件上来看，不一定总 是如此。根据PCIe总线争用或每个SM资源的可用性，完成不同的CUDA流可能仍然需要 互相等待。
{% endhint %}

### CUDA流

所有的CUDA操作都在一个流中显式或隐式地运行。流分为 两种类型：

* 隐式声明的流（空流）

如果没有显式地指定一个流，那么内核启动和数据传输将默认使用空流。

* 显式声明的流（非空流）&#x20;

非空流可以被显式地创建和管理。如果想要重叠不同的CUDA操作，必须 使用非空流。

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

请注意附加的流标识符作为第五个参数。默认情况下，流标识符被设置为默认流。这 个函数与主机是异步的，所以调用发布后，控制权将立即返回到主机。将复制操作和非空 流进行关联是很容易的，但是首先需要使用如下代码创建一个非空流：

```c
cudaError_t cudaStreamCreate(cudaStream_t* pStream)
```

cudaStreamCreate创建了一个可以显式管理的非空流。之后，返回到pStream中的流就 可以被当作流参数供cudaMemcpyAsync和其他异步CUDA的API函数来使用。在使用异步 CUDA函数时，常见的疑惑在于，它们可能会从先前启动的异步操作中返回错误代码。

当执行异步数据传输时，必须使用固定（或非分页的）主机内存。可以使用cuda MallocHost函数或cudaHostAlloc函数分配固定内存：

```c
cudaError t cudaMallocHost(void **ptr, sizet size);
cudaError_t cudaHostAlloc(void **pHost, size_t ssize, unsigned int flags);
```

在主机虚拟内存中固定分配，可以确保其在CPU内存中的物理位置在应用程序的整个 生命周期中保持不变。否则，操作系统可以随时自由改变主机虚拟内存的物理位置。如果 在没有固定主机内存的情况下执行一个异步CUDA转移操作，操作系统可能会在物理层面 上移动数组，这样会导致未定义的行为。

{% hint style="info" %}
* `cudaMalloc`分配的是页锁定内存，也称为固定内存。
* 页锁定内存不会被分页到磁盘，因此对于GPU访问非常高效。
* 在某些情况下，人们更喜欢直接使用`cudaMallocHost`来分配页锁定内存，因为它更容易使用。
{% endhint %}

在非默认流中启动内核，必须在内核执行配置中提供一个流标识符作为第四个参数：

```c
kernel name<<<grid, block, sharedMemSize,stream>>>(argument list);
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

**cudaStreamSynchronize强制阻塞主机，直到在给定流中所有的操作都完成了。cudaStreamQuery会检查流中所有操作是否都已经完成，但在它们完成前不会阻塞主机。**当所有操作都完成时cudaStreamQuery函数会返回cudaSuccess，当一个或多个操作仍在执行或等待执行时返回cudaErrorNotReady。

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

<figure><img src="../../.gitbook/assets/图片.png" alt=""><figcaption></figcaption></figure>















