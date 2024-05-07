# 🤗 QiuckLearnFromPicture

### GPU 加速应用程序与 CPU 应用程序对比

<figure><img src="../../.gitbook/assets/图片 (101).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/图片 (102).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/图片 (103).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/图片 (104).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/图片 (105).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/图片 (106).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/图片 (107).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/图片 (108).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/图片 (109).png" alt=""><figcaption></figcaption></figure>

### CUDA 线程层次结构

<figure><img src="../../.gitbook/assets/图片 (110).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/图片 (111).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/图片 (112).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/图片 (113).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/图片 (114).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/图片 (115).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/图片 (116).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/图片 (117).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/图片 (118).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/图片 (119).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/图片 (120).png" alt=""><figcaption></figcaption></figure>

### CUDA 提供的线程层次结构变量

<figure><img src="../../.gitbook/assets/图片 (121).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/图片 (122).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/图片 (123).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/图片 (124).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/图片 (125).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/图片 (126).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/图片 (127).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/图片 (128).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/图片 (129).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/图片 (130).png" alt=""><figcaption></figcaption></figure>

### 协调并行线程

<figure><img src="../../.gitbook/assets/图片 (132).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/图片 (133).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/图片 (134).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/图片 (135).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/图片 (136).png" alt=""><figcaption></figcaption></figure>

通过这些变量，公式 `threadIdx.x + blockIdx.x * blockDim.x` 可将每个线程映射到向量的元素中

<figure><img src="../../.gitbook/assets/图片 (137).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/图片 (138).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/图片 (139).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/图片 (140).png" alt=""><figcaption></figcaption></figure>

{% hint style="info" %}
<mark style="color:red;">必须使用代码检查并确保经由公式</mark> <mark style="color:red;"></mark><mark style="color:red;">`threadIdx.x + blockIdx.x * blockDim.x`</mark> <mark style="color:red;"></mark><mark style="color:red;">计算出的 dataIndex 小于 N（数据元素数量）。</mark>
{% endhint %}

### 网格跨度循环

<figure><img src="../../.gitbook/assets/图片 (141).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/图片 (142).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/图片 (143).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/图片 (144).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/图片 (145).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/图片 (146).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/图片 (147).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/图片 (148).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/图片 (149).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/图片 (150).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/图片 (151).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/图片 (152).png" alt=""><figcaption></figcaption></figure>

```c
#include <stdio.h>

void init(int *a, int N)
{
  int i;
  for (i = 0; i < N; ++i)
  {
    a[i] = i;
  }
}

__global__
void doubleElements(int *a, int N)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  for (int i = idx; i < N; i += stride)
  {
    a[i] *= 2;
  }
}

bool checkElementsAreDoubled(int *a, int N)
{
  int i;
  for (i = 0; i < N; ++i)
  {
    if (a[i] != i*2) return false;
  }
  return true;
}

int main()
{
  int N = 10000;
  int *a;

  size_t size = N * sizeof(int);
  cudaMallocManaged(&a, size);

  init(a, N);

  size_t threads_per_block = 256;
  size_t number_of_blocks = 32;

  doubleElements<<<number_of_blocks, threads_per_block>>>(a, N);
  cudaDeviceSynchronize();

  bool areDoubled = checkElementsAreDoubled(a, N);
  printf("All elements were doubled? %s\n", areDoubled ? "TRUE" : "FALSE");

  cudaFree(a);
}
```

### 并发 CUDA 流

<figure><img src="../../.gitbook/assets/图片 (1).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/图片 (1) (1).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/图片 (2).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/图片 (3).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/图片 (4).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/图片 (6).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/图片 (7).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/图片 (8).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/图片 (9).png" alt=""><figcaption></figcaption></figure>

默认流与非默认流中的操作不会发生重叠。

<figure><img src="../../.gitbook/assets/图片 (10).png" alt=""><figcaption></figcaption></figure>

{% hint style="info" %}
&#x20;<mark style="color:red;">默认流较为特殊。默认流中执行任何操作期间，任何非默认流中皆不可同时执行任何操作，</mark><mark style="color:red;">**默认流将等待非默认流全部执行完毕后再开始运行，而且在其执行完毕后，其他非默认流才能开始执行。**</mark>
{% endhint %}

#### **使用流将复制和计算进行重叠**

通过使用默认流，典型的三步式 CUDA 程序会顺次执行 HtoD 复制、计算和 DtoH 复制（为便于演示，下面的图片中使用简略代码），如下图：

<figure><img src="../../.gitbook/assets/图片 (11).png" alt=""><figcaption></figcaption></figure>

但是这样明显是不可行的。回忆一下，非默认流中的操作顺序不固定，因此可能会出现这种情况：

<figure><img src="../../.gitbook/assets/图片 (12).png" alt=""><figcaption></figcaption></figure>

在其所需的数据传输到 GPU 之前，计算可能便会开始，

<figure><img src="../../.gitbook/assets/图片 (13).png" alt=""><figcaption></figcaption></figure>

我们还可采用另一种初级做法：将所有操作全部发布在同一个非默认流中，以确保数据和计算的顺序，

<figure><img src="../../.gitbook/assets/图片 (14).png" alt=""><figcaption></figcaption></figure>

但这样做与使用默认流没有区别，结果是依然没有重叠。思考一下，如果采用现有程序并将数据分为 2 块

<figure><img src="../../.gitbook/assets/图片 (15).png" alt=""><figcaption></figcaption></figure>

**如果现将针对每个数据块的所有操作移至其各自独立的非默认流**，数据和计算顺序得以保持，同时能够实现部分重叠。

<figure><img src="../../.gitbook/assets/图片 (16).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/图片 (17).png" alt=""><figcaption></figcaption></figure>

根据假设，通过增加数据块数量，重叠效果可能更好。要获得理想的分块数量，最好的途径是观察程序性能。

<figure><img src="../../.gitbook/assets/图片 (18).png" alt=""><figcaption></figcaption></figure>

将数据分块以在多个流中使用时，索引可能较为棘手。让我们通过几个示例了解一下如何进行索引。首先为所有数据块分配所需数据，为使示例更加清晰，我们使用了较小规模的数据。

1. 为CPU和GPU分配内存（N表示总的数据量）；

```c
cudaMallocHost(&data_cpu, N)
cudaMalloc(&data_gpu, N)
```

2. 接下来，定义流的数量，并通过执行循环代码以数组形式创建和收集流（此处定义的流的数量为2）；

```c
num_streams = 2
for stream_i in num_streams
	cudaStreamCreate(stream)
	streams[stream_i] = stream
```

3. 每个数据块的大小取决于数据条目的数量以及流的数量；

```c
chunk_size = N / num_streams
```

4. 每个流需要处理一个数据块，我们需要计算其在整个数据集中的索引。为此，我们将遍历流的数量，从 0 开始然后乘以数据块大小。从 lower 索引开始，并使用一个数据块大小的数据，如此即可从全部数据中获得流数据，此方法将会应用至每一个 stream\_i；

```c
for stream_i in num_streams
	lower = chunk_size*stream_i
```

5. 计算完这些值后，我们现在即可执行非默认流 HtoD 内存复制；

```c
cudaMemcpyAsync(
	data_cpu+lower,
	data_gpu+lower, 	sizeof(uint64_t)*chunk_size, 	cudaMemcpyHostToDevice,
	streams[stream_i]
)
```

<figure><img src="../../.gitbook/assets/图片 (20).png" alt=""><figcaption></figcaption></figure>

&#x20;上面的示例中，N 被流数量整除。如果不能整除呢？**为解决该问题，我们使用向上取整的除法运算来计算数据块大小**。但是这还是会有问题，如下图：

<figure><img src="../../.gitbook/assets/图片 (21).png" alt=""><figcaption></figcaption></figure>

我们确实可以访问所有数据，但又产生了新问题：<mark style="color:red;">对于最后一个数据块而言，数据块大小过小，存在线程访问越界的问题。</mark>

解决方法如下：

1. 为每个数据块计算 upper 索引（不得超过 N）；

```c
upper = min(lower+chunk_size, N)
```

2. 然后使用 upper 和 lower 计算数据块 width；

```c
width = upper - lower
```

3. 现在使用 width 而非数据块大小进行迭代；

<figure><img src="../../.gitbook/assets/图片 (22).png" alt="" width="563"><figcaption></figcaption></figure>

这样我们就能完美适配数据，而不受其大小或流数量的影响。

**复制与计算重叠的代码示例**

```c
// Able to handle when `num_entries % num_streams != 0`.

const uint64_t num_entries = 10;
const uint64_t num_iters = 1UL << 10;

cudaMallocHost(&data_cpu, sizeof(uint64_t)*num_entries);
cudaMalloc    (&data_gpu, sizeof(uint64_t)*num_entries);

// Set the number of streams to not evenly divide num_entries.
const uint64_t num_streams = 3;

cudaStream_t streams[num_streams];
for (uint64_t stream = 0; stream < num_streams; stream++)
    cudaStreamCreate(&streams[stream]);

// Use round-up division (`sdiv`, defined in helper.cu) so `num_streams*chunk_size`
// is never less than `num_entries`.
// This can result in `num_streams*chunk_size` being greater than `num_entries`, meaning
// we will need to guard against out-of-range errors in the final "tail" stream (see below).
const uint64_t chunk_size = sdiv(num_entries, num_streams);

for (uint64_t stream = 0; stream < num_streams; stream++) {

    const uint64_t lower = chunk_size*stream;
    // For tail stream `lower+chunk_size` could be out of range, so here we guard against that.
    const uint64_t upper = min(lower+chunk_size, num_entries);
    // Since the tail stream width may not be `chunk_size`,
    // we need to calculate a separate `width` value.
    const uint64_t width = upper-lower;

    // Use `width` instead of `chunk_size`.
    cudaMemcpyAsync(data_gpu+lower, data_cpu+lower, 
           sizeof(uint64_t)*width, cudaMemcpyHostToDevice, 
           streams[stream]);

    // Use `width` instead of `chunk_size`.
    decrypt_gpu<<<80*32, 64, 0, streams[stream]>>>
        (data_gpu+lower, width, num_iters);

    // Use `width` instead of `chunk_size`.
    cudaMemcpyAsync(data_cpu+lower, data_gpu+lower, 
           sizeof(uint64_t)*width, cudaMemcpyDeviceToHost, 
           streams[stream]);
}

// Destroy streams.
for (uint64_t stream = 0; stream < num_streams; stream++)
    cudaStreamDestroy(streams[stream]);

```

完整代码如下：

```c
#include <cstdint>
#include <iostream>
#include "helpers.cuh"
#include "encryption.cuh"

void encrypt_cpu(uint64_t * data, uint64_t num_entries, 
                 uint64_t num_iters, bool parallel=true) {

    #pragma omp parallel for if (parallel)
    for (uint64_t entry = 0; entry < num_entries; entry++)
        data[entry] = permute64(entry, num_iters);
}

__global__ 
void decrypt_gpu(uint64_t * data, uint64_t num_entries, 
                 uint64_t num_iters) {

    const uint64_t thrdID = blockIdx.x*blockDim.x+threadIdx.x;
    const uint64_t stride = blockDim.x*gridDim.x;

    for (uint64_t entry = thrdID; entry < num_entries; entry += stride)
        data[entry] = unpermute64(data[entry], num_iters);
}

bool check_result_cpu(uint64_t * data, uint64_t num_entries,
                      bool parallel=true) {

    uint64_t counter = 0;

    #pragma omp parallel for reduction(+: counter) if (parallel)
    for (uint64_t entry = 0; entry < num_entries; entry++)
        counter += data[entry] == entry;

    return counter == num_entries;
}

int main (int argc, char * argv[]) {

    Timer timer;
    Timer overall;

    const uint64_t num_entries = 1UL << 26;
    const uint64_t num_iters = 1UL << 10;
    const bool openmp = true;

    // Define the number of streams.
    const uint64_t num_streams = 32;
    
    // Use round-up division to calculate chunk size.
    const uint64_t chunk_size = sdiv(num_entries, num_streams);

    timer.start();
    uint64_t * data_cpu, * data_gpu;
    cudaMallocHost(&data_cpu, sizeof(uint64_t)*num_entries);
    cudaMalloc    (&data_gpu, sizeof(uint64_t)*num_entries);
    timer.stop("allocate memory");
    check_last_error();

    timer.start();
    encrypt_cpu(data_cpu, num_entries, num_iters, openmp);
    timer.stop("encrypt data on CPU");

    timer.start();
    
    // Create array for storing streams.
    cudaStream_t streams[num_streams];
    
    // Create number of streams and store in array.
    for (uint64_t stream = 0; stream < num_streams; stream++)
        cudaStreamCreate(&streams[stream]);
    timer.stop("create streams");
    check_last_error();

    overall.start();
    timer.start();
    
    // For each stream...
    for (uint64_t stream = 0; stream < num_streams; stream++) {
        
        // ...calculate index into global data (`lower`) and size of data for it to process (`width`).
        const uint64_t lower = chunk_size*stream;
        const uint64_t upper = min(lower+chunk_size, num_entries);
        const uint64_t width = upper-lower;

        // ...copy stream's chunk to device.
        cudaMemcpyAsync(data_gpu+lower, data_cpu+lower, 
               sizeof(uint64_t)*width, cudaMemcpyHostToDevice, 
               streams[stream]);

        // ...compute stream's chunk.
        decrypt_gpu<<<80*32, 64, 0, streams[stream]>>>
            (data_gpu+lower, width, num_iters);

        // ...copy stream's chunk to host.
        cudaMemcpyAsync(data_cpu+lower, data_gpu+lower, 
               sizeof(uint64_t)*width, cudaMemcpyDeviceToHost, 
               streams[stream]);
    }

    for (uint64_t stream = 0; stream < num_streams; stream++)
	// Synchronize streams before checking results on host.
        cudaStreamSynchronize(streams[stream]);    
    
    // Note modification of timer instance use.
    timer.stop("asynchronous H2D->kernel->D2H");
    overall.stop("total time on GPU");
    check_last_error();
    
    timer.start();
    const bool success = check_result_cpu(data_cpu, num_entries, openmp);
    std::cout << "STATUS: test " 
              << ( success ? "passed" : "failed")
              << std::endl;
    timer.stop("checking result on CPU");

    timer.start();      
    for (uint64_t stream = 0; stream < num_streams; stream++)
        // Destroy streams.
        cudaStreamDestroy(streams[stream]);    
    timer.stop("destroy streams");
    check_last_error();

    timer.start();
    cudaFreeHost(data_cpu);
    cudaFree    (data_gpu);
    timer.stop("free memory");
    check_last_error();
}

```

### reference

* [https://blog.csdn.net/qq\_31985307/article/details/126237070](https://blog.csdn.net/qq\_31985307/article/details/126237070)
* [https://learn.nvidia.com/courses/course-detail?course\_id=course-v1:DLI+S-AC-04+V1-ZH](https://learn.nvidia.com/courses/course-detail?course\_id=course-v1:DLI+S-AC-04+V1-ZH)
