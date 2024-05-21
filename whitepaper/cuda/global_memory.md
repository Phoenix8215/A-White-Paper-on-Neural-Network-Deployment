---
description: >-
  source from 《Professional CUDA C Programming》By Max Grossman&Ty McKercher.I
  made some changes according to my own understanding.
---

# 🤔 全局内存(Global Memory)访问模式

大多数设备端数据访问都是从全局内存开始的，并且多数GPU应用程序容易受内存带 宽的限制。为了在读写数据时达到最佳的性能，内存访问操作必须满足一定的条件。CUDA执行 模型的显著特征之一就是指令必须以线程束为单位进行发布和执行。存储操作也是同样。 在执行内存指令时，线程束中的每个线程都提供了一个正在加载或存储的内存地址。根据线程束中内存地址的分布，内存访问可以被分成不 同的模式。

### 对齐与合并访问

核函数的内存请求通常是在DRAM设备和片上内存间以128字节或32字节内存事务来 实现的。 所有对全局内存的访问都会通过二级缓存，也有许多访问会通过一级缓存，这取决于 访问类型和GPU架构。如果这两级缓存都被用到，那么内存访问是由一个128字节的内存 事务实现的。如果只使用了二级缓存，那么这个内存访问是由一个32字节的内存事务实现 的。

一行一级缓存是128个字节，它映射到设备内存中一个128字节的对齐段。如果线程束 中的每个线程请求一个4字节的值，那么每次请求就会获取128字节的数据，这恰好与缓存 行和设备内存段的大小相契合。

因此在优化应用程序时，你需要注意设备内存访问的两个特性：

* 对齐内存访问(Aligned memory accesses)
* 合并内存访问(Coalesced memory accesses)

当设备内存事务的第一个地址是用于事务服务的缓存粒度的偶数倍时（32字节的二级 缓存或128字节的一级缓存），就会出现对齐内存访问。运行非对齐的加载会造成带宽浪 费。

当一个线程束中全部的32个线程访问一个连续的内存块时，就会出现合并内存访问。&#x20;

对齐合并内存访问的理想状态是线程束从对齐内存地址开始访问一个连续的内存块。

为了最大化全局内存吞吐量，为了组织内存操作进行对齐合并是很重要的。下图描述了 对齐与合并内存的加载操作。在这种情况下，只需要一个128字节的内存事务从设备内存 中读取数据。

<figure><img src="../../.gitbook/assets/图片 (4) (1).png" alt=""><figcaption></figcaption></figure>

下图展示了非对齐和未合并的内存访问。在这种情况下，可能需要3个128 字节的内存事务来从设备内存中读取数据：<mark style="color:red;">一个在偏移量为0的地方开始，读取连续地址 之后的数据；一个在偏移量为256的地方开始，读取连续地址之前的数据；</mark>另一个在偏移 量为128的地方开始读取大量的数据。Note that most of the bytes fetched by the lower and upper memory transactions will not be used, leading to wasted bandwidth.

<figure><img src="../../.gitbook/assets/图片 (1) (1) (1).png" alt=""><figcaption></figcaption></figure>

<mark style="color:red;">一般来说，需要优化内存事务效率：用最少的事务次数满足最多的内存请求。事务数 量和吞吐量的需求随设备的计算能力变化。</mark>

### 全局内存读取

#### 内存加载访问模式&#x20;

内存加载可以分为两类：&#x20;

* 缓存加载（启用一级缓存）
* 没有缓存的加载（禁用一级缓存）

内存加载的访问模式有如下特点：&#x20;

* <mark style="color:red;">有缓存与没有缓存：如果启用一级缓存，则内存加载被缓存</mark>&#x20;
* <mark style="color:red;">对齐与非对齐：如果内存访问的第一个地址是32字节的倍数，则对齐加载</mark>&#x20;
* <mark style="color:red;">合并与未合并：如果线程束访问一个连续的数据块，则加载合并</mark>

缓存加载 缓存加载操作经过一级缓存，在粒度为128字节的一级缓存行上由设备内存事务进行 传输。缓存加载可以分为对齐/非对齐及合并/非合并。

下图所示为理想情况：对齐与合并内存访问。线程束中所有线程请求的地址都在128 字节的缓存行范围内。完成内存加载操作只需要一个128字节的事务。总线的使用率为100%，在这个事务中没有未使用的数据。

<figure><img src="../../.gitbook/assets/图片 (2) (1) (1).png" alt=""><figcaption></figcaption></figure>

下图所示为另一种情况：访问是对齐的，引用的地址不是连续的线程ID，而是128 字节范围内的随机值。由于线程束中线程请求的地址仍然在一个缓存行范围内，所以只需 要一个128字节的事务来完成这一内存加载操作，总线利用率仍然是100%。

<figure><img src="../../.gitbook/assets/图片 (3) (1) (1).png" alt=""><figcaption></figcaption></figure>

下图也说明了一种情况：线程束请求32个连续4个字节的非对齐数据元素。在全局 内存中线程束的线程请求的地址落在两个128字节段范围内。因为当启用一级缓存时，由 SM执行的物理加载操作必须在128个字节的界线上对齐，所以要求有两个128字节的事务 来执行这段内存加载操作。总线利用率为50%，并且在这两个事务中加载的字节有一半是未使用的。

<figure><img src="../../.gitbook/assets/图片 (4) (1) (1).png" alt=""><figcaption></figcaption></figure>

下图说明了一种情况：线程束中所有线程请求相同的地址。因为被引用的字节落在 一个缓存行范围内，所以只需请求一个内存事务，但总线利用率非常低。如果加载的值是4字节的，则总线利用率是4字节请求/128字节加载＝3.125%。

<figure><img src="../../.gitbook/assets/图片 (5) (1).png" alt=""><figcaption></figcaption></figure>

下图所示为最坏的情况：线程束中线程请求分散于全局内存中的32个4字节地址。 尽管线程束请求的字节总数仅为128个字节，但地址要占用N个缓存行（0＜N≤32）。完成 一次内存加载操作需要申请N次内存事务。

<figure><img src="../../.gitbook/assets/图片 (6) (1).png" alt=""><figcaption></figcaption></figure>

* CPU一级缓存和GPU一级缓存之间的差异

&#x20;CPU一级缓存优化了时间和空间局部性。GPU一级缓存是专为空间局部性而不是为时 间局部性设计的。频繁访问一个一级缓存的内存位置不会增加数据留在缓存中的概率。

### 没有缓存的加载

没有缓存的加载在内存段的粒度上为32个字节，下图为理想情况：对齐与合并内存访问。128个字节请求的地址占用了4个内存 段，总线利用率为100%。

<figure><img src="../../.gitbook/assets/图片 (7) (1).png" alt=""><figcaption></figcaption></figure>

下图说明了一种情况：线程束请求32个连续的4字节元素但加载没有对齐到128个字 节的边界。请求的地址最多落在5个内存段内，总线利用率至少为80%。与这些类型的请 求缓存加载相比，使用非缓存加载会提升性能，这是因为加载了更少的未请求字节。

<figure><img src="../../.gitbook/assets/图片 (8) (1).png" alt=""><figcaption></figcaption></figure>

下图说明了一种情况：线程束中所有线程请求相同的数据。地址落在一个内存段 内，总线的利用率是请求的4字节/加载的32字节＝12.5%，在这种情况下，非缓存加载性 能也是优于缓存加载的性能。

<figure><img src="../../.gitbook/assets/图片 (9) (1).png" alt=""><figcaption></figcaption></figure>

下图说明了最坏的一种情况：线程束请求32个分散在全局内存中的4字节字。由于 请求的128个字节最多落在N个32字节的内存分段内而不是N个128个字节的缓存行内，所 以相比于缓存加载，即便是最坏的情况也有所改善。

<figure><img src="../../.gitbook/assets/图片 (10) (1).png" alt=""><figcaption></figcaption></figure>

### 非对齐读取的示例

为了说明核函数中非对齐访问对性能的影响，我们以向量加法为例进行修改，去掉所有的内存加载操作，来指定一个偏移量。注意在下面的核函数中使用了两 种索引。新的索引k由给定的偏移量上移，由于偏移量的值可能会导致加载出现非对齐加 载。只有加载数组A和数组B的操作会用到索引k。对数组C的写操作仍使用原来的索引i， 以确保写入访问保持对齐。

```c
 __global__ void readOffset(float *A, float *B, float *C, const int n, 
    int offset) {
   unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
   unsigned int k = i + offset;
   if (k < n) C[i] = A[k] + B[k];
 } 
```

完整代码如下：

```c
#include <stdio.h>
#include <time.h>
#include <sys/time.h>

//sudo nvprof --devices 0 --metrics gld_transactions ./readSegment 0
//sudo nvprof --devices 0 --metrics gld_transactions ./readSegment 11
//sudo nvprof --devices 0 --metrics gld_transactions ./readSegment 128

//sudo nvprof --devices 0 --metrics gld_efficiency ./readSegment 0
//sudo nvprof --devices 0 --metrics gld_efficiency ./readSegment 11
//sudo nvprof --devices 0 --metrics gld_efficiency ./readSegment 128


void initialData(float* data, int size){
    time_t t;
    srand((unsigned int) time(&t));
    for(int i=0;i<size;i++){
        data[i] = (float)( rand() & 0xFF)/10.0f;
    }
}

void checkResult(float *hostRef, float *gpuRef, const int N){
    double eplison = 1.0E-5;
    int match = 1;
    for(int i=0;i<N;i++){
        if(abs(hostRef[i]-gpuRef[i])>eplison){
            match = 0;
            printf("do not match\n");
            break;
        }
    }

    if(match) printf("match!\n");
    return;

}

double seconds(){
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

__global__ void warmup(float* A, float* B, float* C, int nElem, int offset){
    unsigned int idx = blockDim.x*blockIdx.x + threadIdx.x;
    C[idx] = A[idx] + B[idx];
}

__global__ void readOffset(float* A, float* B, float* C, int nElem, int offset){
    unsigned int idx = blockDim.x*blockIdx.x + threadIdx.x;
    unsigned int k = idx+offset;
    if(k<nElem)C[idx] = A[k]+B[k];
}

__global__ void readOffsetUnroll4(float* A, float* B, float* C, const int n, int offset){
    unsigned i = blockDim.x*blockIdx.x + threadIdx.x;
    unsigned k = i + offset;
    if((k + 3*blockDim.x)<n){
        C[i] = A[k] + B[k];
        C[i+blockDim.x] = A[k+blockDim.x] + B[k+blockDim.x];
        C[i+2*blockDim.x] = A[k+2*blockDim.x] + B[k+2*blockDim.x];
        C[i+3*blockDim.x] = A[k+3*blockDim.x] + B[k+3*blockDim.x];
    }
}

void sumArrayOnHost(float* h_a, float* h_b, float* h_c, const size_t size, int offset){
    for(int i=offset, k=0;i<size;i++,k++){
        h_c[k] = h_a[i] + h_b[i];
    }
}


int main(int argc, char** argv){
    int dev = 0;
    cudaSetDevice(dev);

    cudaDeviceProp DeviceProp;
    cudaGetDeviceProperties(&DeviceProp, dev);

    printf("device %d: %s starting...\n", dev, DeviceProp.name);

    int nElem = 1<<24;
    size_t nBytes = nElem * sizeof(float);

    int blocksize = 512;
    int offset = 0;
    if(argc>1) offset = atoi(argv[1]);
    if(argc>2) blocksize = atoi(argv[2]);

    dim3 block(blocksize, 1);
    dim3 grid((nElem+block.x-1)/block.x, 1);

    float* h_A = (float *)malloc(nBytes);
    float* h_B = (float *)malloc(nBytes);
    float* hostRef = (float *)malloc(nBytes);
    float* gpuRef = (float *)malloc(nBytes);

    initialData(h_A, nElem);
    memcpy(h_B, h_A, nBytes);

    sumArrayOnHost(h_A, h_B, hostRef, nElem, offset);

    float *d_A, *d_B, *d_C;
    cudaMalloc((float**)&d_A, nBytes);
    cudaMalloc((float**)&d_B, nBytes);
    cudaMalloc((float**)&d_C, nBytes);

    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

    double iStart = seconds();
    warmup<<<grid, block>>>(d_A, d_B, d_C, nElem, offset);
    cudaDeviceSynchronize();
    double iElaps = seconds() - iStart;
    printf("warmup <<< %4d, %4d >>> offset %4d elpsed %f sec\n", grid.x, block.x, offset, iElaps);

    dim3 blockunroll(blocksize, 1);
    dim3 gridunroll(((nElem+block.x-1)/block.x)/4, 1);

    iStart = seconds();
    readOffset<<<grid, block>>>(d_A, d_B, d_C, nElem, offset);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    printf("readOffset <<< %4d, %4d >>> offset %4d elpsed %f sec\n", grid.x, block.x, offset, iElaps);

    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
    checkResult(hostRef, gpuRef, nElem-offset);

    iStart = seconds();
    readOffset<<<gridunroll, blockunroll>>>(d_A, d_B, d_C, nElem, offset);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    printf("readOffsetunroll <<< %4d, %4d >>> offset %4d elpsed %f sec\n", gridunroll.x, blockunroll.x, offset, iElaps);

    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
    checkResult(hostRef, gpuRef, nElem-offset);

    free(h_A);
    free(h_B);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaDeviceReset();
    return EXIT_SUCCESS;
}
```

使用不同的偏移量进行试验。

```c
 $ ./readSegment 0
 readOffset <<< 32768,  512 >>> offset    0 elapsed 0.001820 sec
 $ ./readSegment 11
 readOffset <<< 32768,  512 >>> offset   11 elapsed 0.001949 sec
 $ ./readSegment 128
 readOffset <<< 32768,  512 >>> offset  128 elapsed 0.001821 sec
```

可以看到使用值为11的偏移量会导致数组A和数组B的内存加载是非对齐的。在这种情况下， 运行时间也是最慢的。

### 全局内存写入

下图所示为理想情况：内存访问是对齐的，并且线程束里所有的线程访问一个连续 的128字节范围。存储请求由一个四段事务实现。

<figure><img src="../../.gitbook/assets/图片 (11) (1).png" alt=""><figcaption></figcaption></figure>

下图所示为内存访问是对齐的，但地址分散在一个192个字节范围内的情况。存储 请求由3个一段事务来实现。

<figure><img src="../../.gitbook/assets/图片 (12) (1).png" alt=""><figcaption></figcaption></figure>

下图所示为内存访问是对齐的，并且地址访问在一个连续的64个字节范围内的情 况。这种存储请求由一个两段事务来完成。

<figure><img src="../../.gitbook/assets/图片 (13).png" alt=""><figcaption></figcaption></figure>

### 非对齐写入的示例&#x20;

为了验证非对齐对内存存储效率的影响，按照下面的方式修改向量加法核函数。仍然 使用两个不同的索引：索引k根据给定的偏移量进行变化，而索引i不变（并因此产生对齐 访问）。使用对齐索引i从数组A和数组B中进行加载，以产生良好的内存加载效率。使用 偏移量索引x写入数组C，可能会造成非对齐写入，这取决于偏移量的值。

```c
 __global__ void writeOffset(float *A, float *B, float *C, 
      const int n, int offset) {
   unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
   unsigned int k = i + offset;
   if (k < n) C[k] = A[i] + B[i];
 }
```

程序输出结果如下：

```bash
 $ ./writeSegment 0
 writeOffset <<< 2048,  512 >>> offset    0 elapsed 0.000134 sec
 $ ./writeSegment 11
 writeOffset <<< 2048,  512 >>> offset   11 elapsed 0.000184 sec
 $ ./writeSegment 128
 writeOffset <<< 2048,  512 >>> offset  128 elapsed 0.000134 sec
```

### 结构体数组与数组结构体

下面是存储成对的浮点数据元素集的例子。首先，考虑这些成对数据元素集如何使用 AoS方法进行存储。如下定义一个结构体，命名为innerStruct：

```c
struct innerStruct {
   float x;
   float y;
 };
```

然后，按照下面的方法定义这些结构体数组。这是利用AoS方式来组织数据的。它存 储的是空间上相邻的数据（例如，x和y），这在CPU上会有良好的**缓存局部性(cache locality)**。

```c
 struct innerStruct myAoS[N];
```

接下来，考虑使用SoA方法来存储数据：

```c
struct innerArray {
   float x[N];
   float y[N];
 };
```

这里，在原结构体中每个字段的所有值都被分到各自的数组中。这不仅能将相邻数据 点紧密存储起来，也能将跨数组的独立数据点存储起来。你可以使用如下结构体定义一个 变量：

```c
struct innerArray moa;
```

下图说明了AoS和SoA方法的内存布局。用AoS模式在GPU上存储示例数据并执行一个只有x字段的应用程序，将导致50%的带宽损失，因为y值在每32个字节段或128个字节 缓存行上隐式地被加载。AoS格式也在不需要的y值上浪费了二级缓存空间。&#x20;

用SoA模式存储数据充分利用了GPU的内存带宽。由于没有相同字段元素的交叉存 取，GPU上的SoA布局提供了合并内存访问，并且可以对全局内存实现更高效的利用。

<figure><img src="../../.gitbook/assets/图片 (14).png" alt=""><figcaption></figcaption></figure>

#### AOS与SOA&#x20;

许多并行编程范式，尤其是SIMD型范式，更倾向于使用SoA。在CUDA C编程中也普 遍倾向于使用SoA，因为数据元素是为全局内存的有效合并访问而预先准备好的，而被相 同内存操作引用的同字段数据元素在存储时是彼此相邻的。&#x20;

在每种数据布局中为了帮助理解访问数据对性能的影响，我们将通过执行相同的数学 运算来比较两个核函数：一个使用AoS数据布局，另一个则使用SoA数据布局。

### 示例：使用AoS数据布局的简单数学运算

```c
#include <stdio.h>
#define LEN 1<<20

#include <time.h>
#include <sys/time.h>

struct innerStruct
{
    int x;
    int y;
};

// struct innerStruct myAoS[N];

//sudo nvprof --devices 0 --metrics gld_efficiency,gst_efficiency ./simpleMathAoS



void initialInnerStruct(innerStruct* A, size_t N){
    time_t t;
    srand((unsigned int) time(&t));
    for(int i=0;i<N;i++){
        A[i].x = (float)( rand() & 0xFF)/10.0f;
        A[i].y = (float)( rand() & 0xFF)/10.0f;
    }
}

void testInnerStructHost(innerStruct* A, innerStruct* B, size_t N){
    for(int i=0;i<N;i++){
        innerStruct tmp = A[i];
        tmp.x += 10.f;
        tmp.y += 20.f;
        B[i] = tmp;
    }
}

__global__ void testInnerStruct(innerStruct *data, innerStruct *result, int N){
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < N){
        innerStruct tmp = data[i];
        tmp.x += 10.f;
        tmp.y += 20.f;
        result[i] = tmp;
    }
}


double seconds(){
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

void checkInnerStruct(innerStruct *hostRef, innerStruct *gpuRef, const int N){
    double eplison = 1.0E-5;
    int match = 1;
    for(int i=0;i<N;i++){
        if(abs(hostRef[i].x-gpuRef[i].x)>eplison){
            match = 0;
            printf("do not match\n");
            break;
        }
        if(abs(hostRef[i].y-gpuRef[i].y)>eplison){
            match = 0;
            printf("do not match\n");
            break;
        }
    }

    if(match) printf("match!\n");
    return;

}

int main(int argc, char** argv){
    int dev = 0;
    cudaSetDevice(dev);

    cudaDeviceProp DeviceProp;
    cudaGetDeviceProperties(&DeviceProp, dev);

    printf("device %d: %s starting...\n", dev, DeviceProp.name);
    
    int nElem = LEN;
    size_t nBytes = nElem * sizeof(innerStruct);
    innerStruct *h_A = (innerStruct *)malloc(nBytes);
    innerStruct *hostRef = (innerStruct *)malloc(nBytes);
    innerStruct *gpuRef = (innerStruct *) malloc(nBytes);

    initialInnerStruct(h_A, nElem);
    testInnerStructHost(h_A, hostRef, nElem);

    innerStruct *d_A, *d_C;
    cudaMalloc((innerStruct**)&d_A, nBytes);
    cudaMalloc((innerStruct**)&d_C, nBytes);

    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);

    int blocksize = 128;
    if(argc>1) blocksize = atoi(argv[1]);

    dim3 block (blocksize, 1);
    dim3 grid((nElem + block.x -1)/ block.x, 1);

    double iStart = seconds();
    testInnerStruct<<<grid, block>>>(d_A, d_C, nElem);
    cudaDeviceSynchronize();
    double iElaps = seconds() - iStart;
    printf("innerStruct <<< %4d, %4d >>> elpsed %f sec\n", grid.x, block.x, iElaps);
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
    checkInnerStruct(hostRef, gpuRef, nElem);

    free(h_A);
    free(hostRef);
    free(gpuRef);
    cudaFree(d_A);
    cudaFree(d_C);

    cudaDeviceReset();
    return EXIT_SUCCESS;

}
```

对于AOS数据布局，加载请求和内存存储请求是重 复的。因为字段x和y在内存中是被相邻存储的，并且有相同的大小，每当执行内存事务时 都要加载特定字段的值，被加载的字节数的一半也必须属于其他字段。因此，请求加载和 存储的50%带宽是未使用的。

### 示例：使用SoA数据布局的简单数学运算

```c
#include <stdio.h>
#define LEN 1<<4

#include <time.h>
#include <sys/time.h>

struct innerArray
{
    float x[LEN];
    float y[LEN];
};

void initialinnerArray(innerArray* A, size_t N){
    time_t t;
    srand((unsigned int) time(&t));
    for(int i=0;i<N;i++){
        A->x[i] = (float)( rand() & 0xFF)/10.0f;
        A->y[i] = (float)( rand() & 0xFF)/10.0f;
    }
}

void testinnerArrayHost(innerArray* data, innerArray* result, size_t N){
    for(int i=0;i<N;i++){
        float tmpx = data->x[i];
        float tmpy = data->y[i];

        tmpx += 10.f;
        tmpy += 20.f;
        result->x[i] = tmpx;
        result->y[i] = tmpy;
    }
}

__global__ void testinnerArray(innerArray *data, innerArray *result, int N){
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < N){
        float tmpx = data->x[i];
        float tmpy = data->y[i];

        tmpx += 10.f;
        tmpy += 20.f;
        result->x[i] = tmpx;
        result->y[i] = tmpy;
    }
}


double seconds(){
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

void checkinnerArray(innerArray *hostRef, innerArray *gpuRef, const int N){
    double eplison = 1.0E-5;
    int match = 1;
    for(int i=0;i<N;i++){
        if(abs(hostRef->x[i]-gpuRef->x[i])>eplison){
            match = 0;
            printf("do not match\n");
            break;
        }
        if(abs(hostRef->y[i]-gpuRef->y[i])>eplison){
            match = 0;
            printf("do not match\n");
            break;
        }
    }

    if(match) printf("match!\n");
    return;

}

int main(int argc, char** argv){
    int dev = 0;
    cudaSetDevice(dev);

    cudaDeviceProp DeviceProp;
    cudaGetDeviceProperties(&DeviceProp, dev);

    printf("device %d: %s starting...\n", dev, DeviceProp.name);
    
    int nElem = LEN;
    printf("%d\n",nElem);
    unsigned int nBytes = nElem * sizeof(innerArray);
    printf("%d\n",sizeof(innerArray));
    printf("%d\n",nBytes);
    innerArray *h_A = (innerArray *)malloc(nBytes);
    innerArray *hostRef = (innerArray *)malloc(nBytes);
    innerArray *gpuRef = (innerArray *) malloc(nBytes);

    initialinnerArray(h_A, nElem);
    testinnerArrayHost(h_A, hostRef, nElem);

    innerArray *d_A, *d_C;
    cudaMalloc((innerArray**)&d_A, nBytes);
    cudaMalloc((innerArray**)&d_C, nBytes);

    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);

    int blocksize = 128;
    if(argc>1) blocksize = atoi(argv[1]);

    dim3 block (blocksize, 1);
    dim3 grid((nElem + block.x -1)/ block.x, 1);

    double iStart = seconds();
    testinnerArray<<<grid, block>>>(d_A, d_C, nElem);
    cudaDeviceSynchronize();
    double iElaps = seconds() - iStart;
    printf("innerArray <<< %4d, %4d >>> elpsed %f sec\n", grid.x, block.x, iElaps);
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
    checkinnerArray(hostRef, gpuRef, nElem);

    free(h_A);
    free(hostRef);
    free(gpuRef);
    cudaFree(d_A);
    cudaFree(d_C);

    cudaDeviceReset();
    return EXIT_SUCCESS;

}
```

与AoS 相比，其性能略有提升。使用更大的输入长度，性能提升将更为明显。
