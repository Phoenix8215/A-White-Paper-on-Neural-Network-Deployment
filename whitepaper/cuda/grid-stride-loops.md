---
description: >-
  翻译自:https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
---

# 🤫 Grid-Stride Loops

在 CUDA 编程中，最常见的任务之一就是使用内核将循环并行化。以我们熟悉的 SAXPY(`Single-Precision A·X Plus Y`) 为例，以下是一个使用 `for` 循环的基本顺序实现。为了有效地并行化这个过程，我们需要启动足够多的线程来充分利用 GPU。

```cpp
void saxpy(int n, float a, float *x, float *y)
{
    for (int i = 0; i < n; ++i)
        y[i] = a * x[i] + y[i];
}
```

在 CUDA 编程中，通常的建议是为每个数据元素启动一个线程，这意味着要并行化上面的 SAXPY 循环，我们需要编写一个假设线程数量足以处理整个数组的内核。

```cpp
__global__
void saxpy(int n, float a, float *x, float *y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) 
        y[i] = a * x[i] + y[i];
}
```

<mark style="color:red;">这种内核风格被称为“单片内核”(</mark>**monolithic kernel**<mark style="color:red;">)，因为它假设有一个大的线程网格能够一次性处理整个数组。</mark>你可以通过以下代码启动 saxpy 内核来处理一百万个元素。

```cpp
// 处理 100 万个元素的 SAXPY
saxpy<<<4096,256>>>(1<<20, 2.0, x, y);
```

然而，在并行化计算时，不建议完全消除循环，而是建议使用网格步长循环，如下所示。

```cpp
__global__
void saxpy(int n, float a, float *x, float *y)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < n; 
         i += blockDim.x * gridDim.x) 
      {
          y[i] = a * x[i] + y[i];
      }
}
```

不同于假设线程网格足够大以覆盖整个数据数组，这种内核通过循环逐步遍历数据数组，每次按网格大小处理一部分数据。

<mark style="color:red;">注意，这里的循环步长是</mark> <mark style="color:red;"></mark><mark style="color:red;">`blockDim.x * gridDim.x`</mark><mark style="color:red;">，也就是网格中的总线程数。因此，如果网格中有 1280 个线程，线程 0 将处理元素 0、1280、2560 等。这就是为什么称其为“网格步长循环”。通过使用步长等于网格大小的循环，我们确保 warp 内的寻址是连续的，从而实现与单片内核相似的最大内存合并效果。</mark>

当网格足够大以覆盖所有循环迭代时，网格步长循环在指令开销上与单片内核中的 `if` 语句基本相同，因为只有在循环条件为真的情况下，循环增量才会被计算。

使用网格步长循环有几个明显的优势。

1. **可扩展性与线程重用**：通过循环，你可以支持任何规模的问题，即使它超过了 CUDA 设备的最大网格大小。此外，你还可以通过限制块的数量来优化性能。例如，通常会

将块的数量设置为设备上多处理器数量的倍数，以实现均衡的利用率。举个例子，我们可以这样启动循环版本的内核：

```cpp
int numSMs;
cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, devId);
// 处理 100 万个元素的 SAXPY
saxpy<<<32*numSMs, 256>>>(1 << 20, 2.0, x, y);
```

当你限制网格中的块数量时，线程会被重用于多次计算。线程重用可以分摊线程创建和销毁的成本，以及内核在循环前后可能执行的任何初始化操作（如线程私有数据或共享数据的初始化）。

2. **调试**：使用循环而不是单片内核，可以更轻松地切换到串行处理，只需启动一个块和一个线程即可。

```cpp
saxpy<<<1,1>>>(1<<20, 2.0, x, y);
```

这种方式使得在模拟串行主机实现时更容易验证结果，也让 `printf` 调试变得更为简单，通过串行化计算消除运行顺序的变化导致的数值差异，从而帮助你在优化并行版本前验证数值的正确性。

3. **可移植性与可读性**：网格步长循环代码与原始顺序循环代码更为相似，因此对其他用户来说更加易于理解。事实上，我们可以轻松地编写一个内核版本，它可以在 GPU 上作为并行 CUDA 内核运行，也可以在 CPU 上作为顺序循环运行。 Hemi 库提供了一个 `grid_stride_range()` 辅助工具，通过 C++11 的基于范围的 `for` 循环使这一过程变得非常简单。

```cpp
HEMI_LAUNCHABLE
void saxpy(int n, float a, float *x, float *y)
{
  for (auto i : hemi::grid_stride_range(0, n)) {
    y[i] = a * x[i] + y[i];
  }
}
```

我们可以使用以下代码启动内核，当编译为 CUDA 时，它将生成一个内核启动，而当编译为 CPU 时，它将生成一个函数调用。

```cpp
hemi::cudaLaunch(saxpy, 1<<20, 2.0, x, y);
```

网格步长循环是让你的 CUDA 内核更具灵活性、可扩展性、易调试性，甚至可移植性的极佳方式。尽管本文中的例子都使用了 CUDA C/C++，但同样的概念也适用于其他 CUDA 语言，如 CUDA Fortran。

### 示例代码

```cpp
#include <stdio.h>

void init(int *a, int N) {
  int i;
  for (i = 0; i < N; ++i)
  {
    a[i] = i;
  }
}

__global__ void doubleElements(int *a, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  for (int i = idx; i < N; i += stride) {
    a[i] *= 2;
  }
}

bool checkElementsAreDoubled(int *a, int N) {
  int i;
  for (i = 0; i < N; ++i) {
    if (a[i] != i*2) return false;
  }
  return true;
}


int main() {
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
  return 0;
}
```
