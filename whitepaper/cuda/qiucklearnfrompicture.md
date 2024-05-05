# QiuckLearnFromPicture

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













