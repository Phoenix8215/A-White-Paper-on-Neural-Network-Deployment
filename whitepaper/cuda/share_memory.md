# 🫢 Share Memory|共享内存|Bank Conflicts

共享内存在 CUDA 内存层次结构中一直扮演着重要角色，被称为用户管理缓存（User-Managed Cache）。

### 共享内存和CPU缓存的区别

共享内存具有与 CPU 缓存类似的优势；<mark style="color:red;">不过，CPU 缓存无法进行显式管理，而共享内存则可以。共享内存的延迟比全局内存低一个数量级，带宽比全局内存高一个数量级。</mark>但共享内存的主要用途来自于同一个block的线程可以共享内存访问。CUDA 程序员可以使用共享变量来保存在内核执行阶段被多次重复使用的数据。此外，由于同一block内的线程可以共享结果，这有助于避免冗余计算。

### 用共享内存进行矩阵转置

以下是基于主机实现的使用单精度浮点值的错位转置算法。假设矩阵存储在一个一维 数组中。通过改变数组索引值来交换行和列的坐标，可以很容易得到转置矩阵。

```c
 void transposeHost(float *out, float *in, const int nx, const int ny) {
   for (int iy = 0; iy < ny; ++iy) {
      for (int ix = 0; ix < nx; ++ix) {
         out[ix*ny+iy] = in[iy*nx+ix];
      }
   }
 }
```

<figure><img src="../../.gitbook/assets/图片 (1).png" alt=""><figcaption></figcaption></figure>

在这个函数中有两个用一维数组存储的矩阵：输入矩阵in和转置矩阵out。矩阵维度被 定义为nx行ny列。可以用一个一维数组执行转置操作:

<figure><img src="../../.gitbook/assets/图片 (1) (1).png" alt=""><figcaption></figcaption></figure>

观察输入和输出布局，你会注意到：

* 读：通过原矩阵的行进行访问，结果为合并访问&#x20;
* 写：通过转置矩阵的列进行访问，结果为交叉访问&#x20;

交叉访问是使GPU性能变得最差的内存访问模式。但是，在矩阵转置操作中这是不可 避免的。这里使用两种转置核函数来提高带宽的利用率：<mark style="color:red;">一种是按行 读取按列存储，另一种则是按列读取按行存储。</mark>

<mark style="color:red;">下图分别展示了这两种方法。你能预测一下这两种实现的相 对性能吗？如果禁用一级缓存加载，那么这两种实现的性能在理论上是相同的。但是，如果启用一级缓存，那么第二种实现的性能表现会更好。按列读取操作是不合并的（因此带 宽将会浪费在未被请求的字节上），将这些额外的字节存入一级缓存意味着下一个读操作 可能会在缓存上执行而不在全局内存上执行。因为写操作不在一级缓存中缓存，所以对按列执行写操作的例子而言，任何缓存都没有意义。</mark>

<figure><img src="../../.gitbook/assets/图片 (2).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/图片 (3).png" alt=""><figcaption></figcaption></figure>

下面的代码片段使用了 matrix\_transpose\_naive 核函数，展示了矩阵转置的实现示例：

```c
__global__ void matrix_transpose_naive(int *input, int *output) {

	int indexX = threadIdx.x + blockIdx.x * blockDim.x;
	int indexY = threadIdx.y + blockIdx.y * blockDim.y;
	int index = indexY * N + indexX;
	int transposedIndex = indexX * N + indexY;

	output[transposedIndex] = input[index];
	
	// output[index] = input[transposedIndex];
}
```

解决这一问题的方法之一是利用高带宽、低延迟的内存，如共享内存。这里的诀窍是以合并方式从全局内存读写。使用共享内存可以提高性能，时间缩短到 21 微秒，速度提高了 3 倍：

```c
__global__ void matrix_transpose_shared(int *input, int *output) {

	// 每个线程块都有一个 BLOCK_SIZE ✕ BLOCK_SIZE 的共享内存数组，用于存储其内部处理的数据
	__shared__ int sharedMemory [BLOCK_SIZE] [BLOCK_SIZE];

	// global index	
	int indexX = threadIdx.x + blockIdx.x * blockDim.x;
	int indexY = threadIdx.y + blockIdx.y * blockDim.y;

	// transposed global memory index
	int tindexX = threadIdx.x + blockIdx.y * blockDim.x;
	int tindexY = threadIdx.y + blockIdx.x * blockDim.y;

	// local index
	int localIndexX = threadIdx.x;
	int localIndexY = threadIdx.y;

	int index = indexY * N + indexX;
	int transposedIndex = tindexY * N + tindexX;

	// reading from global memory in coalesed manner and performing tanspose in shared memory
	sharedMemory[localIndexX][localIndexY] = input[index];

	__syncthreads();

	// writing into global memory in coalesed fashion via transposed data in shared memory
	output[transposedIndex] = sharedMemory[localIndexY][localIndexX];
}
```

### Bank conflicts对共享内存的影响

通过Visual Profiler其实可以分析出上述的核函数存在内存访问没有对齐的问题：

<figure><img src="../../.gitbook/assets/图片 (174).png" alt=""><figcaption></figcaption></figure>

为了获得高内存带宽，共享内存被分为32个同样大小的内存模型，它们被称为存储 体(Bank)，它们可以被同时访问。有32个存储体是因为在一个线程束中有32个线程。共享内存是 一个一维地址空间。如果通过线程束发布共享内存加载或存储操作，且在每个存储体 上只访问不多于一个的内存地址，那么该操作可由一个内存事务来完成。否则，该操作由 多个内存事务来完成，这样就降低了内存带宽的利用率。

Volta GPU 有 32 个存储 体，每个存储 体 4 字节宽。如下图所示：

<figure><img src="../../.gitbook/assets/图片 (4).png" alt=""><figcaption></figcaption></figure>

{% hint style="info" %}
bank的宽度，代表的是一个bank所存储的数据的大小宽度。&#x20;

* 可以是4个字节(32 bit, 单精度浮点数)&#x20;
* 也可以是8个字节(64bit，双精度浮点数)
{% endhint %}

每31个bank，就会进行一次stride。 比如说bank的宽度是4字节。我们在`shared memory`中申请了`float A[256]`大小的空间，那么 A\[0], A\[1], …, A\[31]分别在bank0, bank1, …, bank31中 A\[32], A\[33], …, A\[63]也分在了bank0, bank1, …, bank31中 所以A\[0], A\[32]是共享同一个bank的！

一个很理想的情况就是，32个thread，分别访问shared memory中的32个不同的bank 没有bank conflict，<mark style="color:red;">一个memory周期完成所有的memory read/write (row major/行优先矩阵访问)</mark>

<figure><img src="../../.gitbook/assets/图片 (8).png" alt=""><figcaption></figcaption></figure>

下图显示了最优的并行访问模式。每个线程访问一个32位字。因为每个线程访问不 同存储体中的地址，所以没有存储体冲突：

<figure><img src="../../.gitbook/assets/图片 (5).png" alt=""><figcaption></figcaption></figure>

下图显示了不规则的随机访问模式。因为每 个线程访问不同的存储体，所以也没有存储体冲突：

<figure><img src="../../.gitbook/assets/图片 (6).png" alt=""><figcaption></figcaption></figure>

一个最不理想的情况就是，32个thread，访问shared memory中的同一个bank ，`bank conflict`最大化，需要32个memory周期才能完成所有的memory read/write (`column major`列优先矩阵访问)

<figure><img src="../../.gitbook/assets/图片 (9).png" alt=""><figcaption></figcaption></figure>

下图显示了另一种不规则的访问模 式，在这里两个线程(一个warp中)访问同一存储体。对于这样一个请求，会产生<mark style="color:red;">2路存储体冲突</mark>：

<figure><img src="../../.gitbook/assets/图片 (7).png" alt=""><figcaption></figcaption></figure>

### 使用padding缓解bank conflict

在前面的矩阵转置示例中，我们利用共享内存获得了更好的性能。不过，我们可以看到 <mark style="color:red;">32 路存储体冲突</mark>。为了解决这个问题，我们可以使用一种称为 "填充 "(padding)的简单技术。这样做的目的是在共享内存中填充一个虚拟列，即增加一列，使线程访问不同的存储体，从而提高性能：

<figure><img src="../../.gitbook/assets/图片 (10).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/图片 (11).png" alt="" width="375"><figcaption></figcaption></figure>

> **为了方便解释，这里使用了8个bank一次stride进行举例。 在实际CUDA设计时，依然是32个bank一次stride!!!**

```c
__global__ void matrix_transpose_shared(int *input, int *output) {

	//这里进行padding操作，其余部分和上面案例代码一样的
	__shared__ int sharedMemory [BLOCK_SIZE] [BLOCK_SIZE + 1];

	int indexX = threadIdx.x + blockIdx.x * blockDim.x;
	int indexY = threadIdx.y + blockIdx.y * blockDim.y;

	int tindexX = threadIdx.x + blockIdx.y * blockDim.x;
	int tindexY = threadIdx.y + blockIdx.x * blockDim.y;

	int localIndexX = threadIdx.x;
	int localIndexY = threadIdx.y;

	int index = indexY * N + indexX;
	int transposedIndex = tindexY * N + tindexX;

	sharedMemory[localIndexX][localIndexY] = input[index];

	__syncthreads();

	output[transposedIndex] = sharedMemory[localIndexY][localIndexX];
}
```

和之前的操作一样，运行代码并借助 Visual Profiler 进行profile。通过这些更改，内核运行时间缩短至 13 微秒，速度进一步提高了 60%。

### reference

* [https://homepages.math.uic.edu/%7Ejan/mcs572f16/mcs572notes/lec35.html](https://homepages.math.uic.edu/\~jan/mcs572f16/mcs572notes/lec35.html)
* [https://s3.amazonaws.com/ebooks.syncfusion.com/downloads/CUDA\_Succinctly/CUDA\_Succinctly.pdf?AWSAccessKeyId=AKIAWH6GYCX3XWVCZEUI\&Expires=1716294471\&Signature=RGeLIPbR%2Byd8TD5WTABuETUKLnk%3D](https://s3.amazonaws.com/ebooks.syncfusion.com/downloads/CUDA\_Succinctly/CUDA\_Succinctly.pdf?AWSAccessKeyId=AKIAWH6GYCX3XWVCZEUI\&Expires=1716294471\&Signature=RGeLIPbR%2Byd8TD5WTABuETUKLnk%3D)
