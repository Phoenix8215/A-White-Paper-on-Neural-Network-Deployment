# 🐻 TensorRT如何工作

本章提供了有关 TensorRT 工作原理的更多详细信息。

## Object Lifetimes

TensorRT 的 API 是基于类的，其中一些类充当其他类的工厂。对于用户拥有的对象，工厂对象的生命周期必须跨越它创建的对象的生命周期。例如， `NetworkDefinition`和`BuilderConfig`类是从`Builder`类创建的，这些类的对象应该在`Builder`工厂对象之前销毁。

{% hint style="info" %}
此规则的一个例外是从`Builder`创建`Engine`。创建`Engine`后，您可以销毁`Builder`、`Network`、解析器和`config`并继续使用`Engine`。
{% endhint %}

## Error Handling and Logging

创建 TensorRT 接口（`builder`、`runtime` 或 `refitter`）时，必须提供`Logger` （ [C++](https://docs.nvidia.com/deeplearning/tensorrt/api/c\_api/classnvinfer1\_1\_1\_i\_logger.html) 、 [Python](https://docs.nvidia.com/deeplearning/tensorrt/api/python\_api/infer/Core/Logger.html) ）接口的实现。`Logger`用于日志记录；它的详细程度是可配置的。由于`Logger`可用于在 TensorRT 生命周期的任何时间点传回信息，因此它的生命周期必须跨越应用程序，实现也必须是线程安全的，因为 TensorRT 可以在内部使用线程。

对对象的 API 调用将使用与相应接口关联的`Logger`。例如，在对`ExecutionContext::enqueueV3()`的调用中，执行上下文是从引擎创建的，该引擎是从运行时创建的，因此 TensorRT 将使用与该运行时关联的记录器。

错误处理的主要方法是`ErrorRecord` ( [C++ ](https://docs.nvidia.com/deeplearning/tensorrt/api/c\_api/classnvinfer1\_1\_1\_i\_error\_recorder.html), [Python](https://docs.nvidia.com/deeplearning/tensorrt/api/python\_api/infer/Core/ErrorRecorder.html) ) 接口。可以实现此接口，并将其附加到 API 对象以接收与该对象关联的错误。对象的记录器也将传递给它创建的任何其他记录器 - 例如，如果将错误记录器附加到引擎，并从该引擎创建执行上下文，它将使用相同的记录器。如果随后将新的错误记录器附加到执行上下文，它将仅接收来自该上下文的错误。如果生成错误但没有找到错误记录器，它将通过关联的记录器发出。

<mark style="color:red;">请注意，CUDA 错误通常是</mark><mark style="color:red;">**异步**</mark><mark style="color:red;">的 - 因此，当执行多个推理或其他 CUDA 流在单个 CUDA 上下文中异步工作时，可能会在与生成它的执行上下文不同的上下文中观察到异步 GPU 错误。</mark>

## Memory

TensorRT 使用大量设备内存，即 GPU 可直接访问的内存，而不是连接到 CPU 的主机内存。由于设备内存通常是一种受限资源，因此了解 TensorRT 如何使用它很重要。

### The Build Phase

在构建期间，TensorRT 为时序层实现分配设备内存。一些实现可能会消耗大量临时内存，尤其是在使用大张量的情况下。可以通过构建器的`maxWorkspace`属性控制最大临时内存量。这<mark style="color:red;">默认为设备全局内存的大小</mark>，但可以在必要时进行限制。如果构建器发现由于工作空间不足而无法运行内核，它将发出一条日志消息来指示这一点。

然而，即使工作空间相对较小，时序层也需要为输入、输出和权重创建缓冲区。 TensorRT 对操作系统因此类分配而返回内存不足是稳健的，但在某些平台上，操作系统可能会成功提供内存，随后如果`killer`进程观察到系统内存不足，会终止 TensorRT 进程。

<mark style="color:red;">在构建阶段，通常在主机内存中至少有两个权重拷贝：来自原始网络的权重拷贝，以及在构建引擎时作为引擎一部分的权重拷贝。此外，当 TensorRT 组合权重（例如</mark><mark style="color:red;">`convolution`</mark><mark style="color:red;">与</mark><mark style="color:red;">`batch normalization`</mark><mark style="color:red;">）时，将创建额外的临时权重张量。</mark>

### The Runtime Phase

在运行时，TensorRT 使用相对较少的主机内存，但可以使用大量的设备内存。

<mark style="color:red;">引擎在反序列化时分配设备内存来存储模型权重。由于序列化引擎几乎都是权重，因此它的大小非常接近权重所需的设备内存量。</mark>

`ExecutionContext`使用两种设备内存：

* 在某些深度学习模型的层实现中，需要使用持久性内存。例如，一些卷积层的实现会使用边缘掩码（edge masks），这种状态信息不能像权重那样在不同的上下文（contexts）之间共享。这是因为边缘掩码的大小取决于层的输入形状（layer input shape），而这个输入形状在不同的上下文中可能会变化。因此，当创建一个执行上下文时，就会分配这种持久性内存，并且它会一直存在，直到执行上下文的生命周期结束。
* 临时内存，它被用来在处理网络时保存中间结果。这种内存用于存储中间激活张量（intermediate activation tensors），也用于存储层实现所需的临时数据。临时内存的使用界限是由`IBuilderConfig::setMemoryPoolLimit()`这个接口控制的。这意味着，开发者可以通过这个接口来设置临时内存的上限，以避免内存使用过多导致的问题。

可以选择通过`ICudaEngine::createExecutionContextWithoutDeviceMemory()`创建一个没有暂存内存的执行上下文，并在网络执行期间自行提供该内存。这允许你在未同时运行的多个上下文之间共享它，或者在推理未运行时用于其他用途。 `ICudaEngine::getDeviceMemorySize()`返回所需的暂存内存量。

构建器在构建网络时发出有关执行上下文使用的持久内存和暂存内存量的信息，日志级别为 `kINFO` 。检查日志，消息类似于以下内容：

```cpp
[08/12/2021-17:39:11] [I] [TRT] Total Host Persistent Memory: 106528
[08/12/2021-17:39:11] [I] [TRT] Total Device Persistent Memory: 29785600
[08/12/2021-17:39:11] [I] [TRT] Total Scratch Memory: 9970688
```

默认情况下，TensorRT 直接从 CUDA 分配设备内存。不过，你可以将 TensorRT 的 `IGpuAllocator`（[C++](https://docs.nvidia.com/deeplearning/tensorrt/api/c\_api/classnvinfer1\_1\_1\_i\_gpu\_allocator.html)、[Python](https://docs.nvidia.com/deeplearning/tensorrt/api/python\_api/infer/Core/GpuAllocator.html)）接口的实现附加到Builder或Runtime，然后自己管理设备内存。如果你的应用程序希望控制所有 GPU 内存并将一部分内存分配给 TensorRT，那么这将非常有用。

英伟达 [cuDNN ](https://developer.nvidia.com/cudnn)和英伟达 [cuBLAS ](https://developer.nvidia.com/cublas)会占用大量设备内存。TensorRT 允许通过构建器配置中的 `TacticSources`（[C++](https://docs.nvidia.com/deeplearning/tensorrt/api/c\_api/namespacenvinfer1.html#a999ab7be02c9acfec0b2c9cc3673abb4)、[Python](https://docs.nvidia.com/deeplearning/tensorrt/api/python\_api/infer/Core/BuilderConfig.html?highlight=tactic\_sources#tensorrt.IBuilderConfig.set\_tactic\_sources)）属性来控制是否使用这些库进行推理。某些插件实现需要这些库，因此在排除这些库时，可能无法成功编译网络。如果设置了相应的策略源，则会使用 `IPluginV2Ext::attachToContext()` 将 `cudnnContext` 和 `cublasContext` 的句柄传递给插件。

CUDA 基础架构和 TensorRT 的设备代码也会消耗设备内存。内存量因平台、设备和 TensorRT 版本而异。你可以使用 `cudaGetMemInfo` 来确定正在使用的设备内存总量。

TensorRT 会在`Builder`和`Runtime`测量关键操作前后的内存使用量。这些内存使用统计数据会打印到 TensorRT 的信息记录器中。例如

```bash
[MemUsageChange] Init CUDA: CPU +535, GPU +0, now: CPU 547, GPU 1293 (MiB)
```

它表示 CUDA 初始化后内存使用量的变化。CPU +535、GPU +0 是运行 CUDA 初始化后增加的内存量。now 之后的内容：是 CUDA 初始化后的 CPU/GPU 内存使用快照。

**注意：**在多用户情况下，`cudaGetMemInfo` 和 TensorRT 报告的内存使用情况容易出现竞争条件，即由不同进程或不同线程完成新的分配/释放。由于 CUDA 无法控制统一内存设备上的内存，因此 `cudaGetMemInfo` 在这些平台上返回的结果可能并不准确。

### CUDA Lazy Loading

CUDA lazy loading 是一项 CUDA 功能，可显著降低 TensorRT 的 GPU 和主机内存使用峰值，并加快 TensorRT 的初始化，对性能的影响可忽略不计（< 1%）。内存使用量和初始化时间的节省取决于模型、软件栈、GPU 平台等。可通过设置环境变量 `CUDA_MODULE_LOADING=LAZY`启用。更多信息请参[CUDA 文档](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#lazy-loading)。

#### L2 Persistent Cache Management

英伟达™（NVIDIA®）Ampere 及更高版本的架构支持持久性二级缓存，该功能允许在选择要删除的行时优先保留二级缓存行。TensorRT 可以利用这一功能将激活值保留在缓存中，从而减少 DRAM 流量和功耗。

缓存分配按执行上下文进行，使用上下文的 `setPersistentCacheLimit` 方法启用。所有上下文（以及使用此功能的其他组件）中的持久缓存总量不应超过 `cudaDeviceProp::persingL2CacheMaxSize`。更多信息，请参阅[《英伟达 CUDA 最佳实践指南》](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)。

## Threading

<mark style="color:red;">一般来说，TensorRT 对象不是线程安全的。</mark>预期的运行时并发模型是不同的线程将在不同的执行上下文上操作。上下文包含执行期间的网络状态（激活值等），因此在不同线程中同时使用上下文会导致未定义的行为。 <mark style="color:red;">以下操作是线程安全的：</mark>

* runtime或engine上的非修改操作。
* 从 `TensorRT runtime`反序列化引擎。
* 从引擎创建执行上下文。
* 注册和注销插件。

在不同线程中使用多个构建器没有线程安全问题；但是，构建器使用时序来确定所提供参数的最快内核，并且使用具有相同 GPU 的多个构建器将扰乱时序和 TensorRT 构建最佳引擎的能力。

## Determinism

TensorRT `builder` 根据使用时间来找到最快的内核来实现给定的算子。时序内核会受到噪声(GPU 上运行的其他工作、GPU 时钟速度的波动等)的影响。时序噪声意味着在构建器的连续运行中，可能不会选择相同的实现。

`AlgorithmSelector` ( [C++](https://docs.nvidia.com/deeplearning/tensorrt/api/c\_api/classnvinfer1\_1\_1\_i\_algorithm\_selector.html) , [Python](https://docs.nvidia.com/deeplearning/tensorrt/api/python\_api/infer/AlgorithmSelector/pyAlgorithmSelector.html) )接口允许您强制构建器为给定层选择特定实现。可以使用它来确保构建器一定选择相同的内核。有关更多信息，请参阅[算法选择和可重现构建](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#algorithm-select)部分。

<mark style="color:red;">一旦构建了引擎，它就是确定性的：在相同的运行时环境中提供相同的输入将产生相同的输出。</mark>

## <mark style="color:red;">Runtime Options</mark>

TensorRT 提供多个运行时库，以满足各种使用情况。运行 TensorRT 引擎的 C++ 应用程序应链接到以下动态库中的一个：

* _默认_运行时为主库（`libnvinfer.so/.dll`）。
* _精简_运行时库（`libnvinfer_lean.so/.dll`）比默认库小得多，只包含运行版本兼容引擎所需的代码。它有一些限制，主要是不能重新适配或序列化引擎。
* _调度_运行时（`libnvinfer_dispatch.so/.dll`）是一个小型临时库，可以加载_精简_运行时，并对其进行重定向调用。_调度_运行时能加载旧版本的_精简_运行时，并对`builder`进行适当的配置，<mark style="color:red;">可用于兼容较新版本的 TensorRT 和较旧版本的plan文件。</mark>使用_调度_运行时与手动加载_精简_运行时几乎相同，但它会检查所加载的_精简_运行时是否实现了应用程序接口，并执行一些参数映射，以尽可能支持应用程序接口的更改。

_精简_运行时包含的运算符实现比_默认_运行时少。由于 TensorRT 会在构建时选择算子的实现方式，因此需要指定_精简_运行时构建的引擎。与_默认_运行时构建的引擎相比，它可能会稍慢一些。

_精简_运行时包含_调度_运行时的所有功能，_默认_运行时包含_精简_运行时的所有功能。

TensorRT 提供了与上述每个库相对应的 Python 包：

* tensorrt
* tensorrt\_lean
* tensorrt\_dispatch
