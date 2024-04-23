# TensorRT的C++接口解析

本章说明 C++ API 的基本用法，假设你从 ONNX 模型开始。 [sampleOnnxMNIST](https://github.com/NVIDIA/TensorRT/tree/main/samples/sampleOnnxMNIST)更详细地说明了这个用例。

C++ API 可以通过头文件`NvInfer.h`访问，并且位于`nvinfer1`命名空间中。例如，一个简单的应用程序：

```cpp
#include “NvInfer.h”

using namespace nvinfer1;
```

<mark style="color:red;">TensorRT C++ API 中的接口类以前缀</mark><mark style="color:red;">`I`</mark><mark style="color:red;">开头，例如</mark><mark style="color:red;">`ILogger`</mark> <mark style="color:red;"></mark><mark style="color:red;">、</mark> <mark style="color:red;"></mark><mark style="color:red;">`IBuilder`</mark><mark style="color:red;">等。</mark>

CUDA 上下文会在 TensorRT 第一次调用 CUDA 时自动创建，如果在该点之前不存在。通常最好在第一次调用 TensoRT 之前自己创建和配置 CUDA 上下文。 <mark style="color:red;">为了说明对象的生命周期，本章中的代码不使用智能指针。</mark>

## The Build Phase

要创建构建器，首先需要实例化`ILogger`接口。此示例捕获所有警告消息，但忽略信息性消息：

```cpp
class Logger : public ILogger           
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} logger;
```

然后，可以创建构建器的实例：

```cpp
IBuilder* builder = createInferBuilder(logger);
```

### Creating a Network Definition

创建构建器后，优化模型的第一步是创建网络定义：

```cpp
uint32_t flag = 1U <<static_cast<uint32_t>
    (NetworkDefinitionCreationFlag::kEXPLICIT_BATCH); 

INetworkDefinition* network = builder->createNetworkV2(flag);
```

为了使用 ONNX 解析器导入模型，需要`kEXPLICIT_BATCH`标志。有关详细信息，请参阅[显式与隐式批处理](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#explicit-implicit-batch)部分。

### Importing a Model using the ONNX Parser

现在，需要从 ONNX 表示中填充网络定义。 ONNX 解析器 API 位于文件`NvOnnxParser.h`中，解析器位于`nvonnxparser` C++ 命名空间中。

```cpp
#include “NvOnnxParser.h”

using namespace nvonnxparser;
```

可以创建一个 ONNX 解析器来填充网络，如下所示：

```cpp
IParser*  parser = createParser(*network, logger);
```

然后，读取模型文件并处理错误。

```cpp
parser->parseFromFile(modelFile, 
    static_cast<int32_t>(ILogger::Severity::kWARNING));
for (int32_t i = 0; i < parser.getNbErrors(); ++i)
{
std::cout << parser->getError(i)->desc() << std::endl;
}
```

TensorRT 网络定义过程中的一个重点是它包含指向模型权重的指针，这些指针由构建器(`Builder`)复制到引擎(`Engine`)中。<mark style="color:red;">由于网络是通过解析器创建的，解析器拥有权重占用的内存，因此在构建器运行之前不应删除解析器对象。</mark>

### Building an Engine

下一步是创建一个配置，指定 TensorRT 应该如何优化模型。

```cpp
IBuilderConfig* config = builder->createBuilderConfig();
```

这个接口有很多属性，你可以设置这些属性来控制 TensorRT 如何优化网络。<mark style="color:red;">一个重要的属性是最大工作空间大小。</mark>层实现通常需要一个临时工作空间，并且此参数限制了网络中任何层可以使用的最大大小。如果提供的工作空间不足，TensorRT 可能无法找到层的实现。<mark style="color:red;">默认情况下，工作区设置为给定设备的总全局内存大小；必要时限制它，例如，在单个设备上构建多个引擎时。</mark>

```cpp
config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1U << 20);
```

一旦指定了配置，就可以构建引擎。

```cpp
IHostMemory*  serializedModel = builder->buildSerializedNetwork(*network, *config);
```

由于序列化引擎包含权重的必要拷贝，因此不再需要解析器、网络定义、构建器配置和构建器，可以安全地删除：

```cpp
delete parser;
delete network;
delete config;
delete builder;
```

然后可以将引擎保存到磁盘，并且可以删除保存它序列化数据的缓冲区。

```cpp
delete serializedModel
```

#### 注意：序列化引擎不能跨平台或不同 TensorRT 版本之间移植。

## Deserializing a Plan

假设您之前已经序列化了一个模型并希望执行推理，将需要创建一个运行时接口的实例。与构建器一样，运行时需要一个记录器(logger)实例：

```cpp
IRuntime* runtime = createInferRuntime(logger);
```

假设您已将模型从缓冲区中读取，然后可以对其进行反序列化以获得引擎：

```cpp
ICudaEngine* engine = 
  runtime->deserializeCudaEngine(modelData, modelSize);
```

## Performing Inference

引擎包含优化后的模型，但要执行推理，我们需要管理中间激活的额外状态。这是通过`ExecutionContext`接口完成的：

```cpp
IExecutionContext *context = engine->createExecutionContext();
```

<mark style="color:red;">一个引擎可以有多个执行上下文，允许一组权重用于多个重叠的推理任务。 （一个例外是使用动态形状时，每个优化配置文件只能有一个执行上下文。）</mark>

要执行推理，您必须为输入和输出传递 `TensorRT` 缓冲区，`TensorRT` 要求你通过调用 `setTensorAddress` 来指定这些缓冲区，`setTensorAddress` 接收张量名称和缓冲区地址。您可以使用提供的输入和输出张量名称查询引擎，以找到在数组中位置：：

```cpp
context->setTensorAddress(INPUT_NAME, inputBuffer);
context->setTensorAddress(OUTPUT_NAME, outputBuffer);
```

如果使用动态形状创建引擎，则还必须指定输入形状：

```cpp
context->setInputShape(INPUT_NAME, inputDims);
```

然后，您可以调用 TensorRT 的 `enqueueV3` 方法以使用CUDA 流启动异步推理：

```cpp
context->enqueueV3(buffers, stream, nullptr);
```

网络是否以异步方式执行取决于网络的结构和特性。例如，数据的维度、DLA 使用、循环和同步插件等，都可能导致同步行为。在内核前后使用 `cudaMemcpyAsync()` 来进行数据传输是很常见的做法，这样可以将数据从 GPU 转移到其他地方。

要确定内核（使用 `cudaMemcpyAsyn()`的时候)何时完成，可使用标准的 CUDA 同步机制，如事件或在流上等待。
