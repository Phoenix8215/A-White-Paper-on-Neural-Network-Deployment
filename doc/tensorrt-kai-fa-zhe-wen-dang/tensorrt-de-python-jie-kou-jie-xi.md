# TensorRT的Python接口解析

本章说明 Python API 的基本用法，假设您从 ONNX 模型开始。 [onnx\_resnet50.py](https://github.com/NVIDIA/TensorRT/blob/main/samples/python/introductory\_parser\_samples/onnx\_resnet50.py)示例更详细地说明了这个用例。

Python API 可以通过tensorrt模块访问：

```Python
import tensorrt as trt
```

## 4.1. The Build Phase

要创建构建器，您需要首先创建一个记录器。 Python 绑定包括一个简单的记录器实现，它将高于某一级别的所有消息记录到`stdout` 。

```Python
logger = trt.Logger(trt.Logger.WARNING)
```

或者，可以通过从`ILogger`类派生来定义您自己的记录器实现：

```Python
class MyLogger(trt.ILogger):
    def __init__(self):
       trt.ILogger.__init__(self)

    def log(self, severity, msg):
        pass # Your custom logging implementation here

logger = MyLogger()
```

然后，您可以创建一个构建器：

```Python
builder = trt.Builder(logger)
```

### 4.1.1. Creating a Network Definition in Python

创建构建器后，优化模型的第一步是创建网络定义。网络定义选项是由多个标记 OR-d 组合而成的。为了使用 ONNX 解析器导入模型，需要`EXPLICIT_BATCH`标志。有关详细信息，请参阅[显式与隐式批处理](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#explicit-implicit-batch)部分。您还可以使用 NetworkDefinitionCreationFlag.STRONGLY\_TYPED 标记指定网络应被视为强类型网络。有关详细信息，请参[阅强类型网络](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-1000-ea/developer-guide/index.html#strongly-typed-networks)。 最后，创建网络：：

```Python
builder = trt.Builder(logger)
```



### 4.1.2. Importing a Model using the ONNX Parser

现在，需要从 ONNX 表示中填充网络定义。您可以创建一个 ONNX 解析器来填充网络，如下所示：

```Python
parser = trt.OnnxParser(network, logger)
```

然后，读取模型文件并处理任何错误：

```Python
success = parser.parse_from_file(model_path)
for idx in range(parser.num_errors):
    print(parser.get_error(idx))

if not success:
    pass # Error handling code here
```

### 4.1.3. Building an Engine

下一步是创建一个构建配置，指定 TensorRT 应该如何优化模型：

```Python
config = builder.create_builder_config()

```

这个接口有很多属性，你可以设置这些属性来控制 TensorRT 如何优化网络。一个重要的属性是最大工作空间大小。层实现通常需要一个临时工作空间，并且此参数限制了网络中任何层可以使用的最大大小。如果提供的工作空间不足，TensorRT 可能无法找到层的实现：

```Python
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20) # 1 MiB

```

指定配置后，可以使用以下命令构建和序列化引擎：

```Python
serialized_engine = builder.build_serialized_network(network, config)

```

将引擎保存到文件以供将来使用。你可以这样做：

```Python
with open(“sample.engine”, “wb”) as f:
    f.write(serialized_engine)
```

**注意：**序列化引擎不能跨平台移植。除平台外，引擎还与所构建的 GPU 模型相关。

## 4.2. Deserializing a Plan

要执行推理，您首先需要使用Runtime接口反序列化引擎。与构建器一样，运行时需要记录器的实例。

```Python
runtime = trt.Runtime(logger)

```

然后，您可以从内存缓冲区反序列化引擎：

```Python
engine = runtime.deserialize_cuda_engine(serialized_engine)

```

如果您需要首先从文件加载引擎，请运行：

```Python
with open(“sample.engine”, “rb”) as f:
    serialized_engine = f.read()
```

## 4.3. Performing Inference

引擎拥有优化的模型，但要执行推理需要额外的中间激活状态。这是通过`IExecutionContext`接口完成的：

```Python
context = engine.create_execution_context()
```

一个引擎可以有多个执行上下文，允许一组权重用于多个重叠的推理任务。 （一个例外是使用动态形状时，每个优化配置文件只能有一个执行上下文。）

要执行推理，必须为输入和输出指定缓冲区：：

```Python
context.set_tensor_address(name, ptr)
```

有几个 Python 软件包允许您在 GPU 上分配内存，包括但不限于官方的 CUDA Python 绑定、PyTorch、cuPy 和 Numba。

填充输入缓冲区后，可以调用 TensorRT 的 execute\_async\_v3 方法，开始使用 CUDA 流进行推理。网络是否以异步方式执行取决于网络的结构和特性。例如，数据相关形状、DLA 使用、循环和同步插件等都可能导致同步行为。

首先，创建 CUDA 数据流。如果已经有一个 CUDA 流，则可以使用指向现有流的指针。例如，对于 PyTorch CUDA 流（即 torch.cuda.Stream()），可以使用 cuda\_stream 属性访问指针；对于 Polygraphy CUDA 流，可以使用 ptr 属性；也可以通过调用 cudaStreamCreate() 直接使用 CUDA Python 绑定创建流。

接下来，开始推理：

```python
context.execute_async_v3(buffers, stream_ptr)
```

在内核之前和之后启动异步传输（cudaMemcpyAsync()）是很常见的做法，以便从 GPU 转移数据（如果还没有的话）。

要确定推理（和异步传输）何时完成，可使用标准的 CUDA 同步机制，如事件或在流上等待。例如，对于 PyTorch CUDA 流或 Polygraphy CUDA 流，可发出 stream.synchronize()。对于使用 CUDA Python 绑定创建的流，请执行 cudaStreamSynchronize(stream)。
