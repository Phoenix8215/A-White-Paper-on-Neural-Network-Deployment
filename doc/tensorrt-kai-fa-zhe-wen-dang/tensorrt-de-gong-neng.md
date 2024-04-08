# TensorRT的功能

本章概述了您可以使用 TensorRT 做什么。

## C++ and Python APIs

TensorRT 的 API 具有 C++ 和 Python 的语言绑定，具有几乎相同的功能。 Python API 促进了与 Python 数据处理工具包和库（如 NumPy 和 SciPy）的互操作性。 C++ API 可以更高效，并且可以更好地满足某些合规性要求，**例如在汽车应用中**。 **注意： Python API 并非适用于所有平台。有关详细信息，请参阅**[**NVIDIA TensorRT 支持矩阵**](https://docs.nvidia.com/deeplearning/sdk/tensorrt-support-matrix/index.html)**。**

## The Programming Model

TensorRT 构建阶段的最高级别接口是Builder （ C++ 、 Python ）。构建器负责优化模型并生成Engine 。

为了构建引擎，您需要：

* 创建网络定义
* 为builder指定配置
* 调用builder创建引擎

`NetworkDefinition`接口（ [C++](https://docs.nvidia.com/deeplearning/tensorrt/api/c\_api/classnvinfer1\_1\_1\_i\_network\_definition.html) 、 [Python](https://docs.nvidia.com/deeplearning/tensorrt/api/python\_api/infer/Graph/Network.html#inetworkdefinition) ）用于定义模型。<mark style="color:red;">将模型传输到 TensorRT 的最常见途径是以 ONNX 格式从框架中导出模型，并使用 TensorRT 的 ONNX 解析器来填充网络定义。但是，您也可以使用 TensorRT 的Layer (</mark> [<mark style="color:red;">C++</mark>](https://docs.nvidia.com/deeplearning/tensorrt/api/c\_api/classnvinfer1\_1\_1\_i\_layer.html) <mark style="color:red;">,</mark> [<mark style="color:red;">Python</mark>](https://docs.nvidia.com/deeplearning/tensorrt/api/python\_api/infer/Graph/LayerBase.html#ilayer) <mark style="color:red;">) 和Tensor (</mark> [<mark style="color:red;">C++</mark>](https://docs.nvidia.com/deeplearning/tensorrt/api/c\_api/classnvinfer1\_1\_1\_i\_tensor.html) <mark style="color:red;">,</mark> [<mark style="color:red;">Python</mark>](https://docs.nvidia.com/deeplearning/tensorrt/api/python\_api/infer/Graph/LayerBase.html#itensor) <mark style="color:red;">) 接口逐步构建定义。</mark>

<mark style="color:red;">无论您选择哪种方式，您还必须定义哪些张量是网络的输入和输出。未标记为输出的张量被认为是可以由构建器优化掉的瞬态值。输入和输出张量必须命名，以便在运行时，TensorRT 知道如何将输入和输出缓冲区绑定到模型。</mark>

`BuilderConfig`接口（ [C++](https://docs.nvidia.com/deeplearning/tensorrt/api/c\_api/classnvinfer1\_1\_1\_i\_builder\_config.html) 、 [Python](https://docs.nvidia.com/deeplearning/tensorrt/api/python\_api/infer/Core/BuilderConfig.html) ）用于指定TensorRT如何优化模型。在可用的配置选项中，您可以<mark style="color:red;">控制 TensorRT 降低计算精度的能力，控制内存和运行时执行速度之间的权衡</mark>，以及限制对 CUDA ®内核的选择。由于构建器可能需要几分钟或更长时间才能运行，因此您还可以控制构建器搜索内核的方式，以及缓存搜索结果以供后续使用。

一旦有了网络定义和构建器配置，就可以调用构建器来创建引擎。<mark style="color:red;">构建器消除了无效计算、折叠常量、重新排序和组合操作以在 GPU 上更高效地运行。</mark>它可以选择性地降低浮点计算的精度，方法是简单地在 16 位浮点中运行它们，或者通过量化浮点值以便可以使用 8 位整数进行计算。它还使用不同的数据格式对每一层的多次实现进行计时，然后计算执行模型的最佳时间表，从而最大限度地降低内核执行和格式转换的综合成本。

构建器(builder)使用序列化形式的plan创建引擎(engine)，plan可以立即反序列化，或保存到磁盘以供以后使用。

**注意：**

* <mark style="color:red;">TensorRT 创建的引擎特定于创建它们的 TensorRT 版本和创建它们的 GPU。</mark>
* <mark style="color:red;">TensorRT 的网络定义不会深度复制参数数组（例如卷积的权重）。因此，在构建阶段完成之前，您不得释放这些阵列的内存。使用 ONNX 解析器导入网络时，解析器拥有权重，因此在构建阶段完成之前不得将其销毁。</mark>
* <mark style="color:red;">构建器时间算法以确定最快的优化策略。与其他使用 GPU 程序并行运行构建器可能会扰乱时序，导致优化不佳。</mark>

### The Runtime Phase

TensorRT 执行阶段的最高级别接口是`Runtime`（ [C++](https://docs.nvidia.com/deeplearning/tensorrt/api/c\_api/classnvinfer1\_1\_1\_i\_runtime.html) 、 [Python](https://docs.nvidia.com/deeplearning/tensorrt/api/python\_api/infer/Core/Runtime.html) ）。 使用运行时时，您通常会执行以下步骤：

* <mark style="color:red;">反序列化plan创建引擎(plan 文件)</mark>
* <mark style="color:red;">从引擎创建执行上下文(context)</mark>&#x20;

然后，反复地：

* <mark style="color:red;">填充输入缓冲区以进行推理</mark>
* <mark style="color:red;">调用</mark><mark style="color:red;">`enqueueV3()`</mark><mark style="color:red;">以运行推理</mark>

`Engine`接口（ [C++](https://docs.nvidia.com/deeplearning/tensorrt/api/c\_api/classnvinfer1\_1\_1\_i\_cuda\_engine.html) 、 [Python](https://docs.nvidia.com/deeplearning/tensorrt/api/python\_api/infer/Core/Engine.html) ）代表一个优化模型。<mark style="color:red;">您可以查询引擎以获取有关网络输入和输出张量的信息——预期的维度、数据类型、数据格式等。</mark>

`ExecutionContext`接口（ [C++](https://docs.nvidia.com/deeplearning/tensorrt/api/c\_api/classnvinfer1\_1\_1\_i\_cuda\_engine.html) 、 [Python](https://docs.nvidia.com/deeplearning/tensorrt/api/python\_api/infer/Core/Engine.html) ）是调用推理的主要接口。执行上下文包含与特定调用关联的所有状态 - <mark style="color:red;">因此您可以拥有与单个引擎关联的多个上下文，并并行运行它们。</mark>

调用推理时，您必须在适当的位置设置输入和输出缓冲区。根据输入输出数据类型的不同，这可能在 CPU 或 GPU 内存中。您可以使用引擎的相关API以确定缓冲区在哪个内存空间中。

<mark style="color:red;">设置缓冲区后，可以同步（执行）或异步（入队）调用推理。在后一种情况下，所需的内核在 CUDA 流上排队，并尽快将控制权返回给应用程序。一些网络需要在 CPU 和 GPU 之间进行多次控制传输，因此控制可能不会立即返回。如果要等待异步执行完成，请使用</mark><mark style="color:red;">`cudaStreamSynchronize`</mark><mark style="color:red;">在流上同步。</mark>

## Plugins

TensorRT 有一个`Plugin`接口，允许应用程序提供 TensorRT 本身不支持的操作的实现。在转换网络时，ONNX 解析器可以找到使用 TensorRT 的`PluginRegistry`创建和注册的插件。

TensorRT 附带一个插件库，其中许多插件和一些附加插件的源代码可以在此处找到。

请参阅[使用自定义层扩展 TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#extending)一章。

## Types and Precision

TensorRT 支持 FP32、FP16、BF16、FP8、INT4、INT8、INT32、INT64、UINT8 和 BOOL 数据类型。有关层 I/O 数据类型规范，请参阅 [TensorRT 算子文档](https://docs.nvidia.com/deeplearning/tensorrt/operators/docs/)。

* FP32、FP16、BF16：未量化浮点类型 - INT8：低精度整数类型 ￼ ◦ 隐式量化 ￼ ■ 解释为量化整数。INT8 类型的张量必须有相关的比例因子（通过校准或 setDynamicRange API）。从 INT8 类型转换到 INT8 类型需要显式 Q/DQ 层。◦ INT4 权重需要通过每个字节打包两个元素的方式进行序列化。FP8：低精度浮点类型 ￼ ◦ 8 位浮点类型，1 位符号、4 位指数、3 位尾数 ◦ 与 FP8 类型之间的转换需要显式 Q/DQ 层。 ◦ UINT8 的网络级输入必须使用 CastLayer 从 UINT8 转换为 FP32 或 FP16，然后才能用于其他操作。 ◦ UINT8 的网络级输出必须由明确插入网络的 CastLayer 生成（仅支持从 FP32/FP16 转换为 UINT8）。 不支持 UINT8 量化。



请参阅[降低精度](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#reduced-precision)部分。

## Quantization

TensorRT 支持量化浮点，其中浮点值被线性压缩并四舍五入为 8 位整数。这显着提高了算术吞吐量，同时降低了存储要求和内存带宽。在量化浮点张量时，TensorRT 需要知道它的动态范围——即表示什么范围的值很重要——量化时会截断超出该范围的值。

动态范围信息可由构建器根据代表性输入数据计算（这称为校准--`calibration`）。或者，您可以在框架中执行量化感知训练，并将模型与必要的动态范围信息一起导入到 TensorRT。

请参阅[使用 INT8](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#working-with-int8)章节。

## 2.6. Tensors and Data Formats

在定义网络时，TensorRT 假设张量由多维 C 样式数组表示。每一层对其输入都有特定的解释：例如，2D 卷积将假定其输入的最后三个维度是 CHW 格式 - 不可以使用 WHC 格式。有关每个层如何解释其输入，请参阅[TensorRT 网络层](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#layers)一章。

请注意，张量最多只能包含 2^31-1 个元素。 在优化网络的同时，TensorRT 在内部执行转换（转换到 HWC，但也包括更复杂的格式）以使用尽可能快的 CUDA 内核。通常，选择格式是为了优化性能，而应用程序无法控制这些选择。然而，底层数据格式暴露在 I/O 边界（网络输入和输出，以及将数据传入和传出的插件），以允许应用程序最大限度地减少不必要的格式转换。

请参阅[I/O 格式](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#reformat-free-network-tensors)部分

## 2.7. Dynamic Shapes

<mark style="color:red;">默认情况下，TensorRT 根据定义时的输入形状（批量大小、图像大小等）优化模型。但是，可以将构建器配置为允许在运行时调整输入维度。为了启用此功能，您可以在构建器配置中指定一个或多个</mark><mark style="color:red;">`OptimizationProfile`</mark> <mark style="color:red;"></mark><mark style="color:red;">（</mark> [<mark style="color:red;">C++</mark>](https://docs.nvidia.com/deeplearning/tensorrt/api/c\_api/classnvinfer1\_1\_1\_i\_optimization\_profile.html) <mark style="color:red;">、</mark> [<mark style="color:red;">Python</mark>](https://docs.nvidia.com/deeplearning/tensorrt/api/python\_api/infer/Core/OptimizationProfile.html?highlight=optimizationprofile) <mark style="color:red;">）实例，其中包含每个输入的最小和最大形状，以及该范围内的优化点。</mark>

<mark style="color:red;">TensorRT 为每个配置文件创建一个优化的引擎，选择适用于 \[最小、最大] 范围内的所有形状的 CUDA 内核，并且对于优化点来说是最快的——通常每个配置文件都有不同的内核。然后，您可以在运行时在配置文件中进行选择。</mark>

请参阅[使用动态形状](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#work\_dynamic\_shapes)一章。

## 2.8. DLA

TensorRT 支持 NVIDIA 的深度学习加速器 (Deep Learning Accelerator)，这是许多 NVIDIA SoC 上的专用推理处理器，支持 TensorRT 层的子集。 TensorRT 允许您在 DLA 上执行部分网络，而在 GPU 上执行其余部分；对于可以在任一设备上执行的层，您可以在构建器配置中逐层选择目标设备。

请参阅使用 [DLA](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#refitting-engine-c)章节。

## 2.9. Updating Weights

在构建引擎时，您可以指定它更新其权重。如果您经常在不更改结构的情况下更新模型的权重，例如在强化学习中或在保留相同结构的同时重新训练模型时，这将很有用。权重更新是通过`Refitter` ( [C++ ](https://docs.nvidia.com/deeplearning/tensorrt/api/c\_api/classnvinfer1\_1\_1\_i\_refitter.html), [Python](https://docs.nvidia.com/deeplearning/tensorrt/api/python\_api/infer/Core/Refitter.html) ) 接口执行的。

请参阅[Refitting An Engine](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#refitting-engine-c) 部分。

## 2.10. trtexec

示例目录中包含一个名为`trtexec`的命令行包装工具。 `trtexec`是一种无需开发自己的应用程序即可快速使用 TensorRT 的工具。 trtexec工具有三个主要用途：

* <mark style="color:red;">在随机或用户提供的输入数据上对网络进行基准测试。</mark>
* <mark style="color:red;">从模型生成序列化引擎。</mark>
* <mark style="color:red;">从构建器生成序列化时序缓存。</mark>

请参阅[trtexec](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#trtexec)部分。

## 2.11. Polygraphy

Polygraphy 是一个工具包，旨在帮助在 TensorRT 和其他框架中运行和调试深度学习模型。它包括一个Python API和一个使用此 API 构建的命令行界面 (CLI) 。

除此之外，使用 Polygraphy，您可以：

* 在多个后端之间运行推理，例如 TensorRT 和 ONNX-Runtime，并比较结果（例如[API](https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/examples/api/01\_comparing\_frameworks) 、 [CLI](https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/examples/cli/run/01\_comparing\_frameworks) ）
* 将模型转换为各种格式，例如具有训练后量化的 TensorRT 引擎（例如[API](https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/examples/api/04\_int8\_calibration\_in\_tensorrt) 、 [CLI](https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/examples/cli/convert/01\_int8\_calibration\_in\_tensorrt) ）
* 查看有关各种类型模型的信息（例如[CLI](https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/examples/cli/inspect) ）
* 在命令行上修改 ONNX 模型：
  * 提取子图（例如[CLI](https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/examples/cli/surgeon/01\_isolating\_subgraphs) ）
  * 简化和清理（例如[CLI](https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/examples/cli/surgeon/02\_folding\_constants) ）
* 隔离 TensorRT 中的错误策略（例如[CLI](https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/examples/cli/debug/01\_debugging\_flaky\_trt\_tactics) ）

有关更多详细信息，请参阅[Polygraphy](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy) 代码仓库。
