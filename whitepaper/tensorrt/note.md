# 🐭 文档简介

NVIDIA® TensorRT™ 是一个促进高性能机器学习推理的 SDK。 它旨在与 TensorFlow、PyTorch 和 MXNet 等训练框架以互补的方式工作。 它特别专注于在 NVIDIA 硬件上快速高效地运行已经训练好的网络。 有关如何安装 TensorRT 的说明，请参阅 [NVIDIA TensorRT 安装指南](https://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html)。

[NVIDIA TensorRT 快速入门指南](https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html)适用于想要试用 TensorRT SDK 的用户； 具体来说，您将学习如何快速构建应用程序以在 TensorRT 引擎上运行推理。       &#x20;

## Structure of this Guide

第 1 章提供了有关如何打包和支持 TensorRT 以及它如何融入开发者生态系统的信息。

第 2 章提供了对 TensorRT 功能的广泛概述。

第 3 章和第 4 章分别介绍了 C++ 和 Python API。

后续章节提供有关高级功能的更多详细信息。

附录包含网络层参考和常见问题解答。

## Samples

[NVIDIA TensorRT 示例支持指南](https://docs.nvidia.com/deeplearning/tensorrt/sample-support-guide/index.html)说明了本手册中讨论的许多主题。可在此处找到其他侧重于嵌入式应用程序的示例。

## Complementary GPU Features

[多实例 GPU](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/index.html)或 MIG 是具有 NVIDIA Ampere 架构或更高架构的 NVIDIA GPU 的一项功能，可实现用户控制的将单个 GPU 划分为多个较小 GPU 的功能。物理分区提供具有 QoS 的专用计算和内存切片，并在 GPU 的一部分上独立执行并行工作负载。对于 GPU 利用率低的 TensorRT 应用程序，MIG 可以在对延迟影响很小或没有影响的情况下产生更高的吞吐量。最佳分区方案是特定于应用程序的。

## Complementary Software

[NVIDIA Triton™](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/index.html)推理服务器是一个更高级别的库，可提供跨 CPU 和 GPU 的优化推理。它提供了启动和管理多个模型的功能，以及用于服务推理的 REST 和 gRPC 端点。

[NVIDIA DALI ®](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/#nvidia-dali-documentation)为预处理图像、音频和视频数据提供高性能原语。 TensorRT 推理可以作为自定义算子集成到 DALI 管道中。可以在[此处](https://github.com/NVIDIA/DL4AGX)找到作为 DALI 的一部分集成的 TensorRT 推理的工作示例。

[TensorFlow-TensorRT (TF-TRT)](https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html)是将 TensorRT 直接集成到 TensorFlow 中。它选择 TensorFlow 图的子图由 TensorRT 加速，同时让图的其余部分由 TensorFlow 本地执行。结果仍然是您可以照常执行的 TensorFlow 图。有关 TF-TRT 示例，请参阅[TensorFlow 中的 TensorRT 示例](https://github.com/tensorflow/tensorrt)。

[PyTorch 量化工具包](https://docs.nvidia.com/deeplearning/tensorrt/pytorch-quantization-toolkit/docs/userguide.html)提供了以降低精度训练模型的工具，然后可以将其导出以在 TensorRT 中进行优化。

此外， [PyTorch Automatic SParsity (ASP)](https://github.com/NVIDIA/apex/tree/master/apex/contrib/sparsity)工具提供了用于训练具有结构化稀疏性的模型的工具，然后可以将其导出并允许 TensorRT 在 NVIDIA Ampere GPU 上利用更快的稀疏策略。

TensorRT 与 NVIDIA 的分析工具、 [NVIDIA Nsight™ Systems](https://developer.nvidia.com/nsight-systems)和[NVIDIA® Deep Learning Profiler (DLProf)](https://docs.nvidia.com/deeplearning/frameworks/dlprof-user-guide/)集成。

## ONNX

TensorRT 从框架中导入训练模型的主要方式是通过[ONNX](https://onnx.ai/)交换格式。 TensorRT 附带一个 ONNX 解析器库来帮助导入模型。在可能的情况下，解析器向后兼容 opset 7； ONNX模型[ Opset 版本转换器](https://github.com/onnx/onnx/blob/master/docs/VersionConverter.md)可以帮助解决不兼容问题。

[GitHub 版本](https://github.com/onnx/onnx-tensorrt/)可能支持比 TensorRT 附带的版本更高的 opset，请参阅 ONNX-TensorRT[运算符支持矩阵](https://github.com/onnx/onnx-tensorrt/blob/master/docs/operators.md)以获取有关受支持的 opset 和运算符的最新信息。

TensorRT 的 ONNX 算子支持列表可在[此处](https://github.com/onnx/onnx-tensorrt/blob/master/docs/operators.md)找到。

PyTorch 原生支持[ONNX 导出](https://pytorch.org/docs/stable/onnx.html)。对于 TensorFlow，推荐的方法是[tf2onnx](https://github.com/onnx/tensorflow-onnx) 。

将模型导出到 ONNX 后的第一步是使用[Polygraphy](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#polygraphy-ovr)运行常量折叠。这通常可以解决 ONNX 解析器中的 TensorRT 转换问题，并且通常可以简化工作流程。有关详细信息，请参阅[此示例](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy/examples/cli/surgeon/02\_folding\_constants)。在某些情况下，可能需要进一步修改 ONNX 模型，例如，用插件替换子图或根据其他操作重新实现不受支持的操作。为了简化此过程，您可以使用[ONNX-GraphSurgeon](https://github.com/NVIDIA/TensorRT/tree/main/tools/onnx-graphsurgeon) 。

## Code Analysis Tools

有关在 TensorRT 中使用 valgrind 和 clang sanitizer 工具的指导，请参阅[故障排除](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#troubleshooting)章节。
