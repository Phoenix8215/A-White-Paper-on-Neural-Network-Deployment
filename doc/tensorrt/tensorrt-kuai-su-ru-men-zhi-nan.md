# 🐱 TensorRT快速入门指南

### TensorRT 生态系统

TensorRT 是一个庞大而灵活的项目。它可以处理各种转换和部署工作流程，哪种工作流程最适合您取决于您的具体用例和问题设置。

#### TensorRT 基本工作流程

<figure><img src="../../.gitbook/assets/图片.png" alt=""><figcaption></figcaption></figure>

#### 转换和部署选项

TensorRT 生态系统分为两个部分：

1. 用户可通过各种途径将其模型转换为已经优化好的 TensorRT 引擎。
2. 用户在部署已经优化后的 TensorRT 引擎时，可将 TensorRT 作为推理 runtime。

<figure><img src="../../.gitbook/assets/图片 (1).png" alt=""><figcaption></figcaption></figure>

使用 TensorRT 转换模型有三个主要选项：

* 使用 TF-TRT
* 自动从ONNX文件中完成转换
* 使用 TensorRT API（C++ 或 Python 语言）手动构建网络

对于 TensorFlow 模型的转换，TensorFlow 集成（TF-TRT）提供了模型转换和高级runtime API，并能在 TensorRT 不支持特定算子时返回到 TensorFlow 中去实现运行。

使用 ONNX 进行转换一种性能更高的模型部署方式。TensorRT 支持使用 TensorRT API 或trtexec从 ONNX 文件中进行自动转换。使用ONNX 转换意味着模型中的所有操作都必须得到 TensorRT 的支持（或者必须为不支持的操作提供自定义插件）。ONNX 转换的结果是一个单一的 TensorRT 引擎，比使用 TF-TRT 的开销更少。

为了尽可能提高性能和可定制性，您还可以使用 TensorRT 网络定义 API 手动构建 TensorRT 引擎。这意味着只使用 TensorRT 操作来构建与目标模型完全相同的网络。用 TensorRT 创建网络后，只需导出模型的权重，然后将权重加载到网络中即可。

* [使用 C++ API 从零开始创建网络定义](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#create\_network\_c)
* [使用 Python API 从零开始创建网络定义](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/#create\_network\_python)

#### 部署

使用 TensorRT 部署模型有三种选择：

* 在 TensorFlow 中部署
* 使用独立的 TensorRT runtime API
* 使用英伟达 Triton 推理服务器

TF-TRT 转换的结果是一个插入了 TensorRT 操作的 TensorFlow 图。这意味着您可以像使用其他 TensorFlow 模型一样使用 Python 来运行 TF-TRT 模型。

TensorRT runtime  API 可以实现最低的开销和最精细的控制，但 TensorRT 本身不支持的算子必须以插件形式实现（[此处](https://github.com/NVIDIA/TensorRT/tree/main/plugin)提供了一个预编写插件库）。

最后，英伟达™ Triton推理服务器是一款开源推理服务软件，可帮助团队在任何基于GPU或CPU的基础设施（云、数据中心或边缘）上部署来自任何框架（TensorFlow、TensorRT、PyTorch、ONNX Runtime或自定义框架）、本地存储或谷歌云平台或AWS S3的人工智能模型。它是一个灵活的项目，具有一些独特的功能，例如异构模型和同一模型多个副本的并发模型执行（多个模型副本可进一步减少延迟）以及负载平衡和模型分析。 如果您必须通过 HTTP 提供模型，例如在云推理解决方案中，它是一个不错的选择。您可以[在这里](https://developer.nvidia.com/nvidia-triton-inference-server)找到英伟达™ Triton 推理服务器主页，[在这里找到](https://github.com/triton-inference-server/server/blob/r22.01/README.md#documentation)相关文档。

#### 选择正确的工作流程

在选择如何转换和部署模型时，有两个最重要的因素：

1. 您选择的框架。
2. 您首选的 TensorRT runtime。

下面的流程图涵盖了本指南中涉及的不同工作流程。该流程图将帮助您根据这两个因素选择合适的路径。

<figure><img src="../../.gitbook/assets/图片 (2).png" alt=""><figcaption></figcaption></figure>

### 使用 ONNX 的部署示例

在本节中，我们将以部署预训练的 ONNX 模型为背景，介绍 TensorRT 转换的五个基本步骤。

在本例中，我们将使用 ONNX 格式转换一个预训练的 ResNet-50 模型；ONNX 是一种与框架无关的模型格式，可从大多数主流框架（包括 TensorFlow 和 PyTorch）导出。有关 ONNX 格式的更多信息，请[点击此处](https://github.com/onnx/onnx/blob/main/docs/IR.md)。

<figure><img src="../../.gitbook/assets/图片 (5).png" alt=""><figcaption></figcaption></figure>

#### 导出模型

TensorRT 根据文件类型不同提供了两种不同的文件转换方式：

* TF-TRT 使用 TensorFlow 保存好的模型文件。
* ONNX path要求模型保存在 ONNX 中。

在本例中，我们使用的是 ONNX，因此需要一个 ONNX 模型。

使用wget从 [ONNX model zoo](https://github.com/onnx/models) 下载预训练的 ResNet-50 模型并解压缩

```sh
wget https://download.onnxruntime.ai/onnx/models/resnet50.tar.gz 
tar xzf resnet50.tar.gz
```

这将解压缩一个预训练的 ResNet-50.onnx文件到resnet50/model.onnx 路径下。

#### batchsize的选择

batchsize会对模型执行的优化产生很大影响。一般来说，在推理时，当我们想优先考虑延迟时，我们会选择较小的batchsize，而当我们想优先考虑吞吐量时，我们会选择较大的batchsize。较大的batchsize需要更长的处理时间，但可以减少每个样本的平均处理时间。

如果在运行前不知道需要多大的batchsize，TensorRT 可以动态处理batchsize大小。也就是说，固定的batchsize会让 TensorRT 进行额外的优化。在本示例工作流中，我们使用的固定batchsize为 64。

```
BATCH_SIZE=64
```

#### 精度的选择

推理所需的数字精度通常低于训练所需的精度。较低的精度可以加快计算速度，降低内存消耗，而不会牺牲任何精度。TensorRT 支持 TF32、FP32、FP16 和 INT8 精度。FP32 是大多数框架的默认训练精度，因此我们将首先使用 FP32 进行推理

```
import numpy as np
PRECISION = np.float32
```

我们将在下一节中设置 TensorRT 引擎在运行时所使用的精度。

#### 转换模型

ONNX 转换是 TensorRT 自动转换方式中最通用、性能最好的路径之一。它适用于 TensorFlow、PyTorch 和许多其他框架。

有几种工具可以帮助你将 ONNX 模型转换为 TensorRT 引擎。一种常见的方法是使用trtexec，它是 TensorRT 附带的一个命令行工具，可以将 ONNX 模型转换为 TensorRT 引擎并对其进行剖析。

我们可以按以下方式运行转换

```
trtexec --onnx=resnet50/model.onnx --saveEngine=resnet_engine.trt
```

这将把resnet50/model.onnx转换为名为resnet\_engine.trt 的 TensorRT 引擎。注意：

*   要告诉trtexec在哪里找到我们的 ONNX 模型

    ```
    --onnx=resnet50/model.onnx
    ```
*   要告诉trtexec保存优化后的 TensorRT 引擎的位置

    ```
    --saveEngine=resnet_engine_intro.trt
    ```

#### 部署模型

成功创建 TensorRT 引擎后，我们如何使用 TensorRT 运行它。

TensorRT runtime有两种类型：一种是与 C++ 和 Python 绑定的独立runtime，另一种与 TensorFlow 的本地集成。在本节中，我们将调用独立runtime的简化封装器（ONNXClassifierWrapper）。我们将生成一批随机的 "假 "数据，并使用ONNXClassifierWrapper在该批数据上运行推理。

1.  设置ONNXClassifierWrapper

    ```python
    from onnx_helper import ONNXClassifierWrapper 
    N_CLASSES = 1000 # 我们的 ResNet-50 在 1000 个类别的 ImageNet 任务上进行训练 
    trt_model = ONNXClassifierWrapper("resnet_engine.trt", [BATCH_SIZE, N_CLASSES], target_dtype = PRECISION)
    ```
2.  设置模型输入

    ```
    BATCH_SIZE=32 
    dummy_input_batch = np.zeros((BATCH_SIZE, 224, 224, 3), dtype = PRECISION)
    ```
3.  向我们的引擎输入一批数据，然后得到预测结果。

    ```
    predictions = trt_model.predict(dummy_input_batch)
    ```

请注意，在运行第一个batch之前，包装器不会加载和初始化引擎，因此该batch处理通常需要比较长的一段时间。

### 使用 TensorRT runtime API

在模型转换和部署方面，使用 TensorRT runtime API是性能最好、最可定制的选择之一，该程序接口有 C++ 和 Python 版本。

TensorRT 包含一个带有 C++ 和 Python 绑定的独立运行时，与使用 TF-TRT 集成相比，其性能和可定制性通常更高。C++ API 的开销较低，但 Python API 与 Python 数据加载器以及 NumPy 和 SciPy 等库配合良好，更易于用于原型设计、调试和测试。

下面的教程说明了如何使用 TensorRT C++ 和 Python API 对图像进行语义分割。该任务使用了一个以 ResNet-101 为骨干的全卷积模型。该模型可接受任意大小的图像，并为每个像素生成相应的预测结果。

本教程包括以下步骤：

1. 导出ONNX模型，并使用trtexec生成TensorRT引擎
2. 使用TensorRT C++ API进行运行时的推理
3. 使用TensorRT C++ API进行运行时的推理

#### 导出ONNX模型，并使用trtexec生成TensorRT引擎

1.  从[TensorRT 开源软件仓库](http://github.com/NVIDIA/TensorRT)下载快速入门教程的源代码。

    ```sh
    git clone https://github.com/NVIDIA/TensorRT.git 
    cd TensorRT/quickstart
    ```
2.  将[预先训练好的 FCN-ResNet-101](https://pytorch.org/hub/pytorch\_vision\_fcn\_resnet101/)模型从torch.hub转换到 ONNX。

























