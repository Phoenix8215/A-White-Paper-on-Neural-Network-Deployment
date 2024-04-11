# pytorch-quantization文档

### Basic Functionalities

#### Quantization function

`fake_tensor_quant` 返回假量化张量（浮点值）。`tensor_quant` 返回量化张量（整数值）和比例。

```python
tensor_quant(inputs, amax, num_bits=8, output_dtype=torch.float, unsigned=False)
fake_tensor_quant(inputs, amax, num_bits=8, output_dtype=torch.float, unsigned=False)
```

Example:

```python
from pytorch_quantization import tensor_quant

# Generate random input. With fixed seed 12345, x should be
# tensor([0.9817, 0.8796, 0.9921, 0.4611, 0.0832, 0.1784, 0.3674, 0.5676, 0.3376, 0.2119])
torch.manual_seed(12345)
x = torch.rand(10)

# fake quantize tensor x. fake_quant_x will be
# tensor([0.9843, 0.8828, 0.9921, 0.4609, 0.0859, 0.1797, 0.3672, 0.5703, 0.3359, 0.2109])
fake_quant_x = tensor_quant.fake_tensor_quant(x, x.abs().max())

# quantize tensor x. quant_x will be
# tensor([126., 113., 127.,  59.,  11.,  23.,  47.,  73.,  43.,  27.])
# with scale=128.0057
quant_x, scale = tensor_quant.tensor_quant(x, x.abs().max())
```

这两个函数的反向传播函数被定义为[直通估计器（STE）](https://arxiv.org/pdf/1308.3432.pdf)。

#### Descriptor and quantizer

`QuantDescriptor` 定义了张量的量化方式。还有一些预定义的 `QuantDescriptor`，例如 `QUANT_DESC_8BIT_PER_TENSOR` 和 `QUANT_DESC_8BIT_CONV2D_WEIGHT_PER_CHANNEL`。

`TensorQuantizer` 是专门用来量化张量的一个模块，量化方式由 `QuantDescriptor` 定义。from pytorch\_quantization.tensor\_quant import QuantDescriptor

```python
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization.nn.modules.tensor_quantizer import TensorQuantizer

quant_desc = QuantDescriptor(num_bits=4, fake_quant=False, axis=(0), unsigned=True)
quantizer = TensorQuantizer(quant_desc)

torch.manual_seed(12345)
x = torch.rand(10, 9, 8, 7)

quant_x = quantizer(x)
```

来看看源码中对这几个参数的解释：

```python
   """Supportive descriptor of quantization

    Describe how a tensor should be quantized. A QuantDescriptor and a tensor defines a quantized tensor.

    Args:
        num_bits: An integer. Number of bits of quantization. It is used to calculate scaling factor. Default 8.
        name: Seems a nice thing to have

    Keyword Arguments:
        fake_quant: A boolean. If True, use fake quantization mode. Default True.
        axis: None, int or tuple of int. axes which will have its own max for computing scaling factor.
            If None (the default), use per tensor scale.
            Must be in the range [-rank(input_tensor), rank(input_tensor)).
            e.g. For a KCRS weight tensor, quant_axis=(0) will yield per channel scaling.
            Default None.
        amax: A float or list/ndarray of floats of user specified absolute max range(绝对值最大的元素). If supplied,
            ignore quant_axis and use this to quantize. If learn_amax is True, will be used to initialize
            learnable amax. Default None.
        learn_amax: A boolean. If True, learn amax. Default False.
        scale_amax: A float. If supplied, multiply amax by scale_amax. Default None. It is useful for some
            quick experiment.
        calib_method: A string. One of ["max", "histogram"] indicates which calibration to use. Except the simple
            max calibration, other methods are all hisogram based. Default "max".
        unsigned: A Boolean. If True, use unsigned. Default False.

    Raises:
        TypeError: If unsupported type is passed in.

    Read-only properties:
        - fake_quant:
        - name:
        - learn_amax:
        - scale_amax:
        - axis:
        - calib_method:
        - num_bits:
        - amax:
        - unsigned:
    """
```

#### Quantized module

该模块有两种主要类型：Conv 和 Linear。这两种模块都可以替代 torch.nn 版本，并对权重和激活进行量化。除了原始模块的参数外，这两种模块都需要 quant\_desc\_input 和 quant\_desc\_weight。

```python
from torch import nn

from pytorch_quantization import tensor_quant
import pytorch_quantization.nn as quant_nn

# pytorch's module
fc1 = nn.Linear(in_features, out_features, bias=True)
conv1 = nn.Conv2d(in_channels, out_channels, kernel_size)

# quantized version
quant_fc1 = quant_nn.Linear(
    in_features, out_features, bias=True,
    quant_desc_input=tensor_quant.QUANT_DESC_8BIT_PER_TENSOR,
    quant_desc_weight=tensor_quant.QUANT_DESC_8BIT_LINEAR_WEIGHT_PER_ROW)
quant_conv1 = quant_nn.Conv2d(
    in_channels, out_channels, kernel_size,
    quant_desc_input=tensor_quant.QUANT_DESC_8BIT_PER_TENSOR,
    quant_desc_weight=tensor_quant.QUANT_DESC_8BIT_CONV2D_WEIGHT_PER_CHANNEL)
```

从源码查看提前定义好的`descriptors`

```python
QUANT_DESC_8BIT_PER_TENSOR = QuantDescriptor(num_bits=8)
QUANT_DESC_UNSIGNED_8BIT_PER_TENSOR = QuantDescriptor(num_bits=8, unsigned=True)
QUANT_DESC_8BIT_CONV1D_WEIGHT_PER_CHANNEL = QuantDescriptor(num_bits=8, axis=(0))
QUANT_DESC_8BIT_CONV2D_WEIGHT_PER_CHANNEL = QuantDescriptor(num_bits=8, axis=(0))
QUANT_DESC_8BIT_CONV3D_WEIGHT_PER_CHANNEL = QuantDescriptor(num_bits=8, axis=(0))
QUANT_DESC_8BIT_LINEAR_WEIGHT_PER_ROW = QuantDescriptor(num_bits=8, axis=(0))
QUANT_DESC_8BIT_CONVTRANSPOSE1D_WEIGHT_PER_CHANNEL = QuantDescriptor(num_bits=8, axis=(0))
QUANT_DESC_8BIT_CONVTRANSPOSE2D_WEIGHT_PER_CHANNEL = QuantDescriptor(num_bits=8, axis=(0))
QUANT_DESC_8BIT_CONVTRANSPOSE3D_WEIGHT_PER_CHANNEL = QuantDescriptor(num_bits=8, axis=(0))
```

### Post training quantization

只需调用 `quant_modules.initialize()`，即可对模型进行训练后量化(PTQ)。

```python
fimport torch
import torchvision
from pytorch_quantization import tensor_quant, quant_modules
from pytorch_quantization import nn as quant_nn

quant_modules.initialize()

model = torchvision.models.resnet18()
model.cuda()

inputs = torch.randn(1, 3, 224, 224, device='cuda')
quant_nn.TensorQuantizer.use_fb_fake_quant = True
torch.onnx.export(model, inputs, 'quant_resnet18.onnx', opset_version=13)
```

上述示例代码通过指定 `quant_nn.TensorQuantizer.use_fb_fake_quant` 来将 `resnet18` 模型中的所有节点替换为 QDQ 算子，并导出为 ONNX 格式的模型文件，实现了模型的量化。值得注意的是：

* `quant_modules.initialize()` 函数会把 `PyTorch-Quantization` 库中所有的量化算子按照数据类型、位宽等特性进行分类，并将其保存在全局变量 `_DEFAULT_QUANT_MAP` 中&#x20;
* 导出的带有 QDQ 节点的 ONNX 模型中，对于输入 `input` 的整个 `tensor` 是共用一个 `scale`，而对于权重 `weight` 则是每个 `channel` 共用一个 `scale`&#x20;
* 导出的带有 QDQ 节点的 ONNX 模型中，`x_zero_point` 是之前提到的偏移量，其值为0，因为整个量化过程是对称量化，其偏移量 Z 为0

#### Calibration

校准是 TensorRT 的术语，即把数据样本传递给量化器，并决定amax(绝对值最大的元素)的最佳值。我们支持 4 种校准方法：

* `max`: 只需使用全局最大绝对值
* `entropy`: TensorRT 的熵校准
* `percentile`: 根据给定的百分位数去除离群值。
* `mse`: 基于 MSE（均方误差）的校准

以下示例校准方法设置为 `mse`：

```python
# Find the TensorQuantizer and enable calibration
for name, module in model.named_modules():
    if name.endswith('_quantizer'):
        module.enable_calib()
        module.disable_quant()  # Use full precision data to calibrate

# Feeding data samples
model(x)
# ...

# Finalize calibration
for name, module in model.named_modules():
    if name.endswith('_quantizer'):
        module.load_calib_amax()
        module.disable_calib()
        module.enable_quant()

# If running on GPU, it needs to call .cuda() again because new tensors will be created by calibration process
model.cuda()

# Keep running the quantized model
# ...
```

{% hint style="info" %}
在将模型导出到 ONNX 之前需要进行校准。
{% endhint %}

### Quantization Aware Training

`Quantization Aware Training` 基于直通估计（STE）导数近似。它有时被称为 "量化感知训练"。我们不使用这个名称，因为它并不反映下面的假设。如果说有什么不同的话，那就是由于采用了 STE 近似方法，所以训练 "不感知 "量化。

校准完成后，量化感知训练只需选择一个训练计划，然后继续训练校准后的模型。通常，它不需要微调很长时间。我们通常使用原始训练计划的 10% 左右，从初始训练学习率的 1% 开始，使用余弦退火学习率计划，按照余弦周期的一半递减，直到初始微调学习率的 1% （初始训练学习率的 0.01%）。

#### Some recommendations

量化感知训练（本质上是一个离散数值优化问题）在数学上并不是一个已经解决的问题。根据我们的经验，这里有一些建议：

* 要使 STE 近似效果良好，最好使用较小的学习率。大的学习率更有可能扩大 STE 近似引入的方差，破坏训练好的网络。
* 在训练过程中不要改变量化表示（scale），至少不要过于频繁。每一步都改变刻度，实际上就等于每一步都改变数据格式（e8m7、e5m10、e3m4 等），这很容易影响收敛性。

### Export to ONNX

导出到 ONNX 的目的是部署到 TensorRT，而不是 ONNX 运行时。因此，我们只将假量化模型导出为 TensorRT 可以接受的形式。假量化将分解为一对 `QuantizeLinear/DequantizeLinear` ONNX 操作。TensorRT 将获取生成的 ONNX 图，并以最优化的方式在 int8 中执行。

{% hint style="info" %}
目前，我们只支持导出 int8 和 fp8 假量化模块。此外，量化模块在导出到 ONNX 之前需要进行校准。
{% endhint %}

假量化模型可以像其他 Pytorch 模型一样导出到 ONNX。有关将 Pytorch 模型导出到 ONNX 的更多信息，请访问 [torch.onnx](https://pytorch.org/docs/stable/onnx.html?highlight=onnx#module-torch.onnx)。例如

```python
import pytorch_quantization
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules

quant_modules.initialize()
model = torchvision.models.resnet50()

# load the calibrated model
state_dict = torch.load("quant_resnet50-entropy-1024.pth", map_location="cpu")
model.load_state_dict(state_dict)
model.cuda()

dummy_input = torch.randn(128, 3, 224, 224, device='cuda')

input_names = [ "actual_input_1" ]
output_names = [ "output1" ]

with pytorch_quantization.enable_onnx_export():
     # enable_onnx_checker needs to be disabled. See notes below.
     torch.onnx.export(
         model, dummy_input, "quant_resnet50.onnx", verbose=True, opset_version=10, enable_onnx_checker=False
         )
```

{% hint style="info" %}
请注意，`opset13` 中的 `QuantizeLinear` 和 `DequantizeLinear` 都添加了`axis`。
{% endhint %}

### Quantizing Resnet50

#### Create a quantized model

```python
import torch
import torch.utils.data
from torch import nn

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor

from torchvision import models

sys.path.append("path to torchvision/references/classification/")
from train import evaluate, train_one_epoch, load_data
```

**Adding quantized modules**

第一步是在神经网络图中添加量化模块。例如，`quant_nn.QuantLinear` 可以用来替代 `nn.Linear`。这些量化层可以通过 "`Monkey-patching` "自动替换，也可以通过手动修改模型定义来替换。

自动层替换是通过 `quant_modules` 完成的。应在创建模型前调用它。

```python
from pytorch_quantization import quant_modules
quant_modules.initialize()
```

这将适用于每个模块的所有实例。如果不希望对所有模块进行量化，则应手动替换已量化的模块。也可以使用 `quant_nn.TensorQuantizer` 将独立的量化器添加到模型中。

{% hint style="info" %}
`Monkey-patching` 是一种编程术语，指的是在运行时动态修改或扩展现有的代码，通常是在不修改原始源代码的情况下实现。这种技术允许开发者在程序运行过程中修改类、函数、方法或模块的行为，以满足特定的需求或修复 `bug`。
{% endhint %}

#### Post training quantization

为了提高推理效率，我们希望为每个量化器选择一个固定的范围。从预先训练好的模型开始，最简单的方法就是校准。

**Calibration**

<mark style="color:red;">我们对激活值使用基于直方图的校准方式，对模型权重使用基于最大值的校准方式。</mark>

```python
quant_desc_input = QuantDescriptor(calib_method='histogram')
quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)

model = models.resnet50(pretrained=True)
model.cuda()
```

要收集激活值的直方图，我们必须向模型输入样本数据。首先，按照训练脚本创建 ImageNet 数据加载器。然后，在每个量化器中启用校准，并将训练数据输入模型。1024 个样本（2 批，每批 512 个）足以估计激活值的分布。

```python
data_path = "PATH to imagenet"
batch_size = 512

traindir = os.path.join(data_path, 'train')
valdir = os.path.join(data_path, 'val')
dataset, dataset_test, train_sampler, test_sampler = load_data(traindir, valdir, False, False)

data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size,
    sampler=train_sampler, num_workers=4, pin_memory=True)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=batch_size,
    sampler=test_sampler, num_workers=4, pin_memory=True)
```

```python
 def collect_stats(model, data_loader, num_batches):
     """Feed data to the network and collect statistic"""

     # Enable calibrators
     for name, module in model.named_modules():
         if isinstance(module, quant_nn.TensorQuantizer):
             if module._calibrator is not None:
                 module.disable_quant()
                 module.enable_calib()
             else:
                 module.disable()

     for i, (image, _) in tqdm(enumerate(data_loader), total=num_batches):
         model(image.cuda())
         if i >= num_batches:
             break

     # Disable calibrators
     for name, module in model.named_modules():
         if isinstance(module, quant_nn.TensorQuantizer):
             if module._calibrator is not None:
                 module.enable_quant()
                 module.disable_calib()
             else:
                 module.enable()

 def compute_amax(model, **kwargs):
     # Load calib result
     for name, module in model.named_modules():
         if isinstance(module, quant_nn.TensorQuantizer):
             if module._calibrator is not None:
                 if isinstance(module._calibrator, calib.MaxCalibrator):
                     module.load_calib_amax()
                 else:
                     module.load_calib_amax(**kwargs)
             print(F"{name:40}: {module}")
     model.cuda()

# It is a bit slow since we collect histograms on CPU
 with torch.no_grad():
     collect_stats(model, data_loader, num_batches=2)
     compute_amax(model, method="percentile", percentile=99.99)
```

校准完成后，量化器将设置一个最大值（`amax`），表示量化空间中可表示的绝对最大输入值。默认情况下，权重`scale`为`per channel`的，而激活函数的`scale`为每个张量。我们可以通过打印每个 `TensorQuantizer` 模块来查看 `amax`。

```
conv1._input_quantizer                  : TensorQuantizer(8bit fake per-tensor amax=2.6400 calibrator=MaxCalibrator(track_amax=False) quant)
conv1._weight_quantizer                 : TensorQuantizer(8bit fake axis=(0) amax=[0.0000, 0.7817](64) calibrator=MaxCalibrator(track_amax=False) quant)
layer1.0.conv1._input_quantizer         : TensorQuantizer(8bit fake per-tensor amax=6.8645 calibrator=MaxCalibrator(track_amax=False) quant)
layer1.0.conv1._weight_quantizer        : TensorQuantizer(8bit fake axis=(0) amax=[0.0000, 0.7266](64) calibrator=MaxCalibrator(track_amax=False) quant)
...
```

**Evaluate the calibrated model**

接下来，我们将在 ImageNet 验证集上评估训练后量化(PTQ)模型的分类准确性。

```python
criterion = nn.CrossEntropyLoss()
with torch.no_grad():
    evaluate(model, criterion, data_loader_test, device="cuda", print_freq=20)

# Save the model
torch.save(model.state_dict(), "/tmp/quant_resnet50-calibrated.pth")
```

top-1 的准确率为 76.1%，接近预训练模型 76.2% 的准确率。

**Use different calibration**

We can try different calibrations without recollecting the histograms, and see which one gets the best accuracy.

```
with torch.no_grad():
    compute_amax(model, method="percentile", percentile=99.9)
    evaluate(model, criterion, data_loader_test, device="cuda", print_freq=20)

with torch.no_grad():
    for method in ["mse", "entropy"]:
        print(F"{method} calibration")
        compute_amax(model, method=method)
        evaluate(model, criterion, data_loader_test, device="cuda", print_freq=20)
```

MSE and entropy should both get over 76%. 99.9% clips too many values for resnet50 and will get slightly lower accuracy.

#### Quantization Aware Training

Optionally, we can fine-tune the calibrated model to improve accuracy further.

```
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

# Training takes about one and half hour per epoch on a single V100
train_one_epoch(model, criterion, optimizer, data_loader, "cuda", 0, 100)

# Save the model
torch.save(model.state_dict(), "/tmp/quant_resnet50-finetuned.pth")
```

After one epoch of fine-tuning, we can achieve over 76.4% top-1 accuracy. Fine-tuning for more epochs with learning rate annealing can improve accuracy further. For example, fine-tuning for 15 epochs with cosine annealing starting with a learning rate of 0.001 can get over 76.7%. It should be noted that the same fine-tuning schedule will improve the accuracy of the unquantized model as well.

**Further optimization**

For efficient inference on TensorRT, we need know more details about the runtime optimization. TensorRT supports fusion of quantizing convolution and residual add. The new fused operator has two inputs. Let us call them conv-input and residual-input. Here the fused operator’s output precision must match the residual input precision. When there is another quantizing node after the fused operator, we can insert a pair of quantizing/dequantizing nodes between the residual-input and the Elementwise-Addition node, so that quantizing node after the Convolution node is fused with the Convolution node, and the Convolution node is completely quantized with INT8 input and output. We cannot use automatic monkey-patching to apply this optimization and we need to manually insert the quantizing/dequantizing nodes.

First create a copy of resnet.py from [https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py), modify the constructor, add explicit bool flag ‘quantize’

```
def resnet50(pretrained: bool = False, progress: bool = True, quantize: bool = False, **kwargs: Any) -> ResNet:
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, quantize, **kwargs)
def _resnet(arch: str, block: Type[Union[BasicBlock, Bottleneck]], layers: List[int], pretrained: bool, progress: bool,
            quantize: bool, **kwargs: Any) -> ResNet:
    model = ResNet(block, layers, quantize, **kwargs)
class ResNet(nn.Module):
    def __init__(self,
                 block: Type[Union[BasicBlock, Bottleneck]],
                 layers: List[int],
                 quantize: bool = False,
                 num_classes: int = 1000,
                 zero_init_residual: bool = False,
                 groups: int = 1,
                 width_per_group: int = 64,
                 replace_stride_with_dilation: Optional[List[bool]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super(ResNet, self).__init__()
        self._quantize = quantize
```

When this `self._quantize` flag is set to `True`, we need replace all the `nn.Conv2d` with `quant_nn.QuantConv2d`.

```
def conv3x3(in_planes: int,
            out_planes: int,
            stride: int = 1,
            groups: int = 1,
            dilation: int = 1,
            quantize: bool = False) -> nn.Conv2d:
    """3x3 convolution with padding"""
    if quantize:
        return quant_nn.QuantConv2d(in_planes,
                                    out_planes,
                                    kernel_size=3,
                                    stride=stride,
                                    padding=dilation,
                                    groups=groups,
                                    bias=False,
                                    dilation=dilation)
    else:
        return nn.Conv2d(in_planes,
                         out_planes,
                         kernel_size=3,
                         stride=stride,
                         padding=dilation,
                         groups=groups,
                         bias=False,
                         dilation=dilation)
  def conv1x1(in_planes: int, out_planes: int, stride: int = 1, quantize: bool = False) -> nn.Conv2d:
      """1x1 convolution"""
      if quantize:
          return quant_nn.QuantConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
      else:
          return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
```

The residual conv add can be find both in both `BasicBlock` and `Bottleneck`. We need first declare quantization node in the `__init__` function.

```
def __init__(self,
             inplanes: int,
             planes: int,
             stride: int = 1,
             downsample: Optional[nn.Module] = None,
             groups: int = 1,
             base_width: int = 64,
             dilation: int = 1,
             norm_layer: Optional[Callable[..., nn.Module]] = None,
             quantize: bool = False) -> None:
    # other code...
    self._quantize = quantize
    if self._quantize:
        self.residual_quantizer = quant_nn.TensorQuantizer(quant_nn.QuantConv2d.default_quant_desc_input)
```

Finally we need patch the `forward` function in both `BasicBlock` and `Bottleneck`, inserting extra quantization/dequantization nodes here.

```
def forward(self, x: Tensor) -> Tensor:
    # other code...
    if self._quantize:
        out += self.residual_quantizer(identity)
    else:
        out += identity
    out = self.relu(out)

    return out
```

The final resnet code with residual quantized can be found in [https://github.com/NVIDIA/TensorRT/blob/master/tools/pytorch-quantization/examples/torchvision/models/classification/resnet.py](https://github.com/NVIDIA/TensorRT/blob/master/tools/pytorch-quantization/examples/torchvision/models/classification/resnet.py)

### Creating Custom Quantized Modules

There are several quantized modules provided by the quantization tool as follows:

* QuantConv1d, QuantConv2d, QuantConv3d, QuantConvTranspose1d, QuantConvTranspose2d, QuantConvTranspose3d
* QuantLinear
* QuantAvgPool1d, QuantAvgPool2d, QuantAvgPool3d, QuantMaxPool1d, QuantMaxPool2d, QuantMaxPool3d

To quantize a module, we need to quantize the input and weights if present. Following are 3 major use-cases:

1. Create quantized wrapper for modules that have only inputs
2. Create quantized wrapper for modules that have inputs as well as weights.
3. Directly add the `TensorQuantizer` module to the inputs of an operation in the model graph.

The first two methods are very useful if it’s needed to automatically replace the original modules (nodes in the graph) with their quantized versions. The third method could be useful when it’s required to manually add the quantization to the model graph at very specific places (more manual, more control).

Let’s see each use-case with examples below.

#### Quantizing Modules With Only Inputs

A suitable example would be quantizing the `pooling` module variants.

Essentially, we need to provide a wrapper function that takes the original module and adds the `TensorQuantizer` module around it so that the input is first quantized and then fed into the original module.

* Create the wrapper by subclassing the original module (`pooling.MaxPool2d`) along with the utilities module (`_utils.QuantInputMixin`).

```
class QuantMaxPool2d(pooling.MaxPool2d, _utils.QuantInputMixin):
```

* The `__init__.py` function would call the original module’s init function and provide it with the corresponding arguments. There would be just one additional argument using `**kwargs` which contains the quantization configuration information. The `QuantInputMixin` utility contains the method `pop_quant_desc_in_kwargs` which extracts this configuration information from the input or returns a default if that input is `None`. Finally the `init_quantizer` method is called that initializes the `TensorQuantizer` module which would quantize the inputs.

```
def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
             return_indices=False, ceil_mode=False, **kwargs):
    super(QuantMaxPool2d, self).__init__(kernel_size, stride, padding, dilation,
                                         return_indices, ceil_mode)
    quant_desc_input = _utils.pop_quant_desc_in_kwargs(self.__class__, input_only=True, **kwargs)
    self.init_quantizer(quant_desc_input)
```

* After the initialization, the `forward` function needs to be defined in our wrapper module that would actually quantize the inputs using the `_input_quantizer` that was initialized in the `__init__` function forwarding the inputs to the base module using `super` call.

```
def forward(self, input):
    quant_input = self._input_quantizer(input)
    return super(QuantMaxPool2d, self).forward(quant_input)
```

* Finally, we need to define a getter method for the `_input_quantizer`. This could, for example, be used to disable the quantization for a particular module using `module.input_quantizer.disable()` which is helpful while experimenting with different layer quantization configuration.

```
@property
def input_quantizer(self):
    return self._input_quantizer
```

A complete quantized pooling module would look like following:

```
class QuantMaxPool2d(pooling.MaxPool2d, _utils.QuantInputMixin):
    """Quantized 2D maxpool"""
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                return_indices=False, ceil_mode=False, **kwargs):
        super(QuantMaxPool2d, self).__init__(kernel_size, stride, padding, dilation,
                                            return_indices, ceil_mode)
        quant_desc_input = _utils.pop_quant_desc_in_kwargs(self.__class__, input_only=True, **kwargs)
        self.init_quantizer(quant_desc_input)

    def forward(self, input):
        quant_input = self._input_quantizer(input)
        return super(QuantMaxPool2d, self).forward(quant_input)

    @property
    def input_quantizer(self):
        return self._input_quantizer
```

#### Quantizing Modules With Weights and Inputs

We give an example of quantizing the `torch.nn.Linear` module. It follows that the only additional change from the previous example of quantizing pooling modules is that we’d need to accomodate the quantization of weights in the Linear module.

* We create the quantized linear module as follows:

```
class QuantLinear(nn.Linear, _utils.QuantMixin):
```

* In the `__init__` function, we first use the `pop_quant_desc_in_kwargs` function to extract the quantization descriptors for both inputs and weights. Second, we initialize the `TensorQuantizer` modules for both inputs and weights using these quantization descriptors.

```
def __init__(self, in_features, out_features, bias=True, **kwargs):
        super(QuantLinear, self).__init__(in_features, out_features, bias)
        quant_desc_input, quant_desc_weight = _utils.pop_quant_desc_in_kwargs(self.__class__, **kwargs)

        self.init_quantizer(quant_desc_input, quant_desc_weight)
```

* Also, override the `forward` function call and pass the inputs and weights through `_input_quantizer` and `_weight_quantizer` respectively before passing the quantized arguments to the actual `F.Linear` call. This step adds the actual input/weight `TensorQuantizer` to the module and eventually the model.

```
def forward(self, input):
    quant_input = self._input_quantizer(input)
    quant_weight = self._weight_quantizer(self.weight)

    output = F.linear(quant_input, quant_weight, bias=self.bias)

    return output
```

* Also similar to the `Linear` module, we add the getter methods for the `TensorQuantizer` modules associated with inputs/weights. This could be used to, for example, disable the quantization mechanism by calling `module_obj.weight_quantizer.disable()`

```
@property
def input_quantizer(self):
    return self._input_quantizer

@property
def weight_quantizer(self):
    return self._weight_quantizer
```

* With all of the above changes, the quantized Linear module would look like following:

```
class QuantLinear(nn.Linear, _utils.QuantMixin):

    def __init__(self, in_features, out_features, bias=True, **kwargs):
        super(QuantLinear, self).__init__(in_features, out_features, bias)
        quant_desc_input, quant_desc_weight = _utils.pop_quant_desc_in_kwargs(self.__class__, **kwargs)

        self.init_quantizer(quant_desc_input, quant_desc_weight)

    def forward(self, input):
        quant_input = self._input_quantizer(input)
        quant_weight = self._weight_quantizer(self.weight)

        output = F.linear(quant_input, quant_weight, bias=self.bias)

        return output

    @property
    def input_quantizer(self):
        return self._input_quantizer

    @property
    def weight_quantizer(self):
        return self._weight_quantizer
```

#### Directly Quantizing Inputs In Graph

It is also possible to directly quantize graph inputs without creating wrappers as explained above.

Here’s an example:

```
test_input = torch.randn(1, 5, 5, 5, dtype=torch.double)

quantizer = TensorQuantizer(quant_nn.QuantLinear.default_quant_desc_input)

quant_input = quantizer(test_input)

out = F.adaptive_avg_pool2d(quant_input, 3)
```

Assume that there is a `F.adaptive_avg_pool2d` operation in the graph and we’d like to quantize this operation. In the example above, we use `TensorQuantizer(quant_nn.QuantLinear.default_quant_desc_input)` to define a quantizer that we then use to actually quantize the `test_input` and then feed this quantized input to the `F.adaptive_avg_pool2d` operation. Note that this quantizer is the same as the ones we used earlier while created quantized versions of torch’s modules.
