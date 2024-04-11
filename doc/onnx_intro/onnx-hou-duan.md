# ONNX后端

### What is an ONNX backend

ONNX 后端是一个可以运行 ONNX 模型的库。由于许多深度学习框架已经存在，您不需要从头开始创建一切。您可以创建一个转换器，将 ONNX 模型转换为相应框架的特定表示形式，然后委托框架执行。例如，onnx-caffe2（作为 caffe2 的一部分）、onnx-coreml 和 onnx-tensorflow 都是作为转换器实现的。

### Unified backend interface

ONNX 在 `onnx/backend/base.py` 中定义了统一的（Python）后端接口。

该接口有三个核心概念: `Device`, `Backend` and `BackendRep`.

* `Device` 是对各种硬件的轻量级抽象， e.g., CPU, GPU, etc.
*   `Backend` 是一个实体，它接受 ONNX 模型的输入，执行计算，然后返回输出。

    对于一次性执行，用户可以使用 `run_node` 和 `run_model` 快速获得结果。

    对于重复多次的执行，用户应使用 `prepare`，即后端为重复执行模型做的所有准备工作（如加载初始化器），并返回一个 BackendRep 句柄。
* BackendRep 是后台准备重复多次的执行模型后返回的句柄。然后，用户将输入信息传递给 BackendRep 的运行函数，以获取相应的结果。

请注意，尽管 ONNX 统一后端接口是用 Python 定义的，但后端并不需要用 Python 实现。例如，可以用 C++ 创建后端，并使用 pybind11 或 cython 等工具来实现接口。

### ONNX backend test

ONNX 提供标准的后端测试套件，以协助后端实施验证。我们强烈建议每个 ONNX 后端都运行该测试。

将 ONNX 后端测试套件集成到您的 CI 中非常简单。以下示例展示了如何集成后端：

* [onnx-caffe2 onnx backend test](https://github.com/pytorch/pytorch/blob/master/caffe2/python/onnx/tests/onnx\_backend\_test.py)
* [onnx-tensorflow onnx backend test](https://github.com/onnx/onnx-tensorflow/blob/main/test/backend/test\_onnx\_backend.py)
* [onnx-coreml onnx backend test](https://github.com/onnx/onnx-coreml/blob/master/tests/onnx\_backend\_models\_test.py)

如果安装了 pytest，就能在运行 ONNX 后端测试后获得覆盖率报告：

```sh
---------- onnx coverage: ----------
Operators (passed/loaded/total): 21/21/70
------------------------------------
╒════════════════════╤════════════════════╕
│ Operator           │ Attributes         │
│                    │ (name: #values)    │
╞════════════════════╪════════════════════╡
│ Slice              │ axes: 2            │
│                    │ ends: 3            │
│                    │ starts: 3          │
├────────────────────┼────────────────────┤
│ Constant           │ value: 1           │
├────────────────────┼────────────────────┤
│ Concat             │ axis: 0            │
├────────────────────┼────────────────────┤
│ Conv               │ group: 6           │
│                    │ kernel_shape: 5    │
│                    │ pads: 4            │
│                    │ strides: 3         │
│                    │ auto_pad: 0        │
│                    │ dilations: 0       │
├────────────────────┼────────────────────┤
│ Reshape            │ shape: 9           │
├────────────────────┼────────────────────┤
│ BatchNormalization │ consumed_inputs: 1 │
│                    │ epsilon: 2         │
│                    │ is_test: 1         │
│                    │ momentum: 0        │
│                    │ spatial: 0         │
├────────────────────┼────────────────────┤
│ Dropout            │ is_test: 1         │
│                    │ ratio: 2           │
├────────────────────┼────────────────────┤
│ MaxPool            │ kernel_shape: 2    │
│                    │ pads: 3            │
│                    │ strides: 2         │
│                    │ auto_pad: 0        │
│                    │ dilations: 0       │
├────────────────────┼────────────────────┤
│ Transpose          │ perm: 1            │
├────────────────────┼────────────────────┤
│ MatMul             │ No attributes      │
├────────────────────┼────────────────────┤
│ Relu               │ No attributes      │
├────────────────────┼────────────────────┤
│ LRN                │ alpha: 2           │
│                    │ beta: 1            │
│                    │ bias: 2            │
│                    │ size: 1            │
├────────────────────┼────────────────────┤
│ Add                │ axis: 1            │
│                    │ broadcast: 1       │
├────────────────────┼────────────────────┤
│ Abs                │ No attributes      │
├────────────────────┼────────────────────┤
│ Pad                │ mode: 3            │
│                    │ paddings: 2        │
│                    │ value: 1           │
├────────────────────┼────────────────────┤
│ Softmax            │ axis: 0            │
├────────────────────┼────────────────────┤
│ GlobalAveragePool  │ No attributes      │
├────────────────────┼────────────────────┤
│ Mul                │ axis: 1            │
│                    │ broadcast: 1       │
├────────────────────┼────────────────────┤
│ Sum                │ No attributes      │
├────────────────────┼────────────────────┤
│ Gemm               │ broadcast: 1       │
│                    │ transB: 1          │
│                    │ alpha: 0           │
│                    │ beta: 0            │
│                    │ transA: 0          │
├────────────────────┼────────────────────┤
│ AveragePool        │ kernel_shape: 3    │
│                    │ pads: 3            │
│                    │ strides: 2         │
│                    │ auto_pad: 0        │
╘════════════════════╧════════════════════╛
```

`Operators (passed/loaded/total)` 一行中的数字：21/21/70 表示后端所有测试用例中涉及的 21 个运算符已通过，ONNX 后端的所有测试用例中涉及 21 个运算符，ONNX 共有 70 个运算符。
