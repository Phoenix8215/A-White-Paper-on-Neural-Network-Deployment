# ONNX中的广播

## Broadcasting in ONNX

在 ONNX 中，只要输入的张量可以广播到相同的形状，那么按元素排列的算子就可以接受不同形状的输入。ONNX 支持两种广播方式：<mark style="color:red;">多向广播和单向广播</mark>。我们将在下文中分别介绍这两种广播类型。

### 多向广播

在 ONNX 中，如果以下条件之一为真，则一组张量可以多向广播到相同的形状：

* 所有张量的形状完全相同。
* 所有张量都有相同的维数。
* 维数少的张量可以在其形状前添加长度为 1 的维数，以满足性质 2 的要求。

例如，多向广播支持以下张量形状：

* shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar ==> shape(result) = (2, 3, 4, 5)
* shape(A) = (2, 3, 4, 5), shape(B) = (5,), ==> shape(result) = (2, 3, 4, 5)
* shape(A) = (4, 5), shape(B) = (2, 3, 4, 5), ==> shape(result) = (2, 3, 4, 5)
* shape(A) = (1, 4, 5), shape(B) = (2, 3, 1, 1), ==> shape(result) = (2, 3, 4, 5)
* shape(A) = (3, 4, 5), shape(B) = (2, 1, 1, 1), ==> shape(result) = (2, 3, 4, 5)

Multidirectional broadcasting is the same as [Numpy’s broadcasting](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html#general-broadcasting-rules).

ONNX 的下列算子支持多向广播：

* [Add](https://onnx.ai/onnx/repo-docs/Broadcasting.html#Operators.md#Add)
* [And](https://onnx.ai/onnx/repo-docs/Broadcasting.html#Operators.md#And)
* [Div](https://onnx.ai/onnx/repo-docs/Broadcasting.html#Operators.md#Div)
* [Equal](https://onnx.ai/onnx/repo-docs/Broadcasting.html#Operators.md#Equal)
* [Greater](https://onnx.ai/onnx/repo-docs/Broadcasting.html#Operators.md#Greater)
* [Less](https://onnx.ai/onnx/repo-docs/Broadcasting.html#Operators.md#Less)
* [Max](https://onnx.ai/onnx/repo-docs/Broadcasting.html#Operators.md#Max)
* [Mean](https://onnx.ai/onnx/repo-docs/Broadcasting.html#Operators.md#Mean)
* [Min](https://onnx.ai/onnx/repo-docs/Broadcasting.html#Operators.md#Min)
* [Mul](https://onnx.ai/onnx/repo-docs/Broadcasting.html#Operators.md#Mul)
* [Or](https://onnx.ai/onnx/repo-docs/Broadcasting.html#Operators.md#Or)
* [Pow](https://onnx.ai/onnx/repo-docs/Broadcasting.html#Operators.md#Pow)
* [Sub](https://onnx.ai/onnx/repo-docs/Broadcasting.html#Operators.md#Sub)
* [Sum](https://onnx.ai/onnx/repo-docs/Broadcasting.html#Operators.md#Sum)
* [Where](https://onnx.ai/onnx/repo-docs/Broadcasting.html#Operators.md#Where)
* [Xor](https://onnx.ai/onnx/repo-docs/Broadcasting.html#Operators.md#Xor)

### 单向广播

在 ONNX 中，如果以下条件之一为真，则张量 B 可单向广播到张量 A：

* 张量 A 和张量 B 的形状完全相同。
* 张量 A 和张量 B 的维数相同。
* 张量 B 的维数少，可以在 B 的形状前添加长度为 1 的维数，以满足性质 2。

当发生单向广播时，输出的形状与 A 的形状相同（即两个输入张量中较大的那个shape）。

在以下示例中，张量 B 可单向广播到张量 A：

* shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar ==> shape(result) = (2, 3, 4, 5)
* shape(A) = (2, 3, 4, 5), shape(B) = (5,), ==> shape(result) = (2, 3, 4, 5)
* shape(A) = (2, 3, 4, 5), shape(B) = (2, 1, 1, 5), ==> shape(result) = (2, 3, 4, 5)
* shape(A) = (2, 3, 4, 5), shape(B) = (1, 3, 1, 5), ==> shape(result) = (2, 3, 4, 5)

ONNX 的下列算子支持单向广播：

* [Gemm](https://onnx.ai/onnx/repo-docs/Broadcasting.html#Operators.md#Gemm)
* [PRelu](https://onnx.ai/onnx/repo-docs/Broadcasting.html#Operators.md#PRelu)
