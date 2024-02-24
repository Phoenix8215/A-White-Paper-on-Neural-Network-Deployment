# 😆 关于ONNX的一些概念

ONNX 可以看作是为数学函数打造的编程语言。它定义了所有关于机器学习推理时所需要的必要操作。线性回归可以用以下方式表示：

```python
def onnx_linear_regressor(X):
    "ONNX code for a linear regression"
    return onnx.Add(onnx.MatMul(X, coefficients), bias)
```

这个例子与开发人员在 Python 中编写的表达式非常相似。因此，使用 ONNX 实现的机器学习模型通常被引用为 ONNX 图(ONNX Graph)。

<figure><img src="../../.gitbook/assets/image (1).png" alt=""><figcaption></figcaption></figure>

ONNX 旨在提供一种通用语言，任何机器学习框架都可以用它来描述自己的模型。使得生产中部署机器学习模型变得更容易。ONNX解释器（或**runtime**）可以在部署环境中专门针对某一任务进行部署和优化。有了 ONNX，我们就有可能建立一个独特的流程，将模型部署到生产环境中，并且独立于各种机器学习框架。_ONNX_实现了一个 python runtime，可用于评估 ONNX 模型和 ONNX 操作。它旨在阐明 ONNX 的语义，帮助理解和调试 ONNX 工具和转换器。它不用于实际生产，所以性能也不是目标（参见[onnx.reference](https://onnx.ai/onnx/api/reference.html#l-reference-implementation)）。

### <mark style="color:red;">Input, Output, Node, Initializer, Attributes</mark>

构建 ONNX Graph意味着使用 ONNX 语言或更准确地说 ONNX运算符实现一个函数。 一个线性回归模型可以这样编写。 **下面几行并不遵循 python 语法，只是一种用来说明模型的伪代码。**

```python
Input: float[M,K] x, float[K,N] a, float[N] c
Output: float[M, N] y

r = onnx.MatMul(x, a)
y = onnx.Add(r, c)
```

这段代码实现了一个函数`f(x, a, c)-> y = x @ a + c，` _x_、_a_、_c_是**inputs**，_y_是**outputs**。_r_ 是中间结果。_MatMul_ 和 _Add_ 是**nodes**。它们也有**inputs**和**outputs**。**node**是 ONNX Operator 中的一种类型，即运算符之一。此图是线性回归部分中的示例构建的。

图还可以有**initializer**。当输入（如线性回归系数）永不改变时，最有效的方法是将其转化为一个常量存储在图中。

```
Input: float[M,K] x
Initializer: float[K,N] a, float[N] c
Output: float[M, N] xac

xa = onnx.MatMul(x, a)
xac = onnx.Add(xa, c)
```

从外观上看，该图表如下图所示：右侧描述了运算符_Add_，其中第二个输入被定义为初始化器。该图是通过以下代码获得的 。



**属性**是运算符的固定参数。运算符[Gemm](https://onnx.ai/onnx/operators/onnx\_\_Gemm.html#l-onnx-doc-gemm)有四个属性：_alpha_、_beta_、_transA_、_transB_。除非运行时通过其应用程序接口允许，否则一旦加载了 ONNX 图形，这些值就不能更改，并在所有预测中保持不变。

### 使用[ protobuf#](broken-reference)进行序列化

将机器学习模型部署到生产环境中通常需要复制用于训练模型的整个生态系统，大多数情况下需要使用_docker_。 一旦模型转换为 ONNX，生产环境只需要一个运行时来执行用 ONNX 运算符定义的图形。该运行时可以用任何适合生产应用的语言开发，如 C、java、python、javascript、C#、Webassembly、ARM......

ONNX 使用_protobuf_将图形序列化为单个块（参见[解析和序列化](https://developers.google.com/protocol-buffers/docs/pythontutorial#parsing-and-serialization)）。其目的是尽可能优化模型大小。

### [元](broken-reference)数据\#

机器学习模型会不断更新。跟踪模型的版本、模型的作者以及模型的训练方式非常重要。ONNX 提供了在模型中存储额外数据的可能性。

*   **doc\_string**：该模型的人可读文档。

    允许 Markdown。
*   **域**：反向 DNS 名称，用于指示模型命名空间或域、

    例如，"org.onnx
*   **元数据道具**：以字典形式命名的元数据`map<string,string>、`

    `(值、 键）`应该是不同的。
*   **模型的作者**：以逗号分隔的姓名列表、

    范本作者和/或其组织的个人姓名。
*   **model\_license**：许可证的知名名称或 URL

    提供模型的依据。
* **model\_version**：模型本身的版本，以整数编码。
* **producer\_name**：用于生成模型的工具名称。
* **producer\_version**：生成工具的版本。
*   **培训信息**：可选扩展名，包含

    培训信息（见[TrainingInfoProto）](https://onnx.ai/onnx/api/classes.html#l-traininginfoproto)

### 可用运算符[和域](broken-reference)列表\#

主要列表在此说明：它融合了标准矩阵[运算符](https://onnx.ai/onnx/operators/index.html#l-onnx-operators)（Add、Sub、MatMul、Transpose、Greater、IsNaN、Shape、Reshape...）、还原（ReduceSum、ReduceMin...）、图像变换（Conv、MaxPool...）、深度神经网络层（RNN、DropOut...）、激活函数（Relu、Softmax...）。 ONNX 并不实现所有现有的机器学习运算符，否则运算符列表将是无限的。

运算符的主列表由一个域**ai.onnx** 标识。 一个**域**可定义为一组运算符。 该列表中的一些运算符专门用于文本，但难以满足需要。主列表中还缺少在标准机器学习中非常流行的基于树的模型，这些模型属于另一个域[**ai.onnx.ml**](http://ai.onnx.ml/)，它包括树基模型（TreeEnsemble Regressor, ...）、预处理（OneHotEncoder, LabelEncoder, ...）、SVM 模型（SVMRegressor, ...）和输入器（Imputer）。

ONNX 只定义了这两个域。但 onnx 库支持任何自定义域和运算符（请参阅[扩展性](broken-reference)）。

### 支持的[类型](broken-reference)\#

ONNX 规范针对使用张量进行数值计算进行了优化。_张量_是一个多维数组。其定义如下

* a 类型：元素类型，张量中的所有元素都相同
* 形状：包含所有维度的数组，该数组可以为空，一个维度可以为空
* 连续数组：表示所有值

该定义不包括_跨距_，也不包括根据现有张量定义张量视图的可能性。ONNX 张量是一个没有跨距的密集全数组。

#### 元素[类型](broken-reference)\#

ONNX 最初是为了帮助部署深度学习模型而开发的。 因此，其规格最初是针对浮点数（32 位）设计的。 当前版本支持所有常见类型。字典[TENSOR\_TYPE\_MAP](https://onnx.ai/onnx/api/mapping.html#l-onnx-types-mapping)提供了_ONNX_和[`numpy`](https://numpy.org/doc/stable/reference/index.html#module-numpy) 之间的对应关系。

```
import re
from onnx import TensorProto

reg = re.compile('^[0-9A-Z_]+$')

values = {}
for att in sorted(dir(TensorProto)):
    if att in {'DESCRIPTOR'}:
        continue
    if reg.match(att):
        values[getattr(TensorProto, att)] = att
for i, att in sorted(values.items()):
    si = str(i)
    if len(si) == 1:
        si = " " + si
    print("%s: onnx.TensorProto.%s" % (si, att))
```

```
 0: onnx.TensorProto.UNDEFINED
 1: onnx.TensorProto.FLOAT
 2: onnx.TensorProto.UINT8
 3: onnx.TensorProto.INT8
 4: onnx.TensorProto.UINT16
 5: onnx.TensorProto.INT16
 6: onnx.TensorProto.INT32
 7: onnx.TensorProto.INT64
 8: onnx.TensorProto.STRING
 9: onnx.TensorProto.BOOL
10: onnx.TensorProto.FLOAT16
11: onnx.TensorProto.DOUBLE
12: onnx.TensorProto.UINT32
13: onnx.TensorProto.UINT64
14: onnx.TensorProto.COMPLEX64
15: onnx.TensorProto.COMPLEX128
16: onnx.TensorProto.BFLOAT16
17: onnx.TensorProto.FLOAT8E4M3FN
18: onnx.TensorProto.FLOAT8E4M3FNUZ
19: onnx.TensorProto.FLOAT8E5M2
20: onnx.TensorProto.FLOAT8E5M2FNUZ
21: onnx.TensorProto.UINT4
22: onnx.TensorProto.INT4
```

ONNX 是强类型语言，其定义不支持隐式转换。即使其他语言支持，也不可能添加两个不同类型的张量或矩阵。这就是必须在图中插入显式转换的原因。

#### 稀疏[张量](broken-reference)\#

稀疏张量可用于表示具有许多空系数的数组。 ONNX 支持二维稀疏张量。[SparseTensorProto](https://onnx.ai/onnx/api/classes.html#l-onnx-sparsetensor-proto)类定义了`dims`、`indices`(int64) 和`values` 等属性。

#### 其他[类型](broken-reference)\#

除了张量和稀疏张量外，ONNX 还通过[SequenceProto](https://onnx.ai/onnx/api/classes.html#l-onnx-sequence-proto) 和[MapProto](https://onnx.ai/onnx/api/classes.html#l-onnx-map-proto) 类型支持张量序列、张量映射、张量映射序列。这些类型很少使用。

### 什么是操作集版本[？](broken-reference)

opset 映射到_onnx_软件包的版本。 每次次版本增加，它都会递增。 每个版本都会带来更新或新的运算符。

```
import onnx
print(onnx.__version__, " opset=", onnx.defs.onnx_opset_version())
```

每个 ONNX 图形还附有一个 opset。这是一个全局信息。操作符_Add_在第 6、7、13 和 14 版中进行了更新。如果图 opset 为 15，则表示操作符_Add_遵循第 14 版规范。如果图 opset 为 12，则操作符_Add_遵循规范版本 7。图中的操作符遵循低于（或等于）全局图操作集的最新定义。

图形可能包含多个域的运算符，例如`ai.onnx`和`ai.onnx.ml`。在这种情况下，图形必须为每个域定义一个全局操作集。该规则适用于同一域内的每个运算符。

### 子图、测试和[循环](broken-reference)\#

ONNX 实现了测试和循环。它们都将另一个 ONNX 图形作为属性。这些结构通常既慢又复杂，最好尽可能避免使用。

#### 如果[#](broken-reference)

操作符[If](https://onnx.ai/onnx/operators/onnx\_\_If.html#l-onnx-doc-if)根据条件评估执行两个图形中的一个。

```
If(condition) then
    execute this ONNX graph (`then_branch`)
else
    execute this ONNX graph (`else_branch`)
```

这两个图形可以使用图形中已经计算出的任何结果，并且必须产生数量完全相同的输出。 这些输出将是运算符`If` 的输出。



#### [扫描](broken-reference)\#

操作符[Scan](https://onnx.ai/onnx/operators/onnx\_\_Scan.html#l-onnx-doc-scan)实现了一个具有固定迭代次数的循环，它对输入的行（或任何其他维度）进行循环，并将输出沿同一轴串联起来。让我们来看一个实现成对距离的示例： .



这个循环的效率很高，即使它仍然比自定义实现的成对距离要慢。它假定输入和输出都是张量，并自动将每次迭代的输出串联成单一张量。前面的示例只有一个，但也可以有多个。

#### [环路](broken-reference)\#

操作符[循环](https://onnx.ai/onnx/operators/onnx\_\_Loop.html#l-onnx-doc-loop)实现了 for 循环和 while 循环。它可以执行固定数量的迭代和/或在不再满足条件时结束。 输出有两种不同的处理方式。第一种类似于循环[扫描](https://onnx.ai/onnx/operators/onnx\_\_Scan.html#l-onnx-doc-scan)，输出被连接成张量（沿第一维度）。第二种机制是将张量连接成张量序列。

### [可扩展性](broken-reference)\#

ONNX 定义了一系列运算符作为标准：ONNX[操作符（ONNX Operators](https://onnx.ai/onnx/operators/index.html#l-onnx-operators)）。 不过，您也可以在此领域或新领域中定义自己的操作符。_onnxruntime_定义了自定义操作符，以改进推理。 每个节点都有类型、名称、命名的输入和输出以及属性。只要在这些约束条件下描述了节点，就可以将节点添加到任何 ONNX 图形中。

成对距离可以用运算符 Scan 来实现，但事实证明，名为 CDist 的专用运算符速度明显更快，足以让我们为它设计一个专用运行时。

### [功能](broken-reference)\#

函数是扩展 ONNX 规范的一种方式。有些模型需要相同的运算符组合。通过创建一个使用现有 ONNX 运算符定义的函数，可以避免这种情况。函数一旦定义，其行为就与其他操作符一样。它有输入、输出和属性。

使用函数有两个好处。第一个是代码更短，更容易阅读。第二个好处是，任何 onnxruntime 都可以利用这些信息更快地运行预测。运行时可以为函数提供特定的实现方式，而不依赖于现有运算符的实现方式。

### 形状（和类型）[推理](broken-reference)\#

执行 ONNX 图形并不需要知道结果的形状，但可以利用这些信息加快执行速度。如果您有以下图形：

```
Add(x, y) -> z
Abs(z) -> w
```

如果_x_和_y_的形状相同，那么_z_和_w_的形状也相同。了解了这一点，就可以重复使用为_z_ 分配的缓冲区，就地计算绝对值_w_。形状推理有助于运行时管理内存，从而提高效率。

在大多数情况下，ONNX 软件包可以根据每个标准算子的输入形状计算输出形状。对于官方列表之外的任何自定义运算符，它显然无法做到这一点。

### [工具](broken-reference)\#

[netron](https://netron.app/)在帮助可视化 ONNX 图形方面非常有用。 这是唯一一个无需编程的工具。第一张截图就是用这个工具制作的。



[onnx2py.py](https://github.com/microsoft/onnxconverter-common/blob/master/onnxconverter\_common/onnx2py.py)根据 ONNX 图形创建一个 python 文件。该脚本可以创建相同的图形。用户可对其进行修改，以改变图形。

[zetane](https://github.com/zetane/viewer)可以加载 onnx 模型，并在执行模型时显示中间结果。
