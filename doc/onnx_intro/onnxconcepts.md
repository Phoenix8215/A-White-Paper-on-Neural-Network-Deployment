# 😆 关于ONNX的一些概念

ONNX 可以看作是一门为数学函数打造的编程语言。它定义了所有关于机器学习推理时所需要的必要操作。线性回归可以用以下方式表示：

```python
def onnx_linear_regressor(X):
    "ONNX code for a linear regression"
    return onnx.Add(onnx.MatMul(X, coefficients), bias)
```

这个例子与在 Python 中编写代码非常相似。因此，使用 ONNX 实现的机器学习模型通常被誉为 ONNX 图(ONNX Graph)。

<figure><img src="../../.gitbook/assets/image (2) (1).png" alt=""><figcaption></figcaption></figure>

ONNX 旨在提供一种通用语言，任何机器学习框架都可以用它来描述自己的模型。使得生产中部署机器学习模型变得更容易。ONNX解释器（或**runtime**）可以在部署环境中专门针对某一任务进行部署和优化。有了 ONNX，我们就可以建立一个独特的流程，将模型部署到生产环境中，并且独立于各种机器学习框架。ONNX实现了一个 python runtime，可用于评估 ONNX 模型和 ONNX 操作。它旨在阐明 ONNX 的语义，帮助理解和调试 ONNX 工具和转换器。它不用于实际生产，性能也不是最终目标。

### <mark style="color:red;">Input, Output, Node, Initializer, Attributes</mark>

构建 ONNX Graph意味着使用 ONNX 语言或更准确地说使用ONNX算子实现一个函数。 一个线性回归模型可以这样编写。 **下面几行并不遵循 python 语法，只是一种用来说明模型的伪代码。**

```python
Input: float[M,K] x, float[K,N] a, float[N] c
Output: float[M, N] y

r = onnx.MatMul(x, a)
y = onnx.Add(r, c)
```

这段代码实现了一个函数`f(x, a, c)-> y = x @ a + c，` _x_、_a_、_c_是**inputs**，_y_是**outputs**。_r_ 是中间结果。_MatMul_ 和 _Add_ 是**nodes**。它们也有**inputs**和**outputs**。**node**是 ONNX 算子中的某一个类型。

**graph**还可以有**initializer**。当输入（如线性回归系数）永不改变时，最好的方法是将其转化为一个常量存储在图中。

```
Input: float[M,K] x
Initializer: float[K,N] a, float[N] c
Output: float[M, N] xac

xa = onnx.MatMul(x, a)
xac = onnx.Add(xa, c)
```

如下图所示：右侧描述了运算符_Add_，其中第二个输入被定义为**initializer**。

<figure><img src="../../.gitbook/assets/image (3).png" alt=""><figcaption></figcaption></figure>

**attribute** 是运算符的固定参数。比如：运算符[Gemm](https://onnx.ai/onnx/operators/onnx\_\_Gemm.html#l-onnx-doc-gemm)有四个属性：_alpha_、_beta_、_transA_、_transB_。除非使用ONNX runtime的API进行修改，否则一旦加载了 ONNX graph，这些值就不能更改，并在模型预测阶段保持不变。

### <mark style="color:red;">使用protobuf进行序列化</mark>

将机器学习模型部署到生产环境中通常需要将训练模型的整个生态系统复制下来，大多数情况下需要使用_docker_。 一旦模型转换为 ONNX，生产环境只需要runtime来执行计算图。该runtime可以用任何适合生产应用的语言开发，如 C、java、python、javascript、C#、Webassembly、ARM......

但要做到这一点，就需要保存 ONNX 计算图。ONNX 使用 protobuf 将计算图序列化为单个块。其目的是尽可能优化模型大小。

### <mark style="color:red;">元数据</mark>

机器学习模型一直在不断更新。所以跟踪**模型的版本**、**模型的作者**以及**模型的训练方式**就显得非常重要。ONNX 提供了在模型中存储额外数据的方式。

*   **doc\_string**: Human-readable documentation for this model.

    Markdown is allowed.
*   **domain**: A reverse-DNS name to indicate the model namespace or domain,

    for example, ‘org.onnx’
*   **metadata\_props**: Named metadata as dictionary `map<string,string>`,

    `(values, keys)` should be distinct.
*   **model\_author**: A comma-separated list of names,

    The personal name of the author(s) of the model, and/or their organizations.
*   **model\_license**: The well-known name or URL of the license

    under which the model is made available.
* **model\_version**: The version of the model itself, encoded in an integer.
* **producer\_name**: The name of the tool used to generate the model.
* **producer\_version**: The version of the generating tool.
*   **training\_info**: An optional extension that contains

    information for training (see [TrainingInfoProto](https://onnx.ai/onnx/api/classes.html#l-traininginfoproto))

### <mark style="color:red;">ONNX算子和域</mark>

主要列表在此说明：[ONNX算子列表](https://onnx.ai/onnx/operators/index.html#l-onnx-operators)。它融合了标准矩阵运算符（Add、Sub、MatMul、Transpose、Greater、IsNaN、Shape、Reshape...）、归约（ReduceSum、ReduceMin...）、图像变换（Conv、MaxPool...）、深度神经网络层（RNN、DropOut...）、激活函数（Relu、Softmax...）。 **ONNX 并不实现所有的机器学习相关的算子，否则列表将是无限的。**

运算符的主列表由一个域**ai.onnx** 标识。 一个**域**可定义为一组算子的集合。 主列表中缺少在标准机器学习中非常流行的基于树的模型，这些模型属于另一个域[**ai.onnx.ml**](http://ai.onnx.ml/)，它包括基于树的模型（TreeEnsemble Regressor, ...）、预处理（OneHotEncoder, LabelEncoder, ...）、SVM 模型（SVMRegressor, ...）和输入器（Imputer）。

ONNX 只定义了这两个域。但 onnx 库支持任何自定义域和运算符。

### <mark style="color:red;">支持的类型</mark>

ONNX 专门为张量的数值计算做了相关优化。**张量**是一个多维数组。其定义如下

* a type: the element type, the same for all elements in the tensor
* a shape: an array with all dimensions, this array can be empty, a dimension can be null
* a contiguous array: it represents all the values

该定义不包括stride，也不能根据已有张量定义新张量。ONNX 张量是一个密集型数据。

#### <mark style="color:red;">元素类型</mark>

ONNX 最初是为了部署深度学习模型而开发的。 因此，其规格是针对浮点数（32 位）设计的。 当前版本支持所有常见类型。字典[TENSOR\_TYPE\_MAP](https://onnx.ai/onnx/api/mapping.html#l-onnx-types-mapping)提供了_ONNX_和[`numpy`](https://numpy.org/doc/stable/reference/index.html#module-numpy) 之间的对应关系。

```python
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

```sh
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

**ONNX 是强类型语言，不支持隐式转换，所以不能将两个不同类型的张量或矩阵进行相加。**

#### <mark style="color:red;">稀疏张量</mark>

稀疏张量可用于表示具有许多空数值的数组。 ONNX 支持二维稀疏张量。[SparseTensorProto](https://onnx.ai/onnx/api/classes.html#l-onnx-sparsetensor-proto)类定义了`dims`、`indices`(int64) 和`values` 等属性。

#### <mark style="color:red;">其他类型</mark>

除了张量和稀疏张量外，ONNX 还通过定义[SequenceProto](https://onnx.ai/onnx/api/classes.html#l-onnx-sequence-proto) 和[MapProto](https://onnx.ai/onnx/api/classes.html#l-onnx-map-proto) 类型支持张量序列、张量映射、张量映射序列。这些类型很少使用。

### <mark style="color:red;">什么是opset版本？</mark>

opset 映射到_onnx_软件包的版本。 每次版本增加，它都会递增。 每个版本都会带来更新或新的运算符。

```python
import onnx
print(onnx.__version__, " opset=", onnx.defs.onnx_opset_version())
```

```bash
1.16.0  opset= 21
```

每个 ONNX 计算图还附有一个 opset。这是一个全局信息。操作符_Add_在第 6、7、13 和 14 版中进行了更新。如果计算图 opset 为 15，则表示操作符_Add_遵循第 14 版规范。如果计算图 opset 为 12，则算子_Add_遵循规范版本 7。

ONNX计算图可能包含多个域的算子，例如`ai.onnx`和`ai.onnx.ml`。在这种情况下，计算图必须为每个域定义一个全局opset。该规则适用于同一个域中的所有算子。

### <mark style="color:red;">子图、测试和循环</mark>

ONNX 实现了测试和循环。它们都将另一个 ONNX 计算图作为属性。这些结构通常既慢又复杂，最好避免使用。

#### <mark style="color:red;">If</mark>

算子[If](https://onnx.ai/onnx/operators/onnx\_\_If.html#l-onnx-doc-if)根据条件评估执行两个图形中的一个。

```python
If(condition) then
    execute this ONNX graph (`then_branch`)
else
    execute this ONNX graph (`else_branch`)
```

这两个图形可以使用图形中已经计算出的任何结果，并且必须产生数量完全相同的输出。 这些输出将是算子`If` 的输出。

<figure><img src="../../.gitbook/assets/image.png" alt=""><figcaption></figcaption></figure>

#### <mark style="color:red;">Scan</mark>

操作符[Scan](https://onnx.ai/onnx/operators/onnx\_\_Scan.html#l-onnx-doc-scan)实现了一个具有固定迭代次数的循环，它对输入的行（或其他维度）进行循环，并将输出沿同一维度串联起来。让我们来看一个示例$$M(i,j)=\|X_i-X_j\|^2$$： .

<figure><img src="../../.gitbook/assets/image (1).png" alt=""><figcaption></figcaption></figure>

这个循环的效率很高，它假定输入和输出都是张量，并自动将每次迭代的输出串联成单一张量。

#### <mark style="color:red;">Loop</mark>

算子Loop实现了 for 循环和 while 循环。它可以执行固定数量的迭代和在不再满足条件时结束。 输出有两种不同的处理方式。第一种类似于循环Scan，输出的是沿着某一维度进行拼接的张量。第二种机制是将张量连接成张量序列。

### <mark style="color:red;">可扩展性</mark>

ONNX 定义了一系列算子作为标准：[ONNX算子（ONNX Operators）](https://onnx.ai/onnx/operators/index.html#l-onnx-operators)。 不过，你也可以在此域或新的域中定义自己的算子。_onnxruntime_自定义了一些算子以改进推理。 每个节点都有**类型、名称、输入和输出以及属性**。只要在这些约束条件下描述了节点，就可以将节点添加到任何 ONNX 计算图中作为算子使用。

### <mark style="color:red;">函数</mark>

函数是扩展 ONNX 规范的一种方式。有些模型需要相同的运算符组合。通过创建一个使用现有 ONNX 算子的函数，可以避免这种情况发生。函数一旦定义，其行为就与其他算子一样，有输入、输出和属性。

使用函数有两个好处。第一个是代码更短，更容易阅读。第二个好处是，任何 onnxruntime 都可以利用这些信息更快地进行模型预测。onnxruntime可以为函数提供特定的实现方式，而不依赖于现有算子的实现方式。

### <mark style="color:red;">形状（和类型）推理</mark>

执行 ONNX 计算图并不需要知道模型输出结果的形状，但可以利用这些信息加快执行速度。有以下计算图：

```
Add(x, y) -> z
Abs(z) -> w
```

如果_x_和_y_的形状相同，那么_z_和_w_的形状也相同。了解了这一点，就可以重复使用为_z_ 分配的缓冲区，就地计算绝对值_w_。形状推理有助于runtime管理内存，从而提高效率。

在大多数情况下，ONNX 软件包可以根据每个标准算子的输入形状计算输出形状。对于官方列表之外的任何自定义运算符，它显然无法做到这一点。

### <mark style="color:red;">一些有用的工具</mark>

[netron](https://netron.app/) 在帮助可视化 ONNX 图形方面非常有用。 这是唯一一个无需编程的工具。第一张截图就是用这个工具制作的。

[onnx2py.py](https://github.com/microsoft/onnxconverter-common/blob/master/onnxconverter\_common/onnx2py.py) 根据 ONNX 图形创建一个 python 文件。该脚本可以创建相同的图形。用户可对其进行修改，以改变图形。

```python
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###########################################################################

"""
Converts onnx model into model.py file for easy editing. Resulting model.py file uses onnx.helper library to
recreate the original onnx model. Constant tensors with more than 10 elements are saved into .npy
files in location model/const#_tensor_name.npy

Example usage:
python -m onnxconverter_common.onnx2py my_model.onnx my_model.py
"""

import sys
import onnx
import collections
import inspect
from collections import OrderedDict
from onnx import helper, numpy_helper, TensorProto, external_data_helper
import numpy as np
import os

from .pytracing import TracingObject

needed_types = set()
const_dir = None
const_counter = None

np_traced = TracingObject("np", np)
helper_traced = TracingObject("helper", helper)
numpy_helper_traced = TracingObject("numpy_helper", numpy_helper)
TensorProtoTraced = TracingObject("TensorProto", TensorProto)
os_traced = TracingObject("os", os)


# <Helpers> These can be inlined into the output script #

def clear_field(proto, field):
    proto.ClearField(field)
    return proto


def order_repeated_field(repeated_proto, key_name, order):
    order = list(order)
    repeated_proto.sort(key=lambda x: order.index(getattr(x, key_name)))


def make_external_tensor(name, data_type, dims, raw_data=None, **kwargs):
    tensor = TensorProto()
    tensor.data_type = data_type
    tensor.name = name
    tensor.dims.extend(dims)
    tensor.raw_data = raw_data if raw_data is not None else b''
    external_data_helper.set_external_data(tensor, **kwargs)
    if raw_data is None:
        tensor.ClearField("raw_data")
    order_repeated_field(tensor.external_data, 'key', kwargs.keys())
    return tensor


def make_node(op_type, inputs, outputs, name=None, doc_string=None, domain=None, **kwargs):
    node = helper.make_node(op_type, inputs, outputs, name, doc_string, domain, **kwargs)
    if doc_string == '':
        node.doc_string = ''
    order_repeated_field(node.attribute, 'name', kwargs.keys())
    return node


def make_graph(*args, doc_string=None, **kwargs):
    graph = helper.make_graph(*args, doc_string=doc_string, **kwargs)
    if doc_string == '':
        graph.doc_string = ''
    return graph

# </Helpers> #


clear_field_traced = TracingObject("clear_field", clear_field)
make_external_tensor_traced = TracingObject("make_external_tensor", make_external_tensor)
make_node_traced = TracingObject("make_node", make_node)
make_graph_traced = TracingObject("make_graph", make_graph)
DATA_DIR_TRACED = None


def convert_tensor_type(i):
    return getattr(TensorProtoTraced, TensorProto.DataType.Name(i))


def convert_field(field):
    global needed_types
    if isinstance(field, (int, str, float, bytes)):
        return field
    elif isinstance(field, onnx.GraphProto):
        converted = convert_graph(field)
    elif isinstance(field, onnx.ModelProto):
        converted = convert_model(field)
    elif isinstance(field, onnx.NodeProto):
        converted = convert_node(field)
    elif isinstance(field, onnx.TensorProto):
        converted = convert_tensor(field)
    elif isinstance(field, onnx.ValueInfoProto):
        converted = convert_value_info(field)
    elif isinstance(field, onnx.OperatorSetIdProto):
        converted = convert_operatorsetid(field)
    elif isinstance(field, collections.abc.Iterable):
        return list(convert_field(x) for x in field)
    else:
        # Missing handler needs to be added
        t = str(type(field))
        needed_types.add(t)
        return field
    # Verify that resulting protobuf is identical to original
    # assert TracingObject.get_py_obj(converted) == field
    return converted


def convert_value_info(val_info):
    name = val_info.name
    is_sequence_type = val_info.type.HasField('sequence_type')
    if is_sequence_type:
        tensor_type = val_info.type.sequence_type.elem_type.tensor_type
    else:
        tensor_type = val_info.type.tensor_type
    elem_type = convert_tensor_type(tensor_type.elem_type)
    kwargs = OrderedDict()

    def convert_shape_dim(d):
        if d.HasField("dim_value"):
            return d.dim_value
        if d.HasField("dim_param"):
            return d.dim_param
        return None

    def convert_shape_denotation(d):
        if d.HasField("denotation"):
            return d.denotation
        return None

    if tensor_type.HasField("shape"):
        kwargs["shape"] = [convert_shape_dim(d) for d in tensor_type.shape.dim]
    else:
        kwargs["shape"] = None
    if any(d.HasField("denotation") for d in tensor_type.shape.dim):
        kwargs["shape_denotation"] = [convert_shape_denotation(d) for d in tensor_type.shape.dim]

    if val_info.HasField("doc_string"):
        kwargs["doc_string"].doc_string

    if is_sequence_type:
        return helper_traced.make_sequence_value_info(name, elem_type, **kwargs)
    else:
        return helper_traced.make_tensor_value_info(name, elem_type, **kwargs)


def convert_operatorsetid(opsetid):
    version = opsetid.version
    if opsetid.HasField("domain"):
        domain = opsetid.domain
        return helper_traced.make_operatorsetid(domain, version)
    else:
        return clear_field_traced(helper_traced.make_operatorsetid('', version), 'domain')


def convert_external_tensor(tensor):
    kwargs = OrderedDict()
    if tensor.HasField("raw_data"):
        kwargs["raw_data"] = tensor.raw_data
    if tensor.external_data:
        for d in tensor.external_data:
            kwargs[d.key] = d.value
    return make_external_tensor_traced(tensor.name, tensor.data_type, tensor.dims, **kwargs)


def convert_tensor(tensor):
    global const_dir, const_counter
    if tensor.data_location == TensorProto.EXTERNAL:
        return convert_external_tensor(tensor)
    np_data = numpy_helper.to_array(tensor)
    if np.product(np_data.shape) <= 10:
        return numpy_helper_traced.from_array(np_data, name=tensor.name)
    dtype = np_data.dtype
    if dtype == object:
        np_data = np_data.astype(str)
    os.makedirs(const_dir, exist_ok=True)
    name = "const" + str(const_counter)
    if tensor.name and len(tensor.name) < 100:
        # Avoid path length limit on windows
        name = name + "_" + tensor.name
    for c in '~"#%&*:<>?/\\{|}':
        name = name.replace(c, '_')
    const_path = "%s/%s.npy" % (const_dir, name)
    np.save(const_path, np_data)
    data_path = os_traced.path.join(DATA_DIR_TRACED, name + '.npy')
    const_counter += 1
    np_dtype = str(dtype)
    np_shape = list(np_data.shape)
    np_array = np_traced.load(data_path).astype(np_dtype).reshape(np_shape)
    return numpy_helper_traced.from_array(np_array, name=tensor.name)


def convert_node(node):
    fields = OrderedDict((f[0].name, f[1]) for f in node.ListFields())
    attributes = fields.pop("attribute", [])
    attrs = OrderedDict((a.name, convert_field(helper.get_attribute_value(a))) for a in attributes)
    fields = OrderedDict((f, convert_field(v)) for f, v in fields.items())
    op_type = fields.pop("op_type")
    if op_type == "Cast" and "to" in attrs:
        attrs["to"] = convert_tensor_type(attrs["to"])
    inputs = fields.pop("input", [])
    outputs = fields.pop("output", [])
    return make_node_traced(op_type, inputs=inputs, outputs=outputs, **fields, **attrs)


def convert_graph(graph):
    fields = OrderedDict((f[0].name, convert_field(f[1])) for f in graph.ListFields())
    nodes = fields.pop("node", [])
    name = fields.pop("name")
    inputs = fields.pop("input", [])
    outputs = fields.pop("output", [])
    return make_graph_traced(name=name, inputs=inputs, outputs=outputs, **fields, nodes=nodes)


def convert_model(model):
    fields = OrderedDict((f[0].name, convert_field(f[1])) for f in model.ListFields())
    graph = fields.pop("graph")
    opset_imports = fields.pop("opset_import", [])
    return helper_traced.make_model(opset_imports=opset_imports, **fields, graph=graph)


def clear_directory(path):
    for f in os.listdir(path):
        if f.endswith(".npy"):
            os.remove(os.path.join(path, f))
    try:
        # Delete if empty
        os.rmdir(path)
    except OSError:
        pass


class MissingHandlerException(Exception):
    pass


FILE_HEADER = '''"""
Run this script to recreate the original onnx model.
Example usage:
python %s.py out_model_path.onnx
"""'''


def convert(model, out_path):
    global needed_types, const_dir, const_counter, DATA_DIR_TRACED
    needed_types = set()
    if out_path.endswith(".py"):
        out_path = out_path[:-3]
    if os.path.exists(out_path):
        clear_directory(out_path)
    const_dir = out_path
    const_dir_name = os.path.basename(out_path)
    const_counter = 0
    TracingObject.reset_cnt(clear_field_traced)
    TracingObject.reset_cnt(make_external_tensor_traced)
    DATA_DIR_TRACED = TracingObject("DATA_DIR", const_dir)

    model_trace = convert_field(model)

    code = FILE_HEADER % os.path.basename(out_path) + "\n"
    code += "\nfrom onnx import helper, numpy_helper, TensorProto\n"
    if TracingObject.get_cnt(make_external_tensor_traced):
        code += ", external_data_helper"
    code += "\n"
    code += "import onnx\n"
    code += "import numpy as np\n"
    code += "import sys\n"
    if os.path.exists(const_dir):
        code += "import os\n"
        code += "\nDATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), %r)\n" % const_dir_name
    if TracingObject.get_cnt(clear_field_traced):
        code += "\n" + inspect.getsource(clear_field)
    code += "\n" + inspect.getsource(order_repeated_field)
    if TracingObject.get_cnt(make_external_tensor_traced):
        code += "\n" + inspect.getsource(make_external_tensor)
    code += "\n" + inspect.getsource(make_node)
    code += "\n" + inspect.getsource(make_graph)
    code += "\n" + "model = " + repr(model_trace) + "\n"
    code += "\nif __name__ == '__main__' and len(sys.argv) == 2:\n"
    code += "    _, out_path = sys.argv\n"
    if TracingObject.get_cnt(make_external_tensor_traced):
        code += "    with open(out_path, 'wb') as f:\n"
        code += "        f.write(model.SerializeToString())\n"
    else:
        code += "    onnx.save(model, out_path)\n"
    with open(out_path + ".py", "wt", encoding='utf8') as file:
        file.write(code)
    if needed_types:
        raise MissingHandlerException("Missing handler for types: %s" % list(needed_types))
    return model_trace


def main():
    _, in_path, out_path = sys.argv
    if not out_path.endswith(".py"):
        out_path = out_path + ".py"

    model = onnx.load(in_path, load_external_data=False)
    try:
        model_trace = convert(model, out_path)
        if TracingObject.get_py_obj(model_trace).SerializeToString() == model.SerializeToString():
            print("\nConversion successful. Converted model is identical.\n")
        else:
            print("\nWARNING: Conversion succeeded but converted model is not identical. "
                  "Difference might be trivial.\n")
    except MissingHandlerException as e:
        print("ERROR:", e)

    print("Model saved to", out_path)
    print("Run 'python %s output.onnx' to generate ONNX file" % out_path)
    print("Import the model with 'from %s import model'" % os.path.basename(out_path[:-3]))


if __name__ == '__main__':
    main()
```

[zetane](https://github.com/zetane/viewer) 可以加载 onnx 模型，并在执行模型时显示中间结果。
