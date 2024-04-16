# 😍 用python操作ONNX

接下来的章节将重点介绍如何使用_onnx_提供的[Python API](https://onnx.ai/onnx/api/index.html#l-python-onnx-api)构建 ONNX 计算图。

### 一个简单的例子：线性回归

线性回归是机器学习中最简单的模型，其表达式如下.我们可以将其视为三个变量的函数 分解为`y = Add(MatMul(X, A), B`)。这就是我们需要用 ONNX 运算符表示的内容。首先是用[ONNX](https://onnx.ai/onnx/operators/index.html#l-onnx-operators) 运算符实现函数。 ONNX 是强类型的。必须为函数的输入和输出定义形状和类型。也就是说，在[make 函数](https://onnx.ai/onnx/api/helper.html#l-onnx-make-function)中，我们需要四个函数来构建计算图：

* `make_tensor_value_info`：声明变量（输入或输出）的形状和类型
* `make_node`：创建一个由操作符类型(算子名称)、输入和输出定义的节点
* `make_graph`：利用前两个函数创建的对象创建 ONNX 计算图
* `make_model`：将计算图和额外的元数据进行合并

在创建过程中，我们需要为图中每个节点的输入和输出命名。计算图的输入和输出由 onnx 对象定义，字符串用于指代中间结果。

```python
# imports

from onnx import TensorProto
from onnx.helper import (
    make_model, make_node, make_graph,
    make_tensor_value_info)
from onnx.checker import check_model

# inputs

# 'X' is the name, TensorProto.FLOAT the type, [None, None] the shape
X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
B = make_tensor_value_info('B', TensorProto.FLOAT, [None, None])

# outputs, the shape is left undefined

Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])

# nodes

# It creates a node defined by the operator type MatMul,
# 'X', 'A' are the inputs of the node, 'XA' the output.
node1 = make_node('MatMul', ['X', 'A'], ['XA'])
node2 = make_node('Add', ['XA', 'B'], ['Y'])

# from nodes to graph
# the graph is built from the list of nodes, the list of inputs,
# the list of outputs and a name.

graph = make_graph([node1, node2],  # nodes
                    'lr',  # a name
                    [X, A, B],  # inputs
                    [Y])  # outputs

# onnx graph
# there is no metadata in this case.

onnx_model = make_model(graph)

# Let's check the model is consistent,
# this function is described in section
# Checker and Shape Inference.
check_model(onnx_model)

# the work is done, let's display it...
print(onnx_model)
```

```sh
ir_version: 10
graph {
  node {
    input: "X"
    input: "A"
    output: "XA"
    op_type: "MatMul"
  }
  node {
    input: "XA"
    input: "B"
    output: "Y"
    op_type: "Add"
  }
  name: "lr"
  input {
    name: "X"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
          }
        }
      }
    }
  }
  input {
    name: "A"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
          }
        }
      }
    }
  }
  input {
    name: "B"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
          }
        }
      }
    }
  }
  output {
    name: "Y"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
        }
      }
    }
  }
}
opset_import {
  version: 21
}
```

<figure><img src="../../.gitbook/assets/dot_linreg.png" alt=""><figcaption></figcaption></figure>

shape定义为`[None, None]`表示该对象是一个两维张量，没有任何形状信息。 通过查看图中每个对象的字段，也可以检查 ONNX 计算图。

```python
from onnx import TensorProto
from onnx.helper import (
    make_model, make_node, make_graph,
    make_tensor_value_info)
from onnx.checker import check_model

def shape2tuple(shape):
    return tuple(getattr(d, 'dim_value', 0) for d in shape.dim)

X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
B = make_tensor_value_info('B', TensorProto.FLOAT, [None, None])
Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])
node1 = make_node('MatMul', ['X', 'A'], ['XA'])
node2 = make_node('Add', ['XA', 'B'], ['Y'])
graph = make_graph([node1, node2], 'lr', [X, A, B], [Y])
onnx_model = make_model(graph)
check_model(onnx_model)

# the list of inputs
print('** inputs **')
print(onnx_model.graph.input)

# in a more nicely format
print('** inputs **')
for obj in onnx_model.graph.input:
    print("name=%r dtype=%r shape=%r" % (
        obj.name, obj.type.tensor_type.elem_type,
        shape2tuple(obj.type.tensor_type.shape)))

# the list of outputs
print('** outputs **')
print(onnx_model.graph.output)

# in a more nicely format
print('** outputs **')
for obj in onnx_model.graph.output:
    print("name=%r dtype=%r shape=%r" % (
        obj.name, obj.type.tensor_type.elem_type,
        shape2tuple(obj.type.tensor_type.shape)))

# the list of nodes
print('** nodes **')
print(onnx_model.graph.node)

# in a more nicely format
print('** nodes **')
for node in onnx_model.graph.node:
    print("name=%r type=%r input=%r output=%r" % (
        node.name, node.op_type, node.input, node.output))
```

```sh
** inputs **
[name: "X"
type {
  tensor_type {
    elem_type: 1
    shape {
      dim {
      }
      dim {
      }
    }
  }
}
, name: "A"
type {
  tensor_type {
    elem_type: 1
    shape {
      dim {
      }
      dim {
      }
    }
  }
}
, name: "B"
type {
  tensor_type {
    elem_type: 1
    shape {
      dim {
      }
      dim {
      }
    }
  }
}
]
** inputs **
name='X' dtype=1 shape=(0, 0)
name='A' dtype=1 shape=(0, 0)
name='B' dtype=1 shape=(0, 0)
** outputs **
[name: "Y"
type {
  tensor_type {
    elem_type: 1
    shape {
      dim {
      }
    }
  }
}
]
** outputs **
name='Y' dtype=1 shape=(0,)
** nodes **
[input: "X"
input: "A"
output: "XA"
op_type: "MatMul"
, input: "XA"
input: "B"
output: "Y"
op_type: "Add"
]
** nodes **
name='' type='MatMul' input=['X', 'A'] output=['XA']
name='' type='Add' input=['XA', 'B'] output=['Y']
```

张量类型是浮点数（= 1）。函数[`onnx.helper.tensor_dtype_to_np_dtype()`](https://onnx.ai/onnx/api/helper.html#onnx.helper.tensor\_dtype\_to\_np\_dtype)会给出与 numpy 对应的数据类型。

```python
from onnx import TensorProto
from onnx.helper import tensor_dtype_to_np_dtype, tensor_dtype_to_string

np_dtype = tensor_dtype_to_np_dtype(TensorProto.FLOAT)
print(f"The converted numpy dtype for {tensor_dtype_to_string(TensorProto.FLOAT)} is {np_dtype}.")
```

```
The converted numpy dtype for TensorProto.FLOAT is float32.
```

{% hint style="info" %}
`%r`同样用于格式化字符串，但它表示将一个值转换为它的“原始”字符串表示形式。这意味着，它会保留字符串中的所有特殊字符，包括空格和换行符等，而不会尝试对它们进行任何转换或解释。这对于调试或者需要保留原始格式的字符串非常有用。

例如：

```python
message = "Hello,\nWorld!"
formatted_message = "The message is: %r" % message
print(formatted_message)
```

输出将会是：

```
The message is: Hello,\nWorld!
```
{% endhint %}

### 序列化

ONNX 建立在 protobuf 的基础之上。它为描述机器学习模型添加了必要的定义，大多数情况下，ONNX 是用来序列化或反序列化模型的。第二部分将介绍数据的序列化和反序列化，如张量、稀疏张量......

#### 模型序列化

ONNX 基于 `protobuf`。它最大限度地减少了在磁盘上保存图形所需的空间。onnx 中的每个对象（参见[Protos](https://onnx.ai/onnx/api/classes.html#l-onnx-classes)）都可以通过`SerializeToString` 方法序列化。

```python
from onnx import TensorProto
from onnx.helper import (
    make_model, make_node, make_graph,
    make_tensor_value_info)
from onnx.checker import check_model

def shape2tuple(shape):
    return tuple(getattr(d, 'dim_value', 0) for d in shape.dim)

X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
B = make_tensor_value_info('B', TensorProto.FLOAT, [None, None])
Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])
node1 = make_node('MatMul', ['X', 'A'], ['XA'])
node2 = make_node('Add', ['XA', 'B'], ['Y'])
graph = make_graph([node1, node2], 'lr', [X, A, B], [Y])
onnx_model = make_model(graph)
check_model(onnx_model)

# The serialization
with open("linear_regression.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

# display
print(onnx_model)
```

```sh
ir_version: 10
graph {
  node {
    input: "X"
    input: "A"
    output: "XA"
    op_type: "MatMul"
  }
  node {
    input: "XA"
    input: "B"
    output: "Y"
    op_type: "Add"
  }
  name: "lr"
  input {
    name: "X"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
          }
        }
      }
    }
  }
  input {
    name: "A"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
          }
        }
      }
    }
  }
  input {
    name: "B"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
          }
        }
      }
    }
  }
  output {
    name: "Y"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
        }
      }
    }
  }
}
opset_import {
  version: 21
}
```

ONNX计算图可通过函数`load`恢复：

```python
from onnx import load

with open("linear_regression.onnx", "rb") as f:
    onnx_model = load(f)

# display
print(onnx_model)
```

```sh
ir_version: 10
graph {
  node {
    input: "X"
    input: "A"
    output: "XA"
    op_type: "MatMul"
  }
  node {
    input: "XA"
    input: "B"
    output: "Y"
    op_type: "Add"
  }
  name: "lr"
  input {
    name: "X"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
          }
        }
      }
    }
  }
  input {
    name: "A"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
          }
        }
      }
    }
  }
  input {
    name: "B"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
          }
        }
      }
    }
  }
  output {
    name: "Y"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
        }
      }
    }
  }
}
opset_import {
  version: 21
}
```

任何模型都可以用这种方式序列化，除非它们的大小超过 2 Gb。接下来的章节将介绍如何克服模型大小这一限制。

#### 张量序列化

张量的序列化过程通常如下：

```python
import numpy
from onnx.numpy_helper import from_array

numpy_tensor = numpy.array([0, 1, 4, 5, 3], dtype=numpy.float32)
print(type(numpy_tensor))

onnx_tensor = from_array(numpy_tensor)
print(type(onnx_tensor))

serialized_tensor = onnx_tensor.SerializeToString()
print(type(serialized_tensor))

with open("saved_tensor.pb", "wb") as f:
    f.write(serialized_tensor)
```

```sh
<class 'numpy.ndarray'>
<class 'onnx.onnx_ml_pb2.TensorProto'>
<class 'bytes'>
```

还有反序列化：

```python
from onnx import TensorProto
from onnx.numpy_helper import to_array

with open("saved_tensor.pb", "rb") as f:
    serialized_tensor = f.read()
print(type(serialized_tensor))

onnx_tensor = TensorProto()
onnx_tensor.ParseFromString(serialized_tensor)
print(type(onnx_tensor))

numpy_tensor = to_array(onnx_tensor)
print(numpy_tensor)
```

```sh
<class 'bytes'>
<class 'onnx.onnx_ml_pb2.TensorProto'>
[0. 1. 4. 5. 3.]
```

数据类型不限于[TensorProto](https://onnx.ai/onnx/api/classes.html#l-tensorproto)：

```python
import onnx
import pprint
pprint.pprint([p for p in dir(onnx)
               if p.endswith('Proto') and p[0] != '_'])
```

```sh
['AttributeProto',
 'FunctionProto',
 'GraphProto',
 'MapProto',
 'ModelProto',
 'NodeProto',
 'OperatorProto',
 'OperatorSetIdProto',
 'OperatorSetProto',
 'OptionalProto',
 'SequenceProto',
 'SparseTensorProto',
 'StringStringEntryProto',
 'TensorProto',
 'TensorShapeProto',
 'TrainingInfoProto',
 'TypeProto',
 'ValueInfoProto']
```

使用函数_`load_tensor_from_string`_可以简化这段代码。

```python
from onnx import load_tensor_from_string

with open("saved_tensor.pb", "rb") as f:
    serialized = f.read()
proto = load_tensor_from_string(serialized)
print(type(proto))
```

```sh
<class 'onnx.onnx_ml_pb2.TensorProto'>
```

### Initializer，默认值

之前的模型假定线性回归的系数也是模型的输入，这不是很方便。为了遵循 onnx 语义，它们应该作为常量或**initializer**成为模型本身的一部分。这个示例修改了上一个示例，将输入`A`和`B`变为initializer。以下两个函数，可以将 numpy 转换为 onnx，也可以反过来转换（参见[数组](https://onnx.ai/onnx/api/numpy\_helper.html#l-numpy-helper-onnx-array)）。

* `onnx.numpy_helper.to_array`：从 onnx 转换到 numpy
* `onnx.numpy_helper.from_array`: 从 numpy 转换到 onnx

<pre class="language-python" data-full-width="false"><code class="lang-python">import numpy
from onnx import numpy_helper, TensorProto
from onnx.helper import (
    make_model, make_node, make_graph,
    make_tensor_value_info)
from onnx.checker import check_model

# initializers
value = numpy.array([0.5, -0.6], dtype=numpy.float32)
<strong>A = numpy_helper.from_array(value, name='A')
</strong>
value = numpy.array([0.4], dtype=numpy.float32)
C = numpy_helper.from_array(value, name='C')

# the part which does not change
X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])
node1 = make_node('MatMul', ['X', 'A'], ['AX'])
node2 = make_node('Add', ['AX', 'C'], ['Y'])
graph = make_graph([node1, node2], 'lr', [X], [Y], [A, C])
onnx_model = make_model(graph)
check_model(onnx_model)

print(onnx_model)
</code></pre>

```sh
ir_version: 10
graph {
  node {
    input: "X"
    input: "A"
    output: "AX"
    op_type: "MatMul"
  }
  node {
    input: "AX"
    input: "C"
    output: "Y"
    op_type: "Add"
  }
  name: "lr"
  initializer {
    dims: 2
    data_type: 1
    name: "A"
    raw_data: "\000\000\000?\232\231\031\277"
  }
  initializer {
    dims: 1
    data_type: 1
    name: "C"
    raw_data: "\315\314\314>"
  }
  input {
    name: "X"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
          }
        }
      }
    }
  }
  output {
    name: "Y"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
        }
      }
    }
  }
}
opset_import {
  version: 21
}
```

<figure><img src="../../.gitbook/assets/dot_linreg2.png" alt=""><figcaption></figcaption></figure>

同样，也可以通过 onnx API来查看`initializers`。

```python
import numpy
from onnx import numpy_helper, TensorProto
from onnx.helper import (
    make_model, make_node, make_graph,
    make_tensor_value_info)
from onnx.checker import check_model

# initializers
value = numpy.array([0.5, -0.6], dtype=numpy.float32)
A = numpy_helper.from_array(value, name='A')

value = numpy.array([0.4], dtype=numpy.float32)
C = numpy_helper.from_array(value, name='C')

# the part which does not change
X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])
node1 = make_node('MatMul', ['X', 'A'], ['AX'])
node2 = make_node('Add', ['AX', 'C'], ['Y'])
graph = make_graph([node1, node2], 'lr', [X], [Y], [A, C])
onnx_model = make_model(graph)
check_model(onnx_model)

print('** initializer **')
for init in onnx_model.graph.initializer:
    print(init)
```

```sh
** initializer **
dims: 2
data_type: 1
name: "A"
raw_data: "\000\000\000?\232\231\031\277"

dims: 1
data_type: 1
name: "C"
raw_data: "\315\314\314>"
```

### Attributes

有些算子需要属性，如[`Transpose`](https://onnx.ai/onnx/operators/onnx\_\_Transpose.html#l-onnx-doc-transpose)算子。 让我们为表达式`y = Add(MatMul(X, Transpose(A))+ B`)创建一个ONNX计算图。`Transpose` 需要一个定义了坐标轴排列顺序的属性：`perm=[1, 0]`。它在函数`make_node` 中作为命名属性添加进去。

```python
from onnx import TensorProto
from onnx.helper import (
    make_model, make_node, make_graph,
    make_tensor_value_info)
from onnx.checker import check_model

# unchanged
X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
B = make_tensor_value_info('B', TensorProto.FLOAT, [None, None])
Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])

# added
node_transpose = make_node('Transpose', ['A'], ['tA'], perm=[1, 0])

# unchanged except A is replaced by tA
node1 = make_node('MatMul', ['X', 'tA'], ['XA'])
node2 = make_node('Add', ['XA', 'B'], ['Y'])

# node_transpose is added to the list
graph = make_graph([node_transpose, node1, node2],
                   'lr', [X, A, B], [Y])
onnx_model = make_model(graph)
check_model(onnx_model)

# the work is done, let's display it...
print(onnx_model)
```

```sh
ir_version: 10
graph {
  node {
    input: "A"
    output: "tA"
    op_type: "Transpose"
    attribute {
      name: "perm"
      ints: 1
      ints: 0
      type: INTS
    }
  }
  node {
    input: "X"
    input: "tA"
    output: "XA"
    op_type: "MatMul"
  }
  node {
    input: "XA"
    input: "B"
    output: "Y"
    op_type: "Add"
  }
  name: "lr"
  input {
    name: "X"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
          }
        }
      }
    }
  }
  input {
    name: "A"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
          }
        }
      }
    }
  }
  input {
    name: "B"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
          }
        }
      }
    }
  }
  output {
    name: "Y"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
        }
      }
    }
  }
}
opset_import {
  version: 21
}
```

<figure><img src="../../.gitbook/assets/dot_att-1.png" alt=""><figcaption></figcaption></figure>

_`make`_函数的整个列表如下，具体使用方法在[make 函数](https://onnx.ai/onnx/api/helper.html#l-onnx-make-function)中有介绍。

```python
import onnx
import pprint
pprint.pprint([k for k in dir(onnx.helper)
               if k.startswith('make')])
```

```sh
['make_attribute',
 'make_attribute_ref',
 'make_empty_tensor_value_info',
 'make_function',
 'make_graph',
 'make_map',
 'make_map_type_proto',
 'make_model',
 'make_model_gen_version',
 'make_node',
 'make_operatorsetid',
 'make_opsetid',
 'make_optional',
 'make_optional_type_proto',
 'make_sequence',
 'make_sequence_type_proto',
 'make_sparse_tensor',
 'make_sparse_tensor_type_proto',
 'make_sparse_tensor_value_info',
 'make_tensor',
 'make_tensor_sequence_value_info',
 'make_tensor_type_proto',
 'make_tensor_value_info',
 'make_training_info',
 'make_value_info']
```

### Opset 和元数据

让我们加载之前创建好的 `ONNX`文件，看看它有哪些元数据。

```python
from onnx import load

with open("linear_regression.onnx", "rb") as f:
    onnx_model = load(f)

for field in ['doc_string', 'domain', 'functions',
              'ir_version', 'metadata_props', 'model_version',
              'opset_import', 'producer_name', 'producer_version',
              'training_info']:
    print(field, getattr(onnx_model, field))
```

```sh
doc_string 
domain 
functions []
ir_version 10
metadata_props []
model_version 0
opset_import [version: 21]
producer_name 
producer_version 
training_info []
```

其中大部分是空的，因为在创建 `ONNX`计算图时没有填充，其中只有两个有数值的变量：

```python
from onnx import load

with open("linear_regression.onnx", "rb") as f:
    onnx_model = load(f)

print("ir_version:", onnx_model.ir_version)
for opset in onnx_model.opset_import:
    print("opset domain=%r version=%r" % (opset.domain, opset.version))
```

```sh
ir_version: 10
opset domain='' version=21
```

`IR`定义了 `ONNX`语言的版本。 `Opset` 定义了所用算子的版本。 `ONNX` 默认使用最新的版本。 也可以使用另一个版本。

```python
from onnx import load

with open("linear_regression.onnx", "rb") as f:
    onnx_model = load(f)

del onnx_model.opset_import[:]
opset = onnx_model.opset_import.add()
opset.domain = ''
opset.version = 14

for opset in onnx_model.opset_import:
    print("opset domain=%r version=%r" % (opset.domain, opset.version))
```

```sh
opset domain='' version=14
```

算子_`Reshape`_的第 5 版将形状定义为输入，但是在第 1 版中却将属性定义为输入。`opset`定义了在描述计算图时所遵循的规范。

元数据可用于存储任何信息，如有关模型生成方式的信息、也可以用版本号区分不同的模型等。

```python
from onnx import load, helper

with open("linear_regression.onnx", "rb") as f:
    onnx_model = load(f)

onnx_model.model_version = 15
onnx_model.producer_name = "something"
onnx_model.producer_version = "some other thing"
onnx_model.doc_string = "documentation about this model"
prop = onnx_model.metadata_props

data = dict(key1="value1", key2="value2")
helper.set_model_props(onnx_model, data)

print(onnx_model)
```

```sh
ir_version: 10
producer_name: "something"
producer_version: "some other thing"
model_version: 15
doc_string: "documentation about this model"
graph {
  node {
    input: "X"
    input: "A"
    output: "XA"
    op_type: "MatMul"
  }
  node {
    input: "XA"
    input: "B"
    output: "Y"
    op_type: "Add"
  }
  name: "lr"
  input {
    name: "X"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
          }
        }
      }
    }
  }
  input {
    name: "A"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
          }
        }
      }
    }
  }
  input {
    name: "B"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
          }
        }
      }
    }
  }
  output {
    name: "Y"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
        }
      }
    }
  }
}
opset_import {
  version: 21
}
metadata_props {
  key: "key1"
  value: "value1"
}
metadata_props {
  key: "key2"
  value: "value2"
}
```

字段`training_info`可用于存储其他计算图。 参见[training\_tool\_test.py](https://github.com/onnx/onnx/blob/main/onnx/test/training\_tool\_test.py)，了解其工作原理。

### <mark style="color:red;">Functions</mark>

函数可以用来缩短构建模型的代码，并为`runtime`更快地运行预测提供更多可能性。如果没有使用函数，`runtime`只能使用基于现有的算子进行默认地实现。

#### 无属性(attribute)的函数

这是一种简单的情况，函数的每个输入都是执行时已知的动态对象。

```python
import numpy
from onnx import numpy_helper, TensorProto
from onnx.helper import (
    make_model, make_node, set_model_props, make_tensor,
    make_graph, make_tensor_value_info, make_opsetid,
    make_function)
from onnx.checker import check_model

new_domain = 'custom'
opset_imports = [make_opsetid("", 14), make_opsetid(new_domain, 1)]

# Let's define a function for a linear regression

node1 = make_node('MatMul', ['X', 'A'], ['XA'])
node2 = make_node('Add', ['XA', 'B'], ['Y'])

linear_regression = make_function(
    new_domain,            # domain name
    'LinearRegression',     # function name
    ['X', 'A', 'B'],        # input names
    ['Y'],                  # output names
    [node1, node2],         # nodes
    opset_imports,          # opsets
    [])                     # attribute names

# Let's use it in a graph.

X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
B = make_tensor_value_info('B', TensorProto.FLOAT, [None, None])
Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])

graph = make_graph(
    [make_node('LinearRegression', ['X', 'A', 'B'], ['Y1'], domain=new_domain),
     make_node('Abs', ['Y1'], ['Y'])],
    'example',
    [X, A, B], [Y])

onnx_model = make_model(
    graph, opset_imports=opset_imports,
    functions=[linear_regression])  # functions to add)
check_model(onnx_model)

# the work is done, let's display it...
print(onnx_model)
```

```sh
ir_version: 10
graph {
  node {
    input: "X"
    input: "A"
    input: "B"
    output: "Y1"
    op_type: "LinearRegression"
    domain: "custom"
  }
  node {
    input: "Y1"
    output: "Y"
    op_type: "Abs"
  }
  name: "example"
  input {
    name: "X"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
          }
        }
      }
    }
  }
  input {
    name: "A"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
          }
        }
      }
    }
  }
  input {
    name: "B"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
          }
        }
      }
    }
  }
  output {
    name: "Y"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
        }
      }
    }
  }
}
opset_import {
  domain: ""
  version: 14
}
opset_import {
  domain: "custom"
  version: 1
}
functions {
  name: "LinearRegression"
  input: "X"
  input: "A"
  input: "B"
  output: "Y"
  node {
    input: "X"
    input: "A"
    output: "XA"
    op_type: "MatMul"
  }
  node {
    input: "XA"
    input: "B"
    output: "Y"
    op_type: "Add"
  }
  opset_import {
    domain: ""
    version: 14
  }
  opset_import {
    domain: "custom"
    version: 1
  }
  domain: "custom"
}
```

#### 带有属性(attribute)的函数

下面的函数与前面的函数相同，只是将一个输入变量`B`转换成了一个名为`bias`的参数。 代码几乎相同，只是 bias 现在是一个常量。 在函数定义中，创建了一个节点`Constant`，用于将参数作为结果插入。它通过`ref_attr_name` 属性与参数相连。

```python
import numpy
from onnx import numpy_helper, TensorProto, AttributeProto
from onnx.helper import (
    make_model, make_node, set_model_props, make_tensor,
    make_graph, make_tensor_value_info, make_opsetid,
    make_function)
from onnx.checker import check_model

new_domain = 'custom'
opset_imports = [make_opsetid("", 14), make_opsetid(new_domain, 1)]

# Let's define a function for a linear regression
# The first step consists in creating a constant
# equal to the input parameter of the function.
cst = make_node('Constant',  [], ['B'])

att = AttributeProto()
att.name = "value"

# This line indicates the value comes from the argument
# named 'bias' the function is given.
att.ref_attr_name = "bias"
att.type = AttributeProto.TENSOR
cst.attribute.append(att)

node1 = make_node('MatMul', ['X', 'A'], ['XA'])
node2 = make_node('Add', ['XA', 'B'], ['Y'])

linear_regression = make_function(
    new_domain,            # domain name
    'LinearRegression',     # function name
    ['X', 'A'],             # input names
    ['Y'],                  # output names
    [cst, node1, node2],    # nodes
    opset_imports,          # opsets
    ["bias"])               # attribute names

# Let's use it in a graph.

X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
B = make_tensor_value_info('B', TensorProto.FLOAT, [None, None])
Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])

graph = make_graph(
    [make_node('LinearRegression', ['X', 'A'], ['Y1'], domain=new_domain,
               # bias is now an argument of the function and is defined as a tensor
               bias=make_tensor('former_B', TensorProto.FLOAT, [1], [0.67])),
     make_node('Abs', ['Y1'], ['Y'])],
    'example',
    [X, A], [Y])

onnx_model = make_model(
    graph, opset_imports=opset_imports,
    functions=[linear_regression])  # functions to add)
check_model(onnx_model)

# the work is done, let's display it...
print(onnx_model)
```

```sh
ir_version: 10
graph {
  node {
    input: "X"
    input: "A"
    output: "Y1"
    op_type: "LinearRegression"
    attribute {
      name: "bias"
      t {
        dims: 1
        data_type: 1
        float_data: 0.6700000166893005
        name: "former_B"
      }
      type: TENSOR
    }
    domain: "custom"
  }
  node {
    input: "Y1"
    output: "Y"
    op_type: "Abs"
  }
  name: "example"
  input {
    name: "X"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
          }
        }
      }
    }
  }
  input {
    name: "A"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
          }
        }
      }
    }
  }
  output {
    name: "Y"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
        }
      }
    }
  }
}
opset_import {
  domain: ""
  version: 14
}
opset_import {
  domain: "custom"
  version: 1
}
functions {
  name: "LinearRegression"
  input: "X"
  input: "A"
  output: "Y"
  attribute: "bias"
  node {
    output: "B"
    op_type: "Constant"
    attribute {
      name: "value"
      type: TENSOR
      ref_attr_name: "bias"
    }
  }
  node {
    input: "X"
    input: "A"
    output: "XA"
    op_type: "MatMul"
  }
  node {
    input: "XA"
    input: "B"
    output: "Y"
    op_type: "Add"
  }
  opset_import {
    domain: ""
    version: 14
  }
  opset_import {
    domain: "custom"
    version: 1
  }
  domain: "custom"
}
```

### 解析(Parsing)

onnx 模块提供了一种定义计算图更快的方法，而且更易于阅读。如果计算图是在单个函数中构建的，那么使用起来就很容易。

```python
import onnx.parser
from onnx.checker import check_model

input = '''
    <
        ir_version: 8,
        opset_import: [ "" : 15]
    >
    agraph (float[I,J] X, float[I] A, float[I] B) => (float[I] Y) {
        XA = MatMul(X, A)
        Y = Add(XA, B)
    }
    '''
onnx_model = onnx.parser.parse_model(input)
check_model(onnx_model)

print(onnx_model)
```

```sh
ir_version: 8
graph {
node {
    input: "X"
    input: "A"
    output: "XA"
    op_type: "MatMul"
    domain: ""
}
node {
    input: "XA"
    input: "B"
    output: "Y"
    op_type: "Add"
    domain: ""
}
name: "agraph"
input {
    name: "X"
    type {
    tensor_type {
        elem_type: 1
        shape {
        dim {
            dim_param: "I"
        }
        dim {
            dim_param: "J"
        }
        }
    }
    }
}
input {
    name: "A"
    type {
    tensor_type {
        elem_type: 1
        shape {
        dim {
            dim_param: "I"
        }
        }
    }
    }
}
input {
    name: "B"
    type {
    tensor_type {
        elem_type: 1
        shape {
        dim {
            dim_param: "I"
        }
        }
    }
    }
}
output {
    name: "Y"
    type {
    tensor_type {
        elem_type: 1
        shape {
        dim {
            dim_param: "I"
        }
        }
    }
    }
}
}
opset_import {
domain: ""
version: 15
}
```

这种方法常用于创建小型模型。

### 检查器和形状推理(Checker and Shape Inference)

onnx 提供了一个检查模型是否有效的函数。 该函数会检查输入变量的类型和维度是否一致。 下面的示例添加了两个不同类型的矩阵，这是不允许的。

```python
import onnx.parser
import onnx.checker

input = '''
    <
        ir_version: 8,
        opset_import: [ "" : 15]
    >
    agraph (float[I,4] X, float[4,2] A, int[4] B) => (float[I] Y) {
        XA = MatMul(X, A)
        Y = Add(XA, B)
    }
    '''
try:
    onnx_model = onnx.parser.parse_model(input)
    onnx.checker.check_model(onnx_model)
except Exception as e:
    print(e)
```

```sh
b'[ParseError at position (line: 6 column: 44)]\nError context:     agraph (float[I,4] X, float[4,2] A, int[4] B) => (float[I] Y) {\nExpected character ) not found.'
```

`check_model`会由于这种类型不一致而引发错误。 但是对于没有指定域的自定义算子来说，`check_model`不会检查数据的类型和形状。

<mark style="color:red;">形状推理的目的只有一个：估计中间结果的形状和类型。 运行时就可以事先估计内存消耗，优化计算。它可以融合某些运算符，可以就地进行计算。</mark>

```python
import onnx.parser
from onnx import helper, shape_inference

input = '''
    <
        ir_version: 8,
        opset_import: [ "" : 15]
    >
    agraph (float[I,4] X, float[4,2] A, float[4] B) => (float[I] Y) {
        XA = MatMul(X, A)
        Y = Add(XA, B)
    }
    '''
onnx_model = onnx.parser.parse_model(input)
inferred_model = shape_inference.infer_shapes(onnx_model)

print(inferred_model)
```

```sh
ir_version: 8
graph {
  node {
    input: "X"
    input: "A"
    output: "XA"
    op_type: "MatMul"
    domain: ""
  }
  node {
    input: "XA"
    input: "B"
    output: "Y"
    op_type: "Add"
    domain: ""
  }
  name: "agraph"
  input {
    name: "X"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
            dim_param: "I"
          }
          dim {
            dim_value: 4
          }
        }
      }
    }
  }
  input {
    name: "A"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
            dim_value: 4
          }
          dim {
            dim_value: 2
          }
        }
      }
    }
  }
  input {
    name: "B"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
            dim_value: 4
          }
        }
      }
    }
  }
  output {
    name: "Y"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
            dim_param: "I"
          }
        }
      }
    }
  }
  value_info {
    name: "XA"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
            dim_param: "I"
          }
          dim {
            dim_value: 2
          }
        }
      }
    }
  }
}
opset_import {
  domain: ""
  version: 15
}
```

有一个新属性`value_info`，用于存储推断出的形状。`dim_param`中的字母`I` ： `"I"`可以看作一个变量。这取决于输入，但函数能够判断出哪些中间结果将共享相同的维度。 形状推理并非对所有算子都有效，例如，`reshape`算子。形状推理只有在形状恒定的情况下才会起作用。 如果形状不恒定，则无法推理出形状。

### Evaluation and Runtime

`ONNX`标准允许框架以 `ONNX`格式导出模型，并允许使用任何支持`ONNX`格式的后端进行推理。它可用于多种平台，并针对快速推理进行了优化。`ONNX`实现了一个 Python `runtime`，有助于理解模型。 它不用于实际生产中，性能也不是它的目标。

#### 评估线性回归模型

完整的 API 描述请参见[onnx.reference](https://onnx.ai/onnx/api/reference.html#l-reference-implementation)。 它接收一个模型（一个`ModelProto`、一个文件名......）。 方法`run`返回字典中指定的一组输入的输出。

```python
import numpy
from onnx import numpy_helper, TensorProto
from onnx.helper import (
    make_model, make_node, set_model_props, make_tensor,
    make_graph, make_tensor_value_info)
from onnx.checker import check_model
from onnx.reference import ReferenceEvaluator

X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
B = make_tensor_value_info('B', TensorProto.FLOAT, [None, None])
Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])
node1 = make_node('MatMul', ['X', 'A'], ['XA'])
node2 = make_node('Add', ['XA', 'B'], ['Y'])
graph = make_graph([node1, node2], 'lr', [X, A, B], [Y])
onnx_model = make_model(graph)
check_model(onnx_model)

sess = ReferenceEvaluator(onnx_model)

x = numpy.random.randn(4, 2).astype(numpy.float32)
a = numpy.random.randn(2, 1).astype(numpy.float32)
b = numpy.random.randn(1, 1).astype(numpy.float32)
feeds = {'X': x, 'A': a, 'B': b}

print(sess.run(None, feeds))
```

```sh
[array([[-1.5184462 ],
       [-1.0666499 ],
       [-0.77610564],
       [-2.116381  ]], dtype=float32)]
```

#### 评估节点

评估器还可以评估一个简单的节点，以检查算子在特定输入时的表现。

```python
import numpy
from onnx import numpy_helper, TensorProto
from onnx.helper import make_node

from onnx.reference import ReferenceEvaluator

node = make_node('EyeLike', ['X'], ['Y'])

sess = ReferenceEvaluator(node)

x = numpy.random.randn(4, 2).astype(numpy.float32)
feeds = {'X': x}

print(sess.run(None, feeds))
```

```sh
[array([[1., 0.],
       [0., 1.],
       [0., 0.],
       [0., 0.]], dtype=float32)]
```

类似的代码也适用于_GraphProto_或_FunctionProto_。

#### Evaluation Step by Step参数`verbose`会显示中间结果信息。

```python
import numpy
from onnx import numpy_helper, TensorProto
from onnx.helper import (
    make_model, make_node, set_model_props, make_tensor,
    make_graph, make_tensor_value_info)
from onnx.checker import check_model
from onnx.reference import ReferenceEvaluator

X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
B = make_tensor_value_info('B', TensorProto.FLOAT, [None, None])
Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])
node1 = make_node('MatMul', ['X', 'A'], ['XA'])
node2 = make_node('Add', ['XA', 'B'], ['Y'])
graph = make_graph([node1, node2], 'lr', [X, A, B], [Y])
onnx_model = make_model(graph)
check_model(onnx_model)

for verbose in [1, 2, 3, 4]:
    print()
    print(f"------ verbose={verbose}")
    print()
    sess = ReferenceEvaluator(onnx_model, verbose=verbose)

    x = numpy.random.randn(4, 2).astype(numpy.float32)
    a = numpy.random.randn(2, 1).astype(numpy.float32)
    b = numpy.random.randn(1, 1).astype(numpy.float32)
    feeds = {'X': x, 'A': a, 'B': b}

    print(sess.run(None, feeds))
```

```sh
------ verbose=1

[array([[-1.5585198 ],
       [ 0.8431341 ],
       [-3.8161776 ],
       [-0.97695297]], dtype=float32)]

------ verbose=2

MatMul(X, A) -> XA
Add(XA, B) -> Y
[array([[2.7036648],
       [2.1685638],
       [1.3111192],
       [2.7933102]], dtype=float32)]

------ verbose=3

 +I X: float32:(4, 2) in [-0.8107846975326538, 0.8805762529373169]
 +I A: float32:(2, 1) in [0.04560143128037453, 1.3560810089111328]
 +I B: float32:(1, 1) in [-0.5273541212081909, -0.5273541212081909]
MatMul(X, A) -> XA
 + XA: float32:(4, 1) in [-1.0593341588974, 0.13150419294834137]
Add(XA, B) -> Y
 + Y: float32:(4, 1) in [-1.5866882801055908, -0.39584994316101074]
[array([[-1.5866883 ],
       [-1.1906545 ],
       [-0.9883075 ],
       [-0.39584994]], dtype=float32)]

------ verbose=4

 +I X: float32:(4, 2):-0.5727394223213196,0.025708714500069618,0.4316417872905731,0.6972395181655884,-0.39831840991973877...
 +I A: float32:(2, 1):[0.8323841691017151, 0.20654942095279694]
 +I B: float32:(1, 1):[-0.4035758674144745]
MatMul(X, A) -> XA
 + XA: float32:(4, 1):[-0.47142910957336426, 0.5033062100410461, -0.1982671618461609, 0.4859170913696289]
Add(XA, B) -> Y
 + Y: float32:(4, 1):[-0.8750050067901611, 0.09973034262657166, -0.601842999458313, 0.08234122395515442]
[array([[-0.875005  ],
       [ 0.09973034],
       [-0.601843  ],
       [ 0.08234122]], dtype=float32)]
```

#### 评估自定义的节点

下面的示例仍然实现了线性回归，但在矩阵_A_ 中加入了单位矩阵

$$
Y = X(A+I)+B
$$

```python
import numpy
from onnx import numpy_helper, TensorProto
from onnx.helper import (
    make_model, make_node, set_model_props, make_tensor,
    make_graph, make_tensor_value_info)
from onnx.checker import check_model
from onnx.reference import ReferenceEvaluator

X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
B = make_tensor_value_info('B', TensorProto.FLOAT, [None, None])
Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])
node0 = make_node('EyeLike', ['A'], ['Eye'])
node1 = make_node('Add', ['A', 'Eye'], ['A1'])
node2 = make_node('MatMul', ['X', 'A1'], ['XA1'])
node3 = make_node('Add', ['XA1', 'B'], ['Y'])
graph = make_graph([node0, node1, node2, node3], 'lr', [X, A, B], [Y])
onnx_model = make_model(graph)
check_model(onnx_model)
with open("linear_regression.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

sess = ReferenceEvaluator(onnx_model, verbose=2)

x = numpy.random.randn(4, 2).astype(numpy.float32)
a = numpy.random.randn(2, 2).astype(numpy.float32) / 10
b = numpy.random.randn(1, 2).astype(numpy.float32)
feeds = {'X': x, 'A': a, 'B': b}

print(sess.run(None, feeds))
```

```sh
EyeLike(A) -> Eye
Add(A, Eye) -> A1
MatMul(X, A1) -> XA1
Add(XA1, B) -> Y
[array([[ 3.249391  , -1.8001835 ],
       [ 1.4209515 ,  1.7799548 ],
       [-0.37126446, -0.15336311],
       [ 2.164718  ,  0.11840849]], dtype=float32)]
```

如果将运算符_EyeLike_和_Add_合并为_AddEyeLike_，效率会更高。下一个示例将这两个运算符替换为`"optimized"`域中的一个运算符。

```python
import numpy
from onnx import numpy_helper, TensorProto
from onnx.helper import (
    make_model, make_node, set_model_props, make_tensor,
    make_graph, make_tensor_value_info, make_opsetid)
from onnx.checker import check_model

X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
B = make_tensor_value_info('B', TensorProto.FLOAT, [None, None])
Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])

node01 = make_node('AddEyeLike', ['A'], ['A1'], domain='optimized')

node2 = make_node('MatMul', ['X', 'A1'], ['XA1'])
node3 = make_node('Add', ['XA1', 'B'], ['Y'])
graph = make_graph([node01, node2, node3], 'lr', [X, A, B], [Y])

onnx_model = make_model(graph, opset_imports=[
    make_opsetid('', 18), make_opsetid('optimized', 1)
])

check_model(onnx_model)
with open("linear_regression_improved.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

我们需要评估这个模型是否等同于第一个模型，这就需要对`AddEyeLike`这个特定节点进行实现。

```python
import numpy
from onnx.reference import ReferenceEvaluator
from onnx.reference.op_run import OpRun

class AddEyeLike(OpRun):

    op_domain = "optimized"

    def _run(self, X, alpha=1.):
        assert len(X.shape) == 2
        assert X.shape[0] == X.shape[1]
        X = X.copy()
        ind = numpy.diag_indices(X.shape[0])
        X[ind] += alpha
        return (X,)

sess = ReferenceEvaluator("linear_regression_improved.onnx", verbose=2, new_ops=[AddEyeLike])

x = numpy.random.randn(4, 2).astype(numpy.float32)
a = numpy.random.randn(2, 2).astype(numpy.float32) / 10
b = numpy.random.randn(1, 2).astype(numpy.float32)
feeds = {'X': x, 'A': a, 'B': b}

print(sess.run(None, feeds))

# Let's check with the previous model.

sess0 = ReferenceEvaluator("linear_regression.onnx",)
sess1 = ReferenceEvaluator("linear_regression_improved.onnx", new_ops=[AddEyeLike])

y0 = sess0.run(None, feeds)[0]
y1 = sess1.run(None, feeds)[0]
print(y0)
print(y1)
print(f"difference: {numpy.abs(y0 - y1).max()}")
```

```sh
AddEyeLike(A) -> A1
MatMul(X, A1) -> XA1
Add(XA1, B) -> Y
[array([[-0.37408   , -0.56797665],
       [-0.72363484,  1.0973046 ],
       [-1.616734  , -2.5929499 ],
       [-1.7821894 , -0.81231964]], dtype=float32)]
[[-0.37408    -0.56797665]
 [-0.72363484  1.0973046 ]
 [-1.616734   -2.5929499 ]
 [-1.7821894  -0.81231964]]
[[-0.37408    -0.56797665]
 [-0.72363484  1.0973046 ]
 [-1.616734   -2.5929499 ]
 [-1.7821894  -0.81231964]]
difference: 0.0
```

预测结果是一样的。让我们在一个足够大的矩阵上比较一下性能，看看是否有显著差异。

```python
import timeit
import numpy
from onnx.reference import ReferenceEvaluator
from onnx.reference.op_run import OpRun

class AddEyeLike(OpRun):

    op_domain = "optimized"

    def _run(self, X, alpha=1.):
        assert len(X.shape) == 2
        assert X.shape[0] == X.shape[1]
        X = X.copy()
        ind = numpy.diag_indices(X.shape[0])
        X[ind] += alpha
        return (X,)

sess = ReferenceEvaluator("linear_regression_improved.onnx", verbose=2, new_ops=[AddEyeLike])

x = numpy.random.randn(4, 100).astype(numpy.float32)
a = numpy.random.randn(100, 100).astype(numpy.float32) / 10
b = numpy.random.randn(1, 100).astype(numpy.float32)
feeds = {'X': x, 'A': a, 'B': b}

sess0 = ReferenceEvaluator("linear_regression.onnx")
sess1 = ReferenceEvaluator("linear_regression_improved.onnx", new_ops=[AddEyeLike])

y0 = sess0.run(None, feeds)[0]
y1 = sess1.run(None, feeds)[0]
print(f"difference: {numpy.abs(y0 - y1).max()}")
print(f"time with EyeLike+Add: {timeit.timeit(lambda: sess0.run(None, feeds), number=1000)}")
print(f"time with AddEyeLike: {timeit.timeit(lambda: sess1.run(None, feeds), number=1000)}")
```

```sh
difference: 0.0
time with EyeLike+Add: 0.07655519099995445
time with AddEyeLike: 0.06604622999998355
```

在这种情况下，似乎值得添加一个优化节点。 这种优化通常被称为_融合_。 两个连续的算子被融合，成为一个经过优化后的算子。&#x20;

### Implementation details

#### Attributes and inputs

两者之间有明显的区别。输入是动态的，每次执行都可能发生变化。属性永远不会改变，优化器据此可以改进计算图。 因此，不可能将输入转化为属性。 而_Constant_是唯一能将属性转化为输入的算子。

#### Shape or no shape

在处理具有可变维度的深度学习模型时，如何在ONNX格式中有效地表示和处理这些模型，以便它们可以在不同的深度学习框架中使用，这是一个具有挑战性的问题。

```python
import numpy
from onnx import numpy_helper, TensorProto, FunctionProto
from onnx.helper import (
    make_model, make_node, set_model_props, make_tensor,
    make_graph, make_tensor_value_info, make_opsetid,
    make_function)
from onnx.checker import check_model
from onnxruntime import InferenceSession

def create_model(shapes):
    new_domain = 'custom'
    opset_imports = [make_opsetid("", 14), make_opsetid(new_domain, 1)]

    node1 = make_node('MatMul', ['X', 'A'], ['XA'])
    node2 = make_node('Add', ['XA', 'A'], ['Y'])

    X = make_tensor_value_info('X', TensorProto.FLOAT, shapes['X'])
    A = make_tensor_value_info('A', TensorProto.FLOAT, shapes['A'])
    Y = make_tensor_value_info('Y', TensorProto.FLOAT, shapes['Y'])

    graph = make_graph([node1, node2], 'example', [X, A], [Y])

    onnx_model = make_model(graph, opset_imports=opset_imports)
    # Let models runnable by onnxruntime with a released ir_version
    onnx_model.ir_version = 8

    return onnx_model

print("----------- case 1: 2D x 2D -> 2D")
onnx_model = create_model({'X': [None, None], 'A': [None, None], 'Y': [None, None]})
check_model(onnx_model)
sess = InferenceSession(onnx_model.SerializeToString(),
                        providers=["CPUExecutionProvider"])
res = sess.run(None, {
    'X': numpy.random.randn(2, 2).astype(numpy.float32),
    'A': numpy.random.randn(2, 2).astype(numpy.float32)})
print(res)

print("----------- case 2: 2D x 1D -> 1D")
onnx_model = create_model({'X': [None, None], 'A': [None], 'Y': [None]})
check_model(onnx_model)
sess = InferenceSession(onnx_model.SerializeToString(),
                        providers=["CPUExecutionProvider"])
res = sess.run(None, {
    'X': numpy.random.randn(2, 2).astype(numpy.float32),
    'A': numpy.random.randn(2).astype(numpy.float32)})
print(res)

print("----------- case 3: 2D x 0D -> 0D")
onnx_model = create_model({'X': [None, None], 'A': [], 'Y': []})
check_model(onnx_model)
try:
    InferenceSession(onnx_model.SerializeToString(),
                     providers=["CPUExecutionProvider"])
except Exception as e:
    print(e)

print("----------- case 4: 2D x None -> None")
onnx_model = create_model({'X': [None, None], 'A': None, 'Y': None})
try:
    check_model(onnx_model)
except Exception as e:
    print(type(e), e)
sess = InferenceSession(onnx_model.SerializeToString(),
                        providers=["CPUExecutionProvider"])
res = sess.run(None, {
    'X': numpy.random.randn(2, 2).astype(numpy.float32),
    'A': numpy.random.randn(2).astype(numpy.float32)})
print(res)
print("----------- end")
```

```bash
----------- case 1: 2D x 2D -> 2D
[array([[ 0.45804122,  0.4898213 ],
       [ 0.0373224 , -0.00160027]], dtype=float32)]
----------- case 2: 2D x 1D -> 1D
[array([-0.3142326 , -0.05584523], dtype=float32)]
----------- case 3: 2D x 0D -> 0D
[ONNXRuntimeError] : 1 : FAIL : Node () Op (MatMul) [ShapeInferenceError] Input tensors of wrong rank (0).
----------- case 4: 2D x None -> None
<class 'onnx.onnx_cpp2py_export.checker.ValidationError'> Field 'shape' of 'type' is required but missing.
[array([3.5581195, 1.8482882], dtype=float32)]
----------- end
```
