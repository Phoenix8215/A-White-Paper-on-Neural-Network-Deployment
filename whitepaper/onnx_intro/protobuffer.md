# 😉 ONNX中的各类Proto

### **中间表示 —— ONNX**

在介绍 ONNX 之前，我们先从本质上来认识一下神经网络的结构。神经网络实际上只是描述了数据计算的过程，其结构可以用计算图表示。比如 a+b 可以用下面的计算图来表示：

<figure><img src="../../.gitbook/assets/图片 (2) (1) (1) (1).png" alt="" width="563"><figcaption></figcaption></figure>

<mark style="color:red;">为了加速计算，一些框架会使用对神经网络“先编译，后执行”的静态图来描述网络。静态图的缺点是难以描述控制流（比如 if-else 分支语句和 for 循环语句）</mark>，直接对其引入控制语句会导致产生不同的计算图。比如循环执行 n 次 a=a+b，对于不同的 n，会生成不同的计算图：

<figure><img src="../../.gitbook/assets/图片 (3) (1) (1) (1).png" alt=""><figcaption></figcaption></figure>

ONNX （Open Neural Network Exchange）是 Facebook 和微软在2017年共同发布的，用于标准描述计算图的一种格式。目前，在数家机构的共同维护下，ONNX 已经对接了多种深度学习框架和多种推理引擎。因此，ONNX 被当成了深度学习框架到推理引擎的桥梁，就像编译器的中间语言一样。<mark style="color:red;">由于各框架兼容性不一，我们通常只用 ONNX 表示更容易部署的静态图。</mark>

### 各类Proto

ONNX是一种神经网络的格式，采用`Protobuf`二进制形式进行序列化模型。 `Protobuf`会根据用于定义的数据结构来进行序列化存储 同理，我们可以根据官方提供的数据结构信息，去修改或者创建onnx。

<figure><img src="../../.gitbook/assets/image-20240407112310900.png" alt=""><figcaption></figcaption></figure>

onnx的各类proto的定义需要看官方文档([https://github.com/onnx/onnx/tree/main](https://github.com/onnx/onnx/tree/main)) 。这里面的`onnx/onnx.in.proto`定义了所有onnx的Proto。有关onnx的IR(`intermediate representation`)信息，看这里([https://github.com/onnx/onnx/blob/main/docs/IR.md](https://github.com/onnx/onnx/blob/main/docs/IR.md))

### 理解onnx中的组织结构

* `ModelProto`(描述的是整个模型的信息)
* `GraphProto`(描述的是整个网络的信息)
* `NodeProto` (描述的是各个计算节点，比如conv, linear)
* `TensorProto` (描述的是tensor的信息，主要包括权重)
* `ValueInfoProto` (描述的是input/output信息)
* `AttributeProto` (描述的是node节点的各种属性信息)

#### onnx中的ValueInfoProto

一般用来定义网络的`input/output` ，会根据`input/output`的`type`来附加属性

<figure><img src="../../.gitbook/assets/image-20240407122349880-1.png" alt="" width="563"><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image-20240407122445898.png" alt="" width="563"><figcaption></figcaption></figure>

#### onnx中的TensorProto

<figure><img src="../../.gitbook/assets/image-20240407122743299.png" alt="" width="563"><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image-20240407122853847.png" alt="" width="563"><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image-20240407122924505.png" alt="" width="563"><figcaption></figcaption></figure>

一般用来定义一个权重，比如`conv`的`w`和`b` ，`dims`是`repeated`类型，意味着是数组，`raw_data`是`bytes`类型。

#### onnx中的NodeProto

<figure><img src="../../.gitbook/assets/image-20240407125750798.png" alt=""><figcaption></figcaption></figure>

一般用来定义一个计算节点，比如`conv`, `linear` ，`input`是`repeated`类型，意味着是数组，`output`是`repeated`类型，意味着是数组，`attribute`有一个自己的`Proto`，`op_type`需要严格根据`onnx`所提供的`Operators`写。

#### onnx中的AttributeProto

<figure><img src="../../.gitbook/assets/image-20240407130648541.png" alt="" width="563"><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image-20240407130708914.png" alt="" width="563"><figcaption></figcaption></figure>

一般用来定义一个`node`的属性。比如说`kernel size`，比较常见的方式就是把`(key, value)`传入`Proto`，之后 `name = key ，i = value` 。

#### onnx中的GraphProto

<figure><img src="../../.gitbook/assets/image-20240407131258053.png" alt="" width="563"><figcaption></figcaption></figure>

一般用来定义模型的全局信息，比如`opset` ，`graph`并不是`repeated`，所以一个`model`对应一个`graph`。

`ONNX`提供了一些很方便的api来创建`ONNX`：`onnx.helper.make_tensor` `onnx.helper.make_tensor_value_info` ，`onnx.helper.make_attribute` `onnx.helper.make_node` ，`onnx.helper.make_graph` ，`onnx.helper.make_model`。

### 使用API构建ONNX模型(<mark style="color:red;">from scratch</mark>)

* example 1

```python
import onnx
from onnx import helper
from onnx import TensorProto

# 理解onnx中的组织结构
#   - ModelProto (描述的是整个模型的信息)
#   --- GraphProto (描述的是整个网络的信息)
#   ------ NodeProto (描述的是各个计算节点，比如conv, linear)
#   ------ TensorProto (描述的是tensor的信息，主要包括权重)
#   ------ ValueInfoProto (描述的是input/output信息)
#   ------ AttributeProto (描述的是node节点的各种属性信息)


def create_onnx():
    # 创建ValueProto
    a = helper.make_tensor_value_info('a', TensorProto.FLOAT, [10, 10])
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10, 10])
    b = helper.make_tensor_value_info('b', TensorProto.FLOAT, [10, 10])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [10, 10])

    # 创建NodeProto
    # 只能填写onnx支持的算子名称，不能瞎写
    mul = helper.make_node('Mul', ['a', 'x'], 'c', "multiply")
    add = helper.make_node('Add', ['c', 'b'], 'y', "add")

    # 构建GraphProto
    graph = helper.make_graph([mul, add], 'sample-linear', [a, x, b], [y])

    # 构建ModelProto
    model = helper.make_model(graph)

    # 检查model是否有错误
    onnx.checker.check_model(model)
    # print(model)

    # 保存model
    onnx.save(model, "sample-linear.onnx")

    return model


if __name__ == "__main__":
    model = create_onnx()

```

* example 2

```python
import numpy as np
import onnx
from onnx import numpy_helper


def create_initializer_tensor(
        name: str,
        tensor_array: np.ndarray,
        data_type: onnx.TensorProto = onnx.TensorProto.FLOAT
) -> onnx.TensorProto:

    initializer = onnx.helper.make_tensor(
        name      = name,
        data_type = data_type,
        dims      = tensor_array.shape,
        vals      = tensor_array.flatten().tolist())

    return initializer


def main():
    
    input_batch    = 1
    input_channel  = 3
    input_height   = 64
    input_width    = 64
    output_channel = 16

    input_shape    = [input_batch, input_channel, input_height, input_width]
    output_shape   = [input_batch, output_channel, 1, 1]

    ##########################创建input/output################################
    model_input_name  = "input0"
    model_output_name = "output0"

    input = onnx.helper.make_tensor_value_info(
            model_input_name,
            onnx.TensorProto.FLOAT,
            input_shape)

    output = onnx.helper.make_tensor_value_info(
            model_output_name, 
            onnx.TensorProto.FLOAT, 
            output_shape)
    

    ##########################创建第一个conv节点##############################
    conv1_output_name = "conv2d_1.output"
    conv1_in_ch       = input_channel
    conv1_out_ch      = 32
    conv1_kernel      = 3
    conv1_pads        = 1

    # 创建conv节点的权重信息
    conv1_weight    = np.random.rand(conv1_out_ch, conv1_in_ch, conv1_kernel, conv1_kernel)
    conv1_bias      = np.random.rand(conv1_out_ch)

    conv1_weight_name = "conv2d_1.weight"
    conv1_weight_initializer = create_initializer_tensor(
        name         = conv1_weight_name,
        tensor_array = conv1_weight,
        data_type    = onnx.TensorProto.FLOAT)


    conv1_bias_name  = "conv2d_1.bias"
    conv1_bias_initializer = create_initializer_tensor(
        name         = conv1_bias_name,
        tensor_array = conv1_bias,
        data_type    = onnx.TensorProto.FLOAT)

    # 创建conv节点，注意conv节点的输入有3个: input, w, b
    conv1_node = onnx.helper.make_node(
        name         = "conv2d_1",
        op_type      = "Conv",
        inputs       = [
            model_input_name, 
            conv1_weight_name,
            conv1_bias_name
        ],
        outputs      = [conv1_output_name],
        kernel_shape = [conv1_kernel, conv1_kernel],
        pads         = [conv1_pads, conv1_pads, conv1_pads, conv1_pads],
    )

    ##########################创建一个BatchNorm节点###########################
    bn1_output_name = "batchNorm1.output"

    # 为BN节点添加权重信息
    bn1_scale = np.random.rand(conv1_out_ch)
    bn1_bias  = np.random.rand(conv1_out_ch)
    bn1_mean  = np.random.rand(conv1_out_ch)
    bn1_var   = np.random.rand(conv1_out_ch)

    # 通过create_initializer_tensor创建权重，方法和创建conv节点一样
    bn1_scale_name = "batchNorm1.scale"
    bn1_bias_name  = "batchNorm1.bias"
    bn1_mean_name  = "batchNorm1.mean"
    bn1_var_name   = "batchNorm1.var"

    bn1_scale_initializer = create_initializer_tensor(
        name         = bn1_scale_name,
        tensor_array = bn1_scale,
        data_type    = onnx.TensorProto.FLOAT)
    bn1_bias_initializer = create_initializer_tensor(
        name         = bn1_bias_name,
        tensor_array = bn1_bias,
        data_type    = onnx.TensorProto.FLOAT)
    bn1_mean_initializer = create_initializer_tensor(
        name         = bn1_mean_name,
        tensor_array = bn1_mean,
        data_type    = onnx.TensorProto.FLOAT)
    bn1_var_initializer  = create_initializer_tensor(
        name         = bn1_var_name,
        tensor_array = bn1_var,
        data_type    = onnx.TensorProto.FLOAT)

    # 创建BN节点，注意BN节点的输入信息有5个: input, scale, bias, mean, var
    bn1_node = onnx.helper.make_node(
        name    = "batchNorm1",
        op_type = "BatchNormalization",
        inputs  = [
            conv1_output_name,
            bn1_scale_name,
            bn1_bias_name,
            bn1_mean_name,
            bn1_var_name
        ],
        outputs=[bn1_output_name],
    )

    ##########################创建一个ReLU节点###########################
    relu1_output_name = "relu1.output"

    # 创建ReLU节点，ReLU不需要权重，所以直接make_node就好了
    relu1_node = onnx.helper.make_node(
        name    = "relu1",
        op_type = "Relu",
        inputs  = [bn1_output_name],
        outputs = [relu1_output_name],
    )

    ##########################创建一个AveragePool节点####################
    avg_pool1_output_name = "avg_pool1.output"

    # 创建AvgPool节点，AvgPool不需要权重，所以直接make_node就好了
    avg_pool1_node = onnx.helper.make_node(
        name    = "avg_pool1",
        op_type = "GlobalAveragePool",
        inputs  = [relu1_output_name],
        outputs = [avg_pool1_output_name],
    )

    ##########################创建第二个conv节点##############################

    # 创建conv节点的属性
    conv2_in_ch  = conv1_out_ch
    conv2_out_ch = output_channel
    conv2_kernel = 1
    conv2_pads   = 0

    # 创建conv节点的权重信息
    conv2_weight    = np.random.rand(conv2_out_ch, conv2_in_ch, conv2_kernel, conv2_kernel)
    conv2_bias      = np.random.rand(conv2_out_ch)
    
    conv2_weight_name = "conv2d_2.weight"
    conv2_weight_initializer = create_initializer_tensor(
        name         = conv2_weight_name,
        tensor_array = conv2_weight,
        data_type    = onnx.TensorProto.FLOAT)

    conv2_bias_name  = "conv2d_2.bias"
    conv2_bias_initializer = create_initializer_tensor(
        name         = conv2_bias_name,
        tensor_array = conv2_bias,
        data_type    = onnx.TensorProto.FLOAT)

    # 创建conv节点，注意conv节点的输入有3个: input, w, b
    conv2_node = onnx.helper.make_node(
        name         = "conv2d_2",
        op_type      = "Conv",
        inputs       = [
            avg_pool1_output_name,
            conv2_weight_name,
            conv2_bias_name
        ],
        outputs      = [model_output_name],
        kernel_shape = [conv2_kernel, conv2_kernel],
        pads         = [conv2_pads, conv2_pads, conv2_pads, conv2_pads],
    )

    ##########################创建graph##############################
    graph = onnx.helper.make_graph(
        name    = "sample-convnet",
        inputs  = [input],
        outputs = [output],
        nodes   = [
            conv1_node, 
            bn1_node, 
            relu1_node, 
            avg_pool1_node, 
            conv2_node],
        initializer =[
            conv1_weight_initializer, 
            conv1_bias_initializer,
            bn1_scale_initializer, 
            bn1_bias_initializer,
            bn1_mean_initializer, 
            bn1_var_initializer,
            conv2_weight_initializer, 
            conv2_bias_initializer
        ],
    )

    ##########################创建model##############################
    model = onnx.helper.make_model(graph, producer_name="onnx-sample")
    model.opset_import[0].version = 12
    
    ##########################验证&保存model##############################
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    print("Congratulations!! Succeed in creating {}.onnx".format(graph.name))
    onnx.save(model, "sample-convnet.onnx")


# 使用onnx.helper创建一个最基本的ConvNet
#         input (ch=3, h=64, w=64)
#           |
#          Conv (in_ch=3, out_ch=32, kernel=3, pads=1)
#           |
#        BatchNorm
#           |
#          ReLU
#           |
#         AvgPool
#           |
#          Conv (in_ch=32, out_ch=10, kernel=1, pads=0)
#           |
#         output (ch=10, h=1, w=1)

if __name__ == "__main__":
    main()

```

虽然`onnx`官方提供了而一些python API来修改onnx，但是推荐大家使用TensorRT下的`onnxsurgeon`，更加方便快捷。

* example 3

```python
import onnx_graphsurgeon as gs
import numpy as np
import onnx

# onnx_graph_surgeon(gs)中的IR会有以下三种结构
# Tensor
#    -- 有两种类型
#       -- Variable:  主要就是那些不到推理不知道的变量
#       -- Constant:  不用推理时，而在推理前就知道的变量
# Node
#    -- 跟onnx中的NodeProto差不多
# Graph
#    -- 跟onnx中的GraphProto差不多

def main() -> None:
    input = gs.Variable(
            name  = "input0",
            dtype = np.float32,
            shape = (1, 3, 224, 224))

    weight = gs.Constant(
            name  = "conv1.weight",
            values = np.random.randn(5, 3, 3, 3))

    bias   = gs.Constant(
            name  = "conv1.bias",
            values = np.random.randn(5))
    
    output = gs.Variable(
            name  = "output0",
            dtype = np.float32,
            shape = (1, 5, 224, 224))

    node = gs.Node(
            op      = "Conv",
            inputs  = [input, weight, bias],
            outputs = [output],
            attrs   = {"pads":[1, 1, 1, 1]})

    graph = gs.Graph(
            nodes   = [node],
            inputs  = [input],
            outputs = [output])

    model = gs.export_onnx(graph)

    onnx.save(model, "../models/sample-conv.onnx")



# 使用onnx.helper创建一个最基本的ConvNet
#         input (ch=3, h=64, w=64)
#           |
#          Conv (in_ch=3, out_ch=32, kernel=3, pads=1)
#           |
#         output (ch=5, h=64, w=64)

if __name__ == "__main__":
    main()


```

### reference

* [https://github.com/open-mmlab/mmdeploy/blob/main/docs/zh\_cn/tutorial/01\_introduction\_to\_model\_deployment.md](https://github.com/open-mmlab/mmdeploy/blob/main/docs/zh\_cn/tutorial/01\_introduction\_to\_model\_deployment.md)
