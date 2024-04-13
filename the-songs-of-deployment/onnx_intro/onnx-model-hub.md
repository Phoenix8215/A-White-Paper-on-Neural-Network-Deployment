# ONNX Model hub

## ONNX Model Hub

ONNX Model Hub 是一种简单快捷的方法，可让您从 ONNX Model Zoo 开始使用最先进的预训练 ONNX 模型。此外，它还能让研究人员和模型开发人员有机会与更广泛的社区分享他们的预训练模型。

### Install

ONNX Model hub 在 ONNX 1.11.0 之后可用。

### Basic usage

在本节中，我们将演示一些基本功能。

```python
from onnx import hub
```

#### Downloading a model by name:

加载函数将默认在模型库中搜索名称匹配的最新模型，将该模型下载到本地缓存中，并将模型加载到 `ModelProto` 对象中，供 `ONNX runtime`使用。

```python
model = hub.load("resnet50")
```

#### Downloading from custom repositories:

可以提供 repo 参数从指定的ONNX hub中下载：

```python
model = hub.load("resnet50", repo="onnx/models:771185265efbdc049fb223bd68ab1aeb1aecde76")
```

#### Listing and inspecting Models:

也提供了用于查询Model Zoo的 API，以了解更多有关可用模型的信息。这不会下载模型，而只是返回与给定参数相匹配的模型信息

```python
# List all models in the onnx/models:main repo
all_models = hub.list_models()

# List all versions/opsets of a specific model
mnist_models = hub.list_models(model="mnist")

# List all models matching a given "tag"
vision_models = hub.list_models(tags=["vision"])
```

我们还可以使用 `get_model_info` 函数在下载前检查模型的元数据：

```python
print(hub.get_model_info(model="mnist", opset=8))
```

This will print something like:

```sh
ModelInfo(
    model=MNIST,
    opset=8,
    path=vision/classification/mnist/model/mnist-8.onnx,
    metadata={
     'model_sha': '2f06e72de813a8635c9bc0397ac447a601bdbfa7df4bebc278723b958831c9bf',
     'model_bytes': 26454,
     'tags': ['vision', 'classification', 'mnist'],
     'io_ports': {
        'inputs': [{'name': 'Input3', 'shape': [1, 1, 28, 28], 'type': 'tensor(float)'}],
        'outputs': [{'name': 'Plus214_Output_0', 'shape': [1, 10], 'type': 'tensor(float)'}]},
     'model_with_data_path': 'vision/classification/mnist/model/mnist-8.tar.gz',
     'model_with_data_sha': '1dd098b0fe8bc750585eefc02013c37be1a1cae2bdba0191ccdb8e8518b3a882',
     'model_with_data_bytes': 25962}
)
```

### Local Caching

ONNX hub会将下载的模型缓存到一个可配置的本地位置，以便后续调用 hub.load 时无需连接网络。

#### Default cache location

hub客户端按以下顺序查找默认缓存位置：

1. `$ONNX_HOME/hub` 如果环境变量 `ONNX_HOME` 被定义
2. `$XDG_CACHE_HOME/hub` 如果环境变量 `XDG_CACHE_HOME` 被定义
3. `~/.cache/onnx/hub` 其中`~` 是用户的家目录

#### Setting the cache location

使用如下方法手动设置缓存位置:

```python
hub.set_dir("my/cache/directory")
```

此外，还可以通过以下方式检查缓存位置：

```python
print(hub.get_dir())
```

#### Additional cache details

要清除模型缓存，只需使用 `shutil` 或 `os` 等 `python` 工具删除缓存目录即可。此外，还可以使用 `force_reload` 选项覆盖缓存模型：

```python
model = hub.load("resnet50", force_reload=True)
```

### Architecture

ONNX hub由客户端和服务器两个主要部分组成。客户端代码目前包含在 onnx 软件包中，可以在 github 仓库（如 ONNX Model Zoo 中的仓库）中以托管 ONNX\_HUB\_MANIFEST.json 的形式指向服务器。该清单文件是一个 JSON 文档，其中列出了所有模型及其元数据，其设计与编程语言无关。下面是一个格式良好的模型清单条目示例：

```json
{
 "model": "BERT-Squad",
 "model_path": "text/machine_comprehension/bert-squad/model/bertsquad-8.onnx",
 "onnx_version": "1.3",
 "opset_version": 8,
 "metadata": {
     "model_sha": "cad65b9807a5e0393e4f84331f9a0c5c844d9cc736e39781a80f9c48ca39447c",
     "model_bytes": 435882893,
     "tags": ["text", "machine comprehension", "bert-squad"],
     "io_ports": {
         "inputs": [
             {
                 "name": "unique_ids_raw_output___9:0",
                 "shape": ["unk__475"],
                 "type": "tensor(int64)"
             },
             {
                 "name": "segment_ids:0",
                 "shape": ["unk__476", 256],
                 "type": "tensor(int64)"
             },
             {
                 "name": "input_mask:0",
                 "shape": ["unk__477", 256],
                 "type": "tensor(int64)"
             },
             {
                 "name": "input_ids:0",
                 "shape": ["unk__478", 256],
                 "type": "tensor(int64)"
             }
         ],
         "outputs": [
             {
                 "name": "unstack:1",
                 "shape": ["unk__479", 256],
                 "type": "tensor(float)"
             },
             {
                 "name": "unstack:0",
                 "shape": ["unk__480", 256],
                 "type": "tensor(float)"
             },
             {
                 "name": "unique_ids:0",
                 "shape": ["unk__481"],
                 "type": "tensor(int64)"
             }
         ]
     },
     "model_with_data_path": "text/machine_comprehension/bert-squad/model/bertsquad-8.tar.gz",
     "model_with_data_sha": "c8c6c7e0ab9e1333b86e8415a9d990b2570f9374f80be1c1cb72f182d266f666",
     "model_with_data_bytes": 403400046
 }
}
```

These important fields are:

* `model`: 用于查询的模型名称
* `model_path`: 存储在 Git LFS 中的模型的相对路径。
* `onnx_version`: ONNX 模型的版本
* `opset_version`: opset 的版本。如果未指定，客户端将下载最新的 opset。
* `metadata/model_sha`: Optional 模型sha校验码，提高下载安全性
* `metadata/tags`: Optional 帮助用户按特定类型查找模型

`metadata`中的所有其他字段对客户端来说都是可选的，但却能为用户提供重要的详细信息。
