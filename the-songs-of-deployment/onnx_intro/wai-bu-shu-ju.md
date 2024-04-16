---
layout:
  title:
    visible: true
  description:
    visible: false
  tableOfContents:
    visible: true
  outline:
    visible: true
  pagination:
    visible: true
---

# 🤣 外部数据

## External Data

用外部数据加载 ONNX 模型

* \[默认] 如果外部数据位于模型的同一目录下，只需使用 `onnx.load()` 即可。

```python
import onnx

onnx_model = onnx.load("path/to/the/model.onnx")
```

* 如果外部数据位于其他目录下，请使用 `load_external_data_for_model()` 指定目录路径，然后使用 `onnx.load()` 加载数据。

```python
import onnx
from onnx.external_data_helper import load_external_data_for_model

onnx_model = onnx.load("path/to/the/model.onnx", load_external_data=False)
load_external_data_for_model(onnx_model, "data/directory/path/")
# Then the onnx_model has loaded the external data from the specific directory
```

### 将 ONNX 模型转换为外部数据

```python
import onnx
from onnx.external_data_helper import convert_model_to_external_data

onnx_model = ... # Your model in memory as ModelProto
convert_model_to_external_data(onnx_model, all_tensors_to_one_file=True, location="filename", size_threshold=1024, convert_attribute=False)
# Must be followed by save_model to save the converted model to a specific path
onnx.save_model(onnx_model, "path/to/save/the/model.onnx")
# Then the onnx_model has converted raw data as external data and saved to specific directory
```

### 将 ONNX 模型转换为外部数据并保存

```python
import onnx

onnx_model = ... # Your model in memory as ModelProto
onnx.save_model(onnx_model, "path/to/save/the/model.onnx", save_as_external_data=True, all_tensors_to_one_file=True, location="filename", size_threshold=1024, convert_attribute=False)
# Then the onnx_model has converted raw data as external data and saved to specific directory
```

### `onnx.checker` 用于带有外部数据的模型

#### Models with External Data (<2GB)

当前检查器支持检查带有外部数据的模型。向检查器指定已加载的 onnx 模型或模型路径。

#### Large models >2GB

不过，对于大于 2GB 的模型，请使用模型路径，外部数据需要位于同一目录下。

```python
import onnx

onnx.checker.check_model("path/to/the/model.onnx")
# onnx.checker.check_model(loaded_onnx_model) will fail if given >2GB model
```

### TensorProto: data\_location and external\_data fields

在 `TensorProto` 消息类型中，有两个字段与外部数据有关。

#### data\_location field

`data_location` 字段存储此张量的数据位置。其值必须是:

* `MESSAGE` - 存储在 `protobuf` 报文内部特定类型字段中的数据。
* `RAW` - 存储在 `raw_data` 字段中的数据。
* `EXTERNAL` - 存储在外部位置的数据，如 `external_data` 字段所述。
* `value` not set - legacy value. Assume data is stored in raw\_data (if set) otherwise in message.

#### external\_data field

`external_data` 字段存储描述数据位置的键值对

可识别的键有:

* `"location"` (required) - 相对于存储 ONNX `protobuf` 模型的文件系统目录的路径。不允许使用诸如...之类的上层目录路径。
* `"offset"` (optional) - 存储数据开始的字节位置。以字符串形式存储的整数。偏移值应是 4096（页面大小）的倍数，以便支持 mmap。
* `"length"` (optional) - 包含数据的字节数。以字符串形式存储的整数。
* `"checksum"` (optional) - `location`所指向文件的SHA1 摘要。

加载 ONNX 文件后，所有 `external_data` 字段都可以用一个附加键（`"basepath"`）来更新，该键存储加载 ONNX 模型文件的目录路径。

#### External data files

外部数据文件中存储的数据将采用与当前 ONNX 中 `raw_data` 字段相同的二进制字节字符串格式。

Reference [https://github.com/onnx/onnx/pull/678](https://github.com/onnx/onnx/pull/678)
