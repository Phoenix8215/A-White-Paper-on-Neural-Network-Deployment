# ✌️ Polygraphy-Cheatsheet

### 安装Polygraphy

```bash
python -m pip install colored polygraphy --extra-index-url https://pypi.ngc.nvidia.com
```

## API

### 从ONNX模型导出engine，并执行推理，比较输出结果

{% hint style="info" %}
以下两个模型均为`identity`模型
{% endhint %}

```python
"""
This script builds and runs a TensorRT engine with FP16 precision enabled
starting from an ONNX identity model.
"""
import numpy as np
from polygraphy.backend.trt import CreateConfig, EngineFromNetwork, NetworkFromOnnxPath, SaveEngine, TrtRunner


def main():
    # We can compose multiple lazy loaders together to get the desired conversion.
    # In this case, we want ONNX -> TensorRT Network -> TensorRT engine (w/ fp16).
    #
    # NOTE: `build_engine` is a *callable* that returns an engine, not the engine itself.
    #   To get the engine directly, you can use the immediately evaluated functional API.
    #   See examples/api/06_immediate_eval_api for details.
    build_engine = EngineFromNetwork(
        NetworkFromOnnxPath("identity.onnx"), config=CreateConfig(fp16=True)
    )  # Note that config is an optional argument.

    # To reuse the engine elsewhere, we can serialize and save it to a file.
    # The `SaveEngine` lazy loader will return the TensorRT engine when called,
    # which allows us to chain it together with other loaders.
    build_engine = SaveEngine(build_engine, path="identity.engine")

    # Once our loader is ready, inference is simply a matter of constructing a runner,
    # activating it with a context manager (i.e. `with TrtRunner(...)`) and calling `infer()`.
    #
    # NOTE: You can use the activate() function instead of a context manager, but you will need to make sure to
    # deactivate() to avoid a memory leak. For that reason, a context manager is the safer option.
    with TrtRunner(build_engine) as runner:
        inp_data = np.ones(shape=(1, 1, 2, 2), dtype=np.float32)

        # NOTE: The runner owns the output buffers and is free to reuse them between `infer()` calls.
        # Thus, if you want to store results from multiple inferences, you should use `copy.deepcopy()`.
        outputs = runner.infer(feed_dict={"x": inp_data})

        assert np.array_equal(outputs["y"], inp_data)  # It's an identity model!

        print("Inference succeeded!")


if __name__ == "__main__":
    main()
```

### 直接导入engine，并执行推理，比较输出结果

```python
"""
This script loads the TensorRT engine built by `build_and_run.py` and runs inference.
"""
import numpy as np
from polygraphy.backend.common import BytesFromPath
from polygraphy.backend.trt import EngineFromBytes, TrtRunner


def main():
    # Just as we did when building, we can compose multiple loaders together
    # to achieve the behavior we want. Specifically, we want to load a serialized
    # engine from a file, then deserialize it into a TensorRT engine.
    load_engine = EngineFromBytes(BytesFromPath("identity.engine"))

    # Inference remains virtually exactly the same as before:
    with TrtRunner(load_engine) as runner:
        inp_data = np.ones(shape=(1, 1, 2, 2), dtype=np.float32)

        # NOTE: The runner owns the output buffers and is free to reuse them between `infer()` calls.
        # Thus, if you want to store results from multiple inferences, you should use `copy.deepcopy()`.
        outputs = runner.infer(feed_dict={"x": inp_data})

        assert np.array_equal(outputs["y"], inp_data)  # It's an identity model!

        print("Inference succeeded!")


if __name__ == "__main__":
    main()
```

### 快速查看模型结构

```bash
polygraphy inspect model *.onnx/*.engine
```

* 输出结果示例

```bash
[I] Loading bytes from /root/fz/api/00_inference_with_tensorrt/identity.engine
[I] ==== TensorRT Engine ====
    Name: Unnamed Network 0 | Explicit Batch Engine
    
    ---- 1 Engine Input(s) ----
    {x [dtype=float32, shape=(1, 1, 2, 2)]}
    
    ---- 1 Engine Output(s) ----
    {y [dtype=float32, shape=(1, 1, 2, 2)]}
    
    ---- Memory ----
    Device Memory: 0 bytes
    
    ---- 1 Profile(s) (2 Tensor(s) Each) ----
    - Profile: 0
        Tensor: x          (Input), Index: 0 | Shapes: min=(1, 1, 2, 2), opt=(1, 1, 2, 2), max=(1, 1, 2, 2)
        Tensor: y         (Output), Index: 1 | Shape: (1, 1, 2, 2)
    
    ---- 1 Layer(s) ----
```

### 比较多个不同推理后端的输出

```python
"""
This script runs an identity model with ONNX-Runtime and TensorRT,
then compares outputs.
"""
from polygraphy.backend.onnxrt import OnnxrtRunner, SessionFromOnnx
from polygraphy.backend.trt import EngineFromNetwork, NetworkFromOnnxPath, TrtRunner
from polygraphy.comparator import Comparator, CompareFunc


def main():
    # The OnnxrtRunner requires an ONNX-RT session.
    # We can use the SessionFromOnnx lazy loader to construct one easily:
    build_onnxrt_session = SessionFromOnnx("identity.onnx")

    # The TrtRunner requires a TensorRT engine.
    # To create one from the ONNX model, we can chain a couple lazy loaders together:
    build_engine = EngineFromNetwork(NetworkFromOnnxPath("identity.onnx"))

    runners = [
        TrtRunner(build_engine),
        OnnxrtRunner(build_onnxrt_session),
    ]

    # `Comparator.run()` will run each runner separately using synthetic input data and
    #   return a `RunResults` instance. See `polygraphy/comparator/struct.py` for details.
    #
    # TIP: To use custom input data, you can set the `data_loader` parameter in `Comparator.run()``
    #   to a generator or iterable that yields `Dict[str, np.ndarray]`.
    run_results = Comparator.run(runners)

    # `Comparator.compare_accuracy()` checks that outputs match between runners.
    #
    # TIP: The `compare_func` parameter can be used to control how outputs are compared (see API reference for details).
    #   The default comparison function is created by `CompareFunc.simple()`, but we can construct it
    #   explicitly if we want to change the default parameters, such as tolerance.
    assert bool(Comparator.compare_accuracy(run_results, compare_func=CompareFunc.simple(atol=1e-8)))

    # We can use `RunResults.save()` method to save the inference results to a JSON file.
    # This can be useful if you want to generate and compare results separately.
    run_results.save("inference_results.json")


if __name__ == "__main__":
    main()

```

查看多个`runner`的输出结果

```bash
polygraphy inspect data inference_results.json
```

* 输出结果示例

```bash
[I] ==== Run Results (2 runners) ====
    
    ---- trt-runner-N0-04/12/24-13:52:48    (1 iterations) ----
    
    y [dtype=float32, shape=(1, 1, 2, 2)] | Stats: mean=0.35995, std-dev=0.25784, var=0.066482, median=0.35968, min=0.00011437 at (0, 0, 1, 0), max=0.72032 at (0, 0, 0, 1), avg-magnitude=0.35995
    ---- onnxrt-runner-N0-04/12/24-13:52:48 (1 iterations) ----
    
    y [dtype=float32, shape=(1, 1, 2, 2)] | Stats: mean=0.35995, std-dev=0.25784, var=0.066482, median=0.35968, min=0.00011437 at (0, 0, 1, 0), max=0.72032 at (0, 0, 0, 1), avg-magnitude=0.35995
```

### 验证单个推理后端的输出结果

`Polygraphy` 提供的`Comparator`可用于比较多个`runner`的少量结果，但不太适合用使用真实数据集来验证单个`runner`，尤其是在数据集较大的情况下。

在数据集较大的情况下，建议直接使用`runner`。与使用`Comparator`不同，使用`runner`可以完全自由地决定如何加载输入数据以及如何验证输出结果。

{% hint style="info" %}
_可以使用 `data_loader`_ 参数为 `Comparator.run()` 提供自定义的输入数据。
{% endhint %}

```python
"""
This script uses the Polygraphy Runner API to validate the outputs
of an identity model using a trivial dataset.
"""
import numpy as np
from polygraphy.backend.trt import EngineFromNetwork, NetworkFromOnnxPath, TrtRunner

# Pretend that this is a very large dataset.
REAL_DATASET = [
    np.ones((1, 1, 2, 2), dtype=np.float32),
    np.zeros((1, 1, 2, 2), dtype=np.float32),
    np.ones((1, 1, 2, 2), dtype=np.float32),
    np.zeros((1, 1, 2, 2), dtype=np.float32),
]  # Definitely real data

# For an identity network, the golden output values are the same as the input values.
# Though such a network appears useless at first glance, it can be very useful in some cases (like here!).
EXPECTED_OUTPUTS = REAL_DATASET


def main():
    build_engine = EngineFromNetwork(NetworkFromOnnxPath("identity.onnx"))

    with TrtRunner(build_engine) as runner:
        for (data, golden) in zip(REAL_DATASET, EXPECTED_OUTPUTS):
            # NOTE: The runner owns the output buffers and is free to reuse them between `infer()` calls.
            #   Thus, if you want to store results from multiple inferences, you should use `copy.deepcopy()`.
            outputs = runner.infer(feed_dict={"x": data})

            assert np.array_equal(outputs["y"], golden)

        print("Validation succeeded!")


if __name__ == "__main__":
    main()

```

### TensorRT API和Polygraphy互操作

Polygraphy 的一个主要特点是与 TensorRT 以及其他后端完全互操作。由于 Polygraphy 没有隐藏底层后端 API，因此可以在 Polygraphy API 和 后端 API (TensorRT等)之间自由切换使用。

Polygraphy 提供了一个 `extend` 装饰器，可用于轻松扩展现有的 Polygraphy 加载器(`loaders`)。 可以用来解决以下情况：

* 在构建引擎前修改 TensorRT 网络
* 使用 Polygraphy 目前不支持的 TensorRT `builder flag`

```python
"""
This script demonstrates how to use Polygraphy in conjunction with APIs
provided by a backend. Specifically, in this case, we use TensorRT APIs
to print the network name and enable FP16 mode.
"""
import numpy as np
import tensorrt as trt
from polygraphy import func
from polygraphy.backend.trt import CreateConfig, EngineFromNetwork, NetworkFromOnnxPath, TrtRunner


# TIP: The immediately evaluated functional API makes it very easy to interoperate
# with backends like TensorRT. For details, see example 06 (`examples/api/06_immediate_eval_api`).

# We can use the `extend` decorator to easily extend lazy loaders provided by Polygraphy
# The parameters our decorated function takes should match the return values of the loader we are extending.

# For `NetworkFromOnnxPath`, we can see from the API documentation that it returns a TensorRT
# builder, network and parser. That is what our function will receive.
@func.extend(NetworkFromOnnxPath("identity.onnx")) # 调用TensorRT的方法，将返回值传给下面的形式参数
def load_network(builder, network, parser):
    # Here we can modify the network. For this example, we'll just set the network name.
    network.name = "MyIdentity"
    print(f"Network name: {network.name}")

    # Notice that we don't need to return anything - `extend()` takes care of that for us!


# In case a builder configuration option is missing from Polygraphy, we can easily set it using TensorRT APIs.
# Our function will receive a TensorRT IBuilderConfig since that's what `CreateConfig` returns.
@func.extend(CreateConfig())
def load_config(config):
    # Polygraphy supports the fp16 flag, but in case it didn't, we could do this:
    config.set_flag(trt.BuilderFlag.FP16)


def main():
    # Since we have no further need of TensorRT APIs, we can come back to regular Polygraphy.
    #
    # NOTE: Since we're using lazy loaders, we provide the functions as arguments - we do *not* call them ourselves.
    # 提供函数地址即可，无需调用
    build_engine = EngineFromNetwork(load_network, config=load_config)

    with TrtRunner(build_engine) as runner:
        inp_data = np.ones(shape=(1, 1, 2, 2), dtype=np.float32)

        # NOTE: The runner owns the output buffers and is free to reuse them between `infer()` calls.
        # Thus, if you want to store results from multiple inferences, you should use `copy.deepcopy()`.
        outputs = runner.infer({"x": inp_data})

        assert np.array_equal(outputs["y"], inp_data)  # It's an identity model!

        print("Inference succeeded!")


if __name__ == "__main__":
    main()

```

查看代码将生成的TensorRT网络

```bash
polygraphy inspect model example.py --trt-network-func load_network --show layers attrs weights
```

### TensorRT中的int8校准

TensorRT 中的 Int8 校准涉及向 TensorRT 提供一组有代表性的输入数据，作为`engine`构建过程的一部分。在 TensorRT [calibration API](https://docs.nvidia.com/deeplearning/tensorrt/api/python\_api/infer/Int8/Calibrator.html)中，要求用户将输入数据复制到 GPU 并管理 TensorRT 生成的校准缓存。

Polygraphy 提供了校准器，既可与 Polygraphy 一起使用，也可直接与 TensorRT 一起使用。在后一种情况下，Polygraphy 校准器的行为与普通 TensorRT int8 校准器完全相同。

```python
"""
This script demonstrates how to use the Calibrator API provided by Polygraphy
to calibrate a TensorRT engine to run in INT8 precision.
"""
import numpy as np
from polygraphy.backend.trt import Calibrator, CreateConfig, EngineFromNetwork, NetworkFromOnnxPath, TrtRunner
from polygraphy.logger import G_LOGGER


# The data loader argument to `Calibrator` can be any iterable or generator that yields `feed_dict`s.
# A `feed_dict` is just a mapping of input names to corresponding inputs.
def calib_data():
    for _ in range(4):
        # TIP: If your calibration data is already on the GPU, you can instead provide GPU pointers
        # (as `int`s) or Polygraphy `DeviceView`s instead of NumPy arrays.
        #
        # For details on `DeviceView`, see `polygraphy/cuda/cuda.py`.
        yield {"x": np.ones(shape=(1, 1, 2, 2), dtype=np.float32)}  # Totally real data


def main():
    # We can provide a path or file-like object if we want to cache calibration data.
    # This lets us avoid running calibration the next time we build the engine.
    #
    # TIP: You can use this calibrator with TensorRT APIs directly (e.g. config.int8_calibrator).
    # You don't have to use it with Polygraphy loaders if you don't want to.
    calibrator = Calibrator(data_loader=calib_data(), cache="identity-calib.cache")

    # We must enable int8 mode in addition to providing the calibrator.
    build_engine = EngineFromNetwork(
        NetworkFromOnnxPath("identity.onnx"), config=CreateConfig(int8=True, calibrator=calibrator)
    )

    # When we activate our runner, it will calibrate and build the engine. If we want to
    # see the logging output from TensorRT, we can temporarily increase logging verbosity:
    with G_LOGGER.verbosity(G_LOGGER.VERBOSE), TrtRunner(build_engine) as runner:
        # Finally, we can test out our int8 TensorRT engine with some dummy input data:
        inp_data = np.ones(shape=(1, 1, 2, 2), dtype=np.float32)

        # NOTE: The runner owns the output buffers and is free to reuse them between `infer()` calls.
        # Thus, if you want to store results from multiple inferences, you should use `copy.deepcopy()`.
        outputs = runner.infer({"x": inp_data})

        assert np.array_equal(outputs["y"], inp_data)  # It's an identity model!


if __name__ == "__main__":
    main()

```

### 使用TensorRT API搭建网络

```python
"""
This script demonstrates how to construct a TensorRT network using the TensorRT Network API.
"""
import numpy as np
import tensorrt as trt
from polygraphy import func
from polygraphy.backend.trt import CreateNetwork, EngineFromNetwork, TrtRunner


INPUT_NAME = "input"
INPUT_SHAPE = (64, 64)
OUTPUT_NAME = "output"


# Just like in example 03, we can use `extend` to add our own functionality to existing lazy loaders.
# `CreateNetwork` will create an empty network, which we can then populate ourselves.
@func.extend(CreateNetwork())
def create_network(builder, network):
    # This network will add 1 to the input tensor.
    inp = network.add_input(name=INPUT_NAME, shape=INPUT_SHAPE, dtype=trt.float32)
    ones = network.add_constant(shape=INPUT_SHAPE, weights=np.ones(shape=INPUT_SHAPE, dtype=np.float32)).get_output(0)
    add = network.add_elementwise(inp, ones, op=trt.ElementWiseOperation.SUM).get_output(0)
    add.name = OUTPUT_NAME
    network.mark_output(add)

    # Notice that we don't need to return anything - `extend()` takes care of that for us!


def main():
    # After we've constructed the network, we can go back to using regular Polygraphy APIs.
    #
    # NOTE: Since we're using lazy loaders, we provide the `create_network` function as
    # an argument - we do *not* call it ourselves.
    build_engine = EngineFromNetwork(create_network)

    with TrtRunner(build_engine) as runner:
        feed_dict = {INPUT_NAME: np.random.random_sample(INPUT_SHAPE).astype(np.float32)}

        # NOTE: The runner owns the output buffers and is free to reuse them between `infer()` calls.
        # Thus, if you want to store results from multiple inferences, you should use `copy.deepcopy()`.
        outputs = runner.infer(feed_dict)

        assert np.array_equal(outputs[OUTPUT_NAME], (feed_dict[INPUT_NAME] + 1))

        print("Inference succeeded!")


if __name__ == "__main__":
    main()

```

查看代码将生成的TensorRT网络

```bash
polygraphy inspect model example.py --trt-network-func create_network --show layers attrs weights
```

### 及时评估函数API

大多数情况下，Polygraphy 附带的惰性加载器(loaders)有几个优点：

* 把工作推迟到真正需要的时候再做，可以节省时间和空间。
* 由于构建的加载器非常轻便，因此使用惰性评估加载器(`lazily evaluated loaders`)的运行程序可以很容易地被复制到其他进程或线程中，然后在那里启动。 如果运行程序引用的是整个模型或者推理会话，那么以这种方式复制它们就不是一件容易的事。
*   它们允许我们通过将多个加载器组合在一起来提前定义一系列的算子。 例如，我们可以创建一个加载器，从 ONNX 导入模型并生成序列化的 TensorRT 引擎：

    ```python
     build_engine = EngineBytesFromNetwork(NetworkFromOnnxPath("/path/to/model.onnx"))
    ```
* 如果向加载器提供了可调用对象，加载器就会获得返回值的所有权。

然而，这有时会导致代码的可读性降低。 例如，请看下面的例子：

```python
 # Each line in this example looks almost the same, but has significantly
 # different behavior. Some of these lines even cause memory leaks!
 EngineBytesFromNetwork(NetworkFromOnnxPath("/path/to/model.onnx")) # This is a loader instance, not an engine!
 EngineBytesFromNetwork(NetworkFromOnnxPath("/path/to/model.onnx"))() # This is an engine.
 EngineBytesFromNetwork(NetworkFromOnnxPath("/path/to/model.onnx")()) # And it's a loader instance again...
 EngineBytesFromNetwork(NetworkFromOnnxPath("/path/to/model.onnx")())() # Back to an engine!
 EngineBytesFromNetwork(NetworkFromOnnxPath("/path/to/model.onnx"))()() # This throws - can you see why?
```

因此，Polygraphy 为每个加载器提供了相对应的及时评估函数(`immediately-evaluated functional equivalents`)。

```python
 parse_network = NetworkFromOnnxPath("/path/to/model.onnx")
 create_config = CreateConfig(fp16=True, tf32=True)
 build_engine = EngineFromNetwork(parse_network, create_config)
 engine = build_engine()
```

对应于:

```python
 builder, network, parser = network_from_onnx_path("/path/to/model.onnx")
 config = create_config(builder, network, fp16=True, tf32=True)   
 engine = engine_from_network((builder, network, parser), config)
```

下面的代码主要说明了如何利用函数式 API 将 ONNX 模型转换为 `TensorRT` 网络、修改`TensorRT`网络、构建启用了 FP16 精度的 `TensorRT` 引擎并运行推理。 最后还将把引擎保存到文件中，看看如何再次加载并运行推理。

```python
"""
This script uses Polygraphy's immediately evaluated functional APIs
to load an ONNX model, convert it into a TensorRT network, add an identity
layer to the end of it, build an engine with FP16 mode enabled,
save the engine, and finally run inference.
"""
import numpy as np
from polygraphy.backend.trt import TrtRunner, create_config, engine_from_network, network_from_onnx_path, save_engine


def main():
    # In Polygraphy, loaders and runners take ownership of objects if they are provided
    # via the return values of callables. 
    #我们不用关心对象的生命周期如果我们使用的是惰性加载器
    # 因为我们是即使评估的，我们获得了对象的所有权，而且要负责资源释放
    builder, network, parser = network_from_onnx_path("identity.onnx")

    # 给网络扩展一个 identity 层
    # 如果使用延迟加载器，需要 func.extend() 函数，如示例03和示例05中那样。
    prev_output = network.get_output(0)
    network.unmark_output(prev_output)
    output = network.add_identity(prev_output).get_output(0)
    output.name = "output"
    network.mark_output(output)

    # Create a TensorRT IBuilderConfig so that we can build the engine with FP16 enabled.
    config = create_config(builder, network, fp16=True)

    # We can free everything we constructed above once we're done building the engine.
    # NOTE: In TensorRT 8.0 and newer, we do *not* need to use a context manager here.
    with builder, network, parser, config:
        engine = engine_from_network((builder, network), config)

    # To reuse the engine elsewhere, we can serialize it and save it to a file.
    save_engine(engine, path="identity.engine")

    # NOTE: In TensorRT 8.0 and newer, we do *not* need to use a context manager to free `engine`.
    with engine, TrtRunner(engine) as runner:
        inp_data = np.ones((1, 1, 2, 2), dtype=np.float32)

        # NOTE: The runner owns the output buffers and is free to reuse them between `infer()` calls.
        # Thus, if you want to store results from multiple inferences, you should use `copy.deepcopy()`.
        outputs = runner.infer(feed_dict={"x": inp_data})

        assert np.array_equal(outputs["output"], inp_data)  # It's an identity model!

        print("Inference succeeded!")


if __name__ == "__main__":
    main()

```

```python
"""
This script uses Polygraphy's immediately evaluated functional APIs
to load the TensorRT engine built by `build_and_run.py` and run inference.
"""
import numpy as np
from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.trt import TrtRunner, engine_from_bytes


def main():
    engine = engine_from_bytes(bytes_from_path("identity.engine"))

    # NOTE: In TensorRT 8.0 and newer, we do *not* need to use a context manager to free `engine`.
    with engine, TrtRunner(engine) as runner:
        inp_data = np.ones((1, 1, 2, 2), dtype=np.float32)

        # NOTE: The runner owns the output buffers and is free to reuse them between `infer()` calls.
        # Thus, if you want to store results from multiple inferences, you should use `copy.deepcopy()`.
        outputs = runner.infer(feed_dict={"x": inp_data})

        assert np.array_equal(outputs["output"], inp_data)  # It's an identity model!

        print("Inference succeeded!")


if __name__ == "__main__":
    main()

```

### 动态shape

为了在 TensorRT 中使用动态输入维度，我们必须在构建引擎时指定一个可能的维度范围。TensorRT 优化配置文件提供了这样的方法，主要包括以下两个步骤：

1. 在引擎构建过程中，指定一个或多个优化配置文件。 一个优化配置文件包括 3种维度：
   * `min`：最小的形状。
   * `opt`：TensorRT应该优化的形状， 一般来说对应于最常用的形状。
   * `max`: 最大的形状
2.  在推理过程中，在执行上下文中设置输入形状，然后查询执行上下文以确定输出形状，调整设备缓冲区的大小，以容纳整个输出。

    对于单输入、单输出模型，大致流程如下：

    ```python
    1context.set_binding_shape(0, inp.shape)

    out_shape = context.get_binding_shape
    out_buf.resize(out_shape)

    # Rest of inference code...
    ```

`Polygraphy`可以简化这两个步骤：

1.  它提供了一个 `Profile` 抽象类，`OrderedDict`类型，可以转换为 TensorRT 的`IOptimizationProfile` ，并包含一些实用功能：

    * `fill_defaults`：使用网络的默认`shape`填充配置文件。
    * `to_trt`：使用Profile中的`shape`创建 TensorRT `IOptimizationProfile`。

    此外，Profile 还能自动处理`shape-tensor`和`non-shape-tensor`这类的输入。
2. `TrtRunner` 会自动处理模型中的动态`shape`。 与`Profile`一样，还能自动处理`shape-tensor`和`non-shape-tensor`这类的输入。此外，运行程序只会在需要时更新上下文绑定的形状、因为改变形状的开销很小。只有当输出设备缓冲区的当前大小小于上下文输出时，才会调整其大小，避免不必要的重新分配。

#### Setting The Stage

为了举例说明，我们设想一个假设的场景：

我们正在使用图像分类模型运行推理工作。通常，我们在在线情况下使用这种模型，即我们希望尽可能缩短延迟时间，因此我们将一次处理一幅图像。在这种情况下，假设批量大小为 `[1]`。

但是，如果用户数量过多，我们就需要采用动态批处理，这样才不会影响吞吐量。我们的批处理规模范围仍然很小，以保持可接受的延迟。我们最常用的批量大小是 4。

在这种情况下，假设 `batch_size` 的范围为 `[1, 32]`。在更罕见的情况下，我们需要离线处理大量数据。在这种情况下，我们使用非常大的批处理量来提高吞吐量。 在这种情况下，假设 `batch_size` 为 `[128]`。

#### Performance Considerations

在实施推理过程中，我们需要考虑一些取舍：

* <mark style="color:red;">一个</mark><mark style="color:red;">`batchsize`</mark><mark style="color:red;">较大的</mark><mark style="color:red;">`Profile`</mark><mark style="color:red;">在整个范围内的表现不如多个</mark><mark style="color:red;">`batchsize`</mark><mark style="color:red;">较小的</mark><mark style="color:red;">`Profile`</mark><mark style="color:red;">。</mark>
* <mark style="color:red;">在</mark><mark style="color:red;">`Profile`</mark><mark style="color:red;">内切换</mark><mark style="color:red;">`shape`</mark><mark style="color:red;">的成本很小，但并不为零。</mark>
*   <mark style="color:red;">在上下文中切换配置文件的成本比在配置文件中切换</mark><mark style="color:red;">`shape`</mark><mark style="color:red;">的成本更高。</mark>

    我们可以为每个配置文件创建单独的执行上下文，并在运行时选择适当的上下文，从而避免切换配置文件的成本。 不过，请记住，每个上下文都需要一些额外的内存。

#### A Possible Solution

假设图像大小为 (3、28、28)，我们将创建三个独立的配置文件，并为每个配置文件创建一个独立的上下文：

1. 对于低延迟情况： `min=(1, 3, 28, 28), opt=(1, 3, 28, 28), max=(1, 3, 28, 28)`
2.  对于动态分批情况： `min=(1, 3, 28, 28), opt=(4, 3, 28, 28), max=(32, 3, 28, 28)`

    请注意，我们在 opt 中使用的批量大小为 "4"，因为这是最常见的情况。
3. 对于离线情况： `min=(128, 3, 28, 28), opt=(128, 3, 28, 28), max=(128, 3, 28, 28)`

我们将为每个上下文创建一个相应的 `TrtRunner`。如果拥有引擎和上下文（不是通过惰性加载器获得的），那么激活`runner`的成本就会很低，它只需要分配输入和输出缓冲区。

```python
"""
This script builds an engine with 3 separate optimization profiles, each
built for a specific use-case. It then creates 3 separate execution contexts
and corresponding `TrtRunner`s for inference.
"""
import numpy as np
from polygraphy.backend.trt import (
    CreateConfig,
    Profile,
    TrtRunner,
    engine_from_network,
    network_from_onnx_path,
    save_engine,
)
from polygraphy.logger import G_LOGGER


def main():
    # A Profile maps each input tensor to a range of shapes.
    # The `add()` method can be used to add shapes for a single input.
    #
    # TIP: To save lines, calls to `add` can be chained:
    #     profile.add("input0", ...).add("input1", ...)
    #
    #   Of course, you may alternatively write this as:
    #     profile.add("input0", ...)
    #     profile.add("input1", ...)
    #
    profiles = [
        # The low-latency case. For best performance, min == opt == max.
        Profile().add("X", min=(1, 3, 28, 28), opt=(1, 3, 28, 28), max=(1, 3, 28, 28)),
        # The dynamic batching case. We use `4` for the opt batch size since that's our most common case.
        Profile().add("X", min=(1, 3, 28, 28), opt=(4, 3, 28, 28), max=(32, 3, 28, 28)),
        # The offline case. For best performance, min == opt == max.
        Profile().add("X", min=(128, 3, 28, 28), opt=(128, 3, 28, 28), max=(128, 3, 28, 28)),
    ]

    # See examples/api/06_immediate_eval_api for details on immediately evaluated functional loaders like `engine_from_network`.
    # Note that we can freely mix lazy and immediately-evaluated loaders.
    engine = engine_from_network(
        network_from_onnx_path("dynamic_identity.onnx"), config=CreateConfig(profiles=profiles)
    )

    # We'll save the engine so that we can inspect it with `inspect model`.
    # This should make it easy to see how the engine bindings are laid out.
    save_engine(engine, "dynamic_identity.engine")

    # We'll create, but not activate, three separate runners, each with a separate context.
    #
    # TIP: By providing a context directly, as opposed to via a lazy loader,
    # we can ensure that the runner will *not* take ownership of it.
    #
    low_latency = TrtRunner(engine.create_execution_context()) # 此处optimization的默认值是0

    # 注意：以下两行代码可能会导致TensorRT显示错误，因为配置文件0已经被第一个执行上下文所使用。我们将使用G_LOGGER.verbosity()来抑制这些错误信息。
    with G_LOGGER.verbosity(G_LOGGER.CRITICAL):
        # We can use the `optimization_profile` parameter of the runner to ensure that the correct optimization profile is used.
        # This eliminates the need to call `set_profile()` later.
        dynamic_batching = TrtRunner(
            engine.create_execution_context(), optimization_profile=1
        )  # Use the second profile, which is intended for dynamic batching.

        # For the sake of example, we *won't* use `optimization_profile` here.
        # Instead, we'll use `set_profile()` after activating the runner.
        offline = TrtRunner(engine.create_execution_context())

    # Finally, we can activate the runners as we need them.
    #
    # NOTE: Since the context and engine are already created, the runner will only need to
    # allocate input and output buffers during activation.

    input_img = np.ones((1, 3, 28, 28), dtype=np.float32)  # An input "image"

    with low_latency:
        outputs = low_latency.infer({"X": input_img})
        assert np.array_equal(outputs["Y"], input_img)  # It's an identity model!

        print("Low latency runner succeeded!")

        # While we're serving requests online, we might decide that we need dynamic batching
        # for a moment.
        #
        # NOTE: We're assuming that activating runners will be cheap here, so we can bring up
        # the dynamic batching runner just-in-time.
        #
        # TIP: If activating the runner is not cheap (e.g. input/output buffers are large),
        # it might be better to keep the runner active the whole time.
        #
        with dynamic_batching:
            # We'll create fake batches by repeating our fake input image.
            small_input_batch = np.repeat(input_img, 4, axis=0)  # Shape: (4, 3, 28, 28)
            outputs = dynamic_batching.infer({"X": small_input_batch})
            assert np.array_equal(outputs["Y"], small_input_batch)

    # If we need dynamic batching again later, we can activate the runner once more.
    #
    # NOTE: This time, we do *not* need to set the profile.
    #
    with dynamic_batching:
        # NOTE: We can use any shape that's in the range of the profile without
        # additional setup - Polygraphy handles the details behind the scenes!
        #
        large_input_batch = np.repeat(input_img, 16, axis=0)  # Shape: (16, 3, 28, 28)
        outputs = dynamic_batching.infer({"X": large_input_batch})
        assert np.array_equal(outputs["Y"], large_input_batch)

        print("Dynamic batching runner succeeded!")

    with offline:
        #  注意：当我们首次激活这个运行器时，我们需要设置配置文件索引（默认情况下它是0）。
        # 由于我们在创建运行器时提供了我们自己的执行上下文，我们需要*只做一次*这个设置。
        # 我们的设置会持续存在，因为上下文即使在运行器被停用后也会保持活动状态。
        # 如果我们允许运行器拥有上下文，那么每次激活运行器时我们都需要重复这个步骤。
        #
        # Alternatively, we could have used the `optimization_profile` parameter (see above).
        #
        offline.set_profile(2)  # Use the third profile, which is intended for the offline case.

        large_offline_batch = np.repeat(input_img, 128, axis=0)  # Shape: (128, 3, 28, 28)
        outputs = offline.infer({"X": large_offline_batch})
        assert np.array_equal(outputs["Y"], large_offline_batch)

        print("Offline runner succeeded!")


if __name__ == "__main__":
    main()
```

```bash
polygraphy inspect model dynamic_identity.engine
```

* 输出结果示例：

```bash
[I] Loading bytes from /root/fz/Polygraphy/examples/api/07_tensorrt_and_dynamic_shapes/dynamic_identity.engine
[I] ==== TensorRT Engine ====
    Name: Unnamed Network 0 | Explicit Batch Engine
    
    ---- 1 Engine Input(s) ----
    {X [dtype=float32, shape=(-1, 3, 28, 28)]}
    
    ---- 1 Engine Output(s) ----
    {Y [dtype=float32, shape=(1, 3, 28, 28)]}
    
    ---- Memory ----
    Device Memory: 0 bytes
    
    ---- 3 Profile(s) (2 Tensor(s) Each) ----
    - Profile: 0
        Tensor: X          (Input), Index: 0 | Shapes: min=(1, 3, 28, 28), opt=(1, 3, 28, 28), max=(1, 3, 28, 28)
        Tensor: Y         (Output), Index: 1 | Shape: (1, 3, 28, 28)
    
    - Profile: 1
        Tensor: X          (Input), Index: 0 | Shapes: min=(1, 3, 28, 28), opt=(4, 3, 28, 28), max=(32, 3, 28, 28)
        Tensor: Y         (Output), Index: 1 | Shape: (1, 3, 28, 28)
    
    - Profile: 2
        Tensor: X          (Input), Index: 0 | Shapes: min=(128, 3, 28, 28), opt=(128, 3, 28, 28), max=(128, 3, 28, 28)
        Tensor: Y         (Output), Index: 1 | Shape: (1, 3, 28, 28)
    
    ---- 1 Layer(s) Per Profile ----
```

### 保存输入数据&处理运行结果

`Comparator.run` 推理的输入和输出可以序列化并保存为 JSON 文件，以便重复使用。输入存储为 `List[Dict[str,ndarray]]`，而输出则存储在 `RunResults` 对象中，该对象可以存储多个运行器的输出。`Polygraphy` 包含方便的应用程序接口，可以轻松加载和操作这些对象。

先生成`inputs.jsons`和`outputs.json`文件

```bash
polygraphy run identity.onnx --trt --onnxrt \
    --save-inputs inputs.json --save-outputs outputs.json
```

```python
"""
This script demonstrates how to use the `load_json` and `RunResults` APIs to load
and manipulate inference inputs and outputs respectively.
"""

from polygraphy.comparator import RunResults
from polygraphy.json import load_json


def main():
    # Use the `load_json` API to load inputs from file.
    #
    # NOTE: save_json和load_json辅助函数只用于非Polygraphy对象
    # Polygraphy objects that support serialization include `save` and `load` methods.
    inputs = load_json("inputs.json")

    # Inputs are stored as a `List[Dict[str, np.ndarray]]`, i.e. a list of feed_dicts,
    # where each feed_dict maps input names to NumPy arrays.
    #
    # TIP: In the typical case, we'll only have one iteration, so we'll only look at the first item.
    # If you need to access inputs from multiple iterations, you can do something like this instead:
    #
    #    for feed_dict in inputs:
    #        for name, array in feed_dict.items():
    #            ... # Do something with the inputs here
    #
    [feed_dict] = inputs
    for name, array in feed_dict.items():
        print(f"Input: '{name}' | Values:\n{array}")

    # Use the `RunResults.load` API to load results from file.
    #
    # TIP: You can provide either a file path or a file-like object here.
    results = RunResults.load("outputs.json")

    # The `RunResults` object is structured like a `Dict[str, List[IterationResult]]``,
    # mapping runner names to inference outputs from one or more iterations.
    # An `IterationResult` behaves just like a `Dict[str, np.ndarray]` mapping output names
    # to NumPy arrays.
    #
    # TIP: In the typical case, we'll only have one iteration, so we can unpack it
    # directly in the loop. If you need to access outputs from multiple iterations,
    # you can do something like this instead:
    #
    #    for runner_name, iters in results.items():
    #        for outputs in iters:
    #             ... # Do something with the outputs here
    #
    for runner_name, [outputs] in results.items():
        print(f"\nProcessing outputs for runner: {runner_name}")
        # Now you can read or modify the outputs for each runner.
        # For the sake of this example, we'll just print them:
        for name, array in outputs.items():
            print(f"Output: '{name}' | Values:\n{array}")


if __name__ == "__main__":
    main()

```

## CLI

### TensorRT中的int8校准

* 自定义数据集校准

```bash
polygraphy convert identity.onnx --int8 \
    --data-loader-script ./data_loader.py \
    --calibration-cache identity_calib.cache \
    -o identity.engine
```

```python
# data_loader.py
"""
Defines a `load_data` function that returns a generator yielding
feed_dicts so that this script can be used as the argument for
the --data-loader-script command-line parameter.
"""
import numpy as np

INPUT_SHAPE = (1, 1, 2, 2)


def load_data():
    for _ in range(5):
        yield {"x": np.ones(shape=INPUT_SHAPE, dtype=np.float32)}  # Still totally real data
```

还可以使用API中的示例，不过需要指定函数名称：

```bash
polygraphy convert identity.onnx --int8 \
    --data-loader-script ../../../api/04_int8_calibration_in_tensorrt/example.py:calib_data \
    -o identity.engine
```

* 使用缓存校准

```bash
polygraphy convert identity.onnx --int8 \
    --calibration-cache identity_calib.cache \
    -o identity.engine
```

### 在TensorRT中构建确定性引擎(engine)

在引擎构建过程中，TensorRT 会运行多个核函数并进行计时，以选出最优的核函数。由于每次计时可能会略有不同，因此这一过程本质上是非确定性的。

在许多情况下，我们可能需要确定性的引擎构建。实现这一点的一种方法是使用 `IAlgorithmSelector` API 来确保每次都选择相同的核函数。

为了简化这一过程，Polygraphy 提供了两个内置算法选择器：`TacticRecorder`和`TacticReplayer，`与之对应的是 `--save-tactics` 和 -`-load-tactics` 选项。

#### Running The Example

1.  创建引擎并保存`replay`文件：

    ```
     polygraphy convert identity.onnx \
         --save-tactics replay.json \
         -o 0.engine
    ```

    生成的 `replay.json` 文件是可读的。我们可以选择使用`inspect tactics`查看它：

    ```
     polygraphy inspect tactics replay.json
    ```
2.  将`replay`文件用于另一个引擎构建：

    ```
     polygraphy convert identity.onnx \
         --load-tactics replay.json \
         -o 1.engine
    ```
3. 验证两个`engine`是否完全相同：

```bash
diff -sa 0.engine 1.engine
```

{% hint style="info" %}
以下是一些 `diff` 命令的常见示例：

1.  **比较两个文件并显示差异：**

    ```
    diff file1.txt file2.txt
    ```

    这将比较 `file1.txt` 和 `file2.txt`，并显示它们之间的差异。
2.  **显示所有差异的详细信息：**

    ```
    diff -u file1.txt file2.txt
    ```

    使用 `-u` 选项将以 Unified diff 格式显示所有差异的详细信息。
3.  **递归比较两个目录并显示差异：**

    ```
    diff -r directory1 directory2
    ```

    这将递归地比较 `directory1` 和 `directory2` 中的文件，并显示它们之间的差异。
4.  **将差异输出到文件中：**

    ```
    diff file1.txt file2.txt > diff_output.txt
    ```

    这将比较 `file1.txt` 和 `file2.txt`，并将差异输出到 `diff_output.txt` 文件中。
5.  **忽略空白字符的差异：**

    ```
    diff -b file1.txt file2.txt
    ```

    使用 `-b` 选项可以忽略空白字符的差异。
6.  **显示所有差异并标出行号：**

    ```
    diff -u -N file1.txt file2.txt
    ```

    使用 `-N` 选项将显示所有差异并在每行前面标出行号。
7.  **只显示不同之处的行：**

    ```
    diff -q file1.txt file2.txt
    ```

    使用 `-q` 选项将只显示不同之处的行，而不会显示详细的差异信息。
{% endhint %}

### 动态shape

1. 使用三个独立的`profile`创建`engine`

```bash
polygraphy convert dynamic_identity.onnx -o dynamic_identity.engine \
    --trt-min-shapes X:[1,3,28,28] --trt-opt-shapes X:[1,3,28,28] --trt-max-shapes X:[1,3,28,28] \
    --trt-min-shapes X:[1,3,28,28] --trt-opt-shapes X:[4,3,28,28] --trt-max-shapes X:[32,3,28,28] \
    --trt-min-shapes X:[128,3,28,28] --trt-opt-shapes X:[128,3,28,28] --trt-max-shapes X:[128,3,28,28]
```

对于有多个输入的模型，只需为每个 `--trt-*-shapes` 参数提供多个输入即可。

`--trt-min-shapes input0:[10,10] input1:[10,10] input2:[10,10] ...`

{% hint style="info" %}
如果`min == opt == max`，可以直接使用`--input-shapes`_`，`_而不用单独设置`min/opt/max`
{% endhint %}

```bash
polygraphy inspect model dynamic_identity.engine
```

* 输出结果示例：

```bash
[I] Loading bytes from /root/fz/Polygraphy/examples/cli/convert/03_dynamic_shapes_in_tensorrt/dynamic_identity.engine
[I] ==== TensorRT Engine ====
    Name: Unnamed Network 0 | Explicit Batch Engine
    
    ---- 1 Engine Input(s) ----
    {X [dtype=float32, shape=(-1, 3, 28, 28)]}
    
    ---- 1 Engine Output(s) ----
    {Y [dtype=float32, shape=(1, 3, 28, 28)]}
    
    ---- Memory ----
    Device Memory: 0 bytes
    
    ---- 3 Profile(s) (2 Tensor(s) Each) ----
    - Profile: 0
        Tensor: X          (Input), Index: 0 | Shapes: min=(1, 3, 28, 28), opt=(1, 3, 28, 28), max=(1, 3, 28, 28)
        Tensor: Y         (Output), Index: 1 | Shape: (1, 3, 28, 28)
    
    - Profile: 1
        Tensor: X          (Input), Index: 0 | Shapes: min=(1, 3, 28, 28), opt=(4, 3, 28, 28), max=(32, 3, 28, 28)
        Tensor: Y         (Output), Index: 1 | Shape: (1, 3, 28, 28)
    
    - Profile: 2
        Tensor: X          (Input), Index: 0 | Shapes: min=(128, 3, 28, 28), opt=(128, 3, 28, 28), max=(128, 3, 28, 28)
        Tensor: Y         (Output), Index: 1 | Shape: (1, 3, 28, 28)
    
    ---- 1 Layer(s) Per Profile ----
```

### 将onnx转换为FP16格式&分析精度损失情况

1. 将模型转换为FP16格式

```bash
polygraphy convert --fp-to-fp16 -o identity_fp16.onnx identity.onnx
```

2. 查看转换后的模型文件

```bash
polygraphy inspect model identity_fp16.onnx
```

* 输出结果示例：

```bash
[I] Loading model: /root/fz/Polygraphy/examples/cli/convert/04_converting_models_to_fp16/identity_fp16.onnx
[I] ==== ONNX Model ====
    Name: test_identity | ONNX Opset: 8
    
    ---- 1 Graph Input(s) ----
    {x [dtype=float32, shape=(1, 1, 2, 2)]}
    
    ---- 1 Graph Output(s) ----
    {y [dtype=float32, shape=(1, 1, 2, 2)]}
    
    ---- 0 Initializer(s) ----
    
    ---- 3 Node(s) ----
```

3. 在 `ONNX-Runtime` 下运行 `FP32` 和 `FP16` 模型，然后比较结果：

```bash
polygraphy run --onnxrt identity.onnx \
   --save-inputs inputs.json --save-outputs outputs_fp32.json
```

```bash
polygraphy run --onnxrt identity_fp16.onnx \
   --load-inputs inputs.json --load-outputs outputs_fp32.json \
   --atol 0.001 --rtol 0.001
```

* 输出结果示例：

```bash
[I] RUNNING | Command: /root/miniconda3/bin/polygraphy run --onnxrt identity_fp16.onnx --load-inputs inputs.json --load-outputs outputs_fp32.json --atol 0.001 --rtol 0.001
[I] Loading input data from inputs.json
[I] onnxrt-runner-N0-04/15/24-09:31:49  | Activating and starting inference
[I] Creating ONNX-Runtime Inference Session with providers: ['CPUExecutionProvider']
[I] onnxrt-runner-N0-04/15/24-09:31:49 
    ---- Inference Input(s) ----
    {x [dtype=float32, shape=(1, 1, 2, 2)]}
[I] onnxrt-runner-N0-04/15/24-09:31:49 
    ---- Inference Output(s) ----
    {y [dtype=float32, shape=(1, 1, 2, 2)]}
[I] onnxrt-runner-N0-04/15/24-09:31:49  | Completed 1 iteration(s) in 0.08607 ms | Average inference time: 0.08607 ms.
[I] Loading inference results from outputs_fp32.json
[I] Accuracy Comparison | onnxrt-runner-N0-04/15/24-09:31:49 vs. onnxrt-runner-N0-04/15/24-09:28:14
[I]     Comparing Output: 'y' (dtype=float32, shape=(1, 1, 2, 2)) with 'y' (dtype=float32, shape=(1, 1, 2, 2))
[I]         Tolerance: [abs=0.001, rel=0.001] | Checking elemwise error
[I]         onnxrt-runner-N0-04/15/24-09:31:49: y | Stats: mean=0.35989, std-dev=0.25781, var=0.066464, median=0.35962, min=0.00011438 at (0, 0, 1, 0), max=0.72021 at (0, 0, 0, 1), avg-magnitude=0.35989
[I]         onnxrt-runner-N0-04/15/24-09:28:14: y | Stats: mean=0.35995, std-dev=0.25784, var=0.066482, median=0.35968, min=0.00011437 at (0, 0, 1, 0), max=0.72032 at (0, 0, 0, 1), avg-magnitude=0.35995
[I]         Error Metrics: y
[I]             Minimum Required Tolerance: elemwise error | [abs=0.00010967] OR [rel=0.00028606] (requirements may be lower if both abs/rel tolerances are set)
[I]             Absolute Difference | Stats: mean=5.6492e-05, std-dev=4.3677e-05, var=1.9077e-09, median=5.8144e-05, min=6.4974e-09 at (0, 0, 1, 0), max=0.00010967 at (0, 0, 0, 1), avg-magnitude=5.6492e-05
[I]             Relative Difference | Stats: mean=0.00014165, std-dev=9.0956e-05, var=8.273e-09, median=0.00011186, min=5.6808e-05 at (0, 0, 1, 0), max=0.00028606 at (0, 0, 1, 1), avg-magnitude=0.00014165
[I]         PASSED | Output: 'y' | Difference is within tolerance (rel=0.001, abs=0.001)
[I]     PASSED | All outputs matched | Outputs: ['y']
[I] Accuracy Summary | onnxrt-runner-N0-04/15/24-09:31:49 vs. onnxrt-runner-N0-04/15/24-09:28:14 | Passed: 1/1 iterations | Pass Rate: 100.0%
[I] PASSED | Runtime: 1.605s | Command: /root/miniconda3/bin/polygraphy run --onnxrt identity_fp16.onnx --load-inputs inputs.json --load-outputs outputs_fp32.json --atol 0.001 --rtol 0.001
```

{% hint style="info" %}
`--atol 0.001` 和 `--rtol 0.001`: 这两个参数分别设置了绝对误差容限和相对误差容限。在模型输出和参考输出之间的比较中，这些参数用于确定何时认为两个值相等。在这种情况下，设置为 0.001 意味着只有当两个值之间的差异小于或等于 0.001 时，才会认为它们是相等的。
{% endhint %}

4. 检查 `FP16` 模型的中间输出结果是否存在 `NaN` 或无穷大

```bash
polygraphy run --onnxrt identity_fp16.onnx --validate
```

### 调试TensorRT策略(`tactics`)

由于 TensorRT 生成器依赖于定时策略(timing tactics)，因此引擎生成是非确定性的。

解决这一问题的方法之一是多次运行生成器，保存每次运行的`tatic`文件，可以对它们进行比较，以确定哪种`tactic`可能是错误的根源。`debug build`命令自动执行上述过程。

有关调试工具如何工作的详细信息，请参阅帮助： `polygraphy debug -h` 和 `polygraphy debug build -h`。

1. 从 `ONNX-Runtime` 生成输出：

```bash
polygraphy run identity.onnx --onnxrt \
    --save-outputs golden.json
```

使用`debug build`重复构建 TensorRT 引擎，并将结果与输出进行比较，每次都保存一个`tactics`文件：

```bash
polygraphy debug build identity.onnx --fp16 --save-tactics replay.json \
    --artifacts-dir replays --artifacts replay.json --until=10 \
    --check polygraphy run polygraphy_debug.engine --trt --load-outputs golden.json
```

* 提供 --until 选项，以便工具知道何时停止，这可以是迭代次数，也可以是`good`或 `bad`。在后一种情况下，工具将分别在找到第一个通过或失败的迭代后停止。
* 我们指定 `--save-tactics` 来保存每次迭代的重放文件，然后使用 `--artifacts` 来告诉`debug build`来管理它们，这包括将它们分类到用 `--artifacts-dir` 指定的目录下的`good`和`bad`子目录中。

3. 使用`inspect diff-tactics`来确定哪些`tactics`可能不好：

```bash
polygraphy inspect diff-tactics --dir replays
```
