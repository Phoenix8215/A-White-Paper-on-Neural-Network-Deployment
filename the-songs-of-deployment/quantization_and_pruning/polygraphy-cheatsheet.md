# ✌️ Polygraphy-Cheatsheet

### Polygraphy介绍

polygraphy 是一个深度学习模型调试工具，包含 python API 和 命令行工具 ，它有的一些功能如下：

* 使用多种后端运行推理计算，包括 TensorRT, onnxruntime, TensorFlow；
* 比较不同后端的逐层计算结果；
* 由模型搭建生成 TensorRT 引擎并序列化为.plan；
* 查看模型网络的逐层信息；
* 修改 onnx 模型，如提取子图，计算图化简；
* 分析 onnx 转 TensorRT 失败原因，将原计算图中可以 / 不可以转 TensorRT 的子图分割保存；
* 隔离 TensorRT 终端错误的tactic；

### 安装Polygraphy

```bash
python -m pip install colored polygraphy --extra-index-url https://pypi.ngc.nvidia.com
```

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

* 输出结果示例

```bash
[I] Configuring with profiles:[
        Profile 0:
            {x [min=[1, 1, 2, 2], opt=[1, 1, 2, 2], max=[1, 1, 2, 2]]}
    ]
[I] Building engine with configuration:
    Flags                  | [FP16]
    Engine Capability      | EngineCapability.DEFAULT
    Memory Pools           | [WORKSPACE: 24217.31 MiB, TACTIC_DRAM: 24217.31 MiB]
    Tactic Sources         | [CUBLAS, CUBLAS_LT, CUDNN, EDGE_MASK_CONVOLUTIONS, JIT_CONVOLUTIONS]
    Profiling Verbosity    | ProfilingVerbosity.DETAILED
    Preview Features       | [FASTER_DYNAMIC_SHAPES_0805, DISABLE_EXTERNAL_TACTIC_SOURCES_FOR_CORE_0805]
[I] Finished engine building in 0.259 seconds
[I] Saving engine to identity.engine
Inference succeeded!
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

* 输出结果示例

```bash
[I] Configuring with profiles:[
        Profile 0:
            {x [min=[1, 1, 2, 2], opt=[1, 1, 2, 2], max=[1, 1, 2, 2]]}
    ]
[I] Building engine with configuration:
    Flags                  | []
    Engine Capability      | EngineCapability.DEFAULT
    Memory Pools           | [WORKSPACE: 24217.31 MiB, TACTIC_DRAM: 24217.31 MiB]
    Tactic Sources         | [CUBLAS, CUBLAS_LT, CUDNN, EDGE_MASK_CONVOLUTIONS, JIT_CONVOLUTIONS]
    Profiling Verbosity    | ProfilingVerbosity.DETAILED
    Preview Features       | [FASTER_DYNAMIC_SHAPES_0805, DISABLE_EXTERNAL_TACTIC_SOURCES_FOR_CORE_0805]
[I] Finished engine building in 0.419 seconds
Validation succeeded!
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

* 输出结果示例

```bash
Network name: MyIdentity
[I] Configuring with profiles:[
        Profile 0:
            {x [min=[1, 1, 2, 2], opt=[1, 1, 2, 2], max=[1, 1, 2, 2]]}
    ]
[I] Building engine with configuration:
    Flags                  | [FP16]
    Engine Capability      | EngineCapability.DEFAULT
    Memory Pools           | [WORKSPACE: 24217.31 MiB, TACTIC_DRAM: 24217.31 MiB]
    Tactic Sources         | [CUBLAS, CUBLAS_LT, CUDNN, EDGE_MASK_CONVOLUTIONS, JIT_CONVOLUTIONS]
    Profiling Verbosity    | ProfilingVerbosity.DETAILED
    Preview Features       | [FASTER_DYNAMIC_SHAPES_0805, DISABLE_EXTERNAL_TACTIC_SOURCES_FOR_CORE_0805]
[I] Finished engine building in 0.428 seconds
Inference succeeded!
```

### ##&#x20;
