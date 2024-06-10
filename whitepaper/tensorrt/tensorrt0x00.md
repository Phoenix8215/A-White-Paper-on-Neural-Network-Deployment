# 🐯 手撕TensoRT源码|0x00

### coding

[https://github.com/Phoenix-LegendX/learn-TensorRT-from-scratch](https://github.com/Phoenix-LegendX/learn-TensorRT-from-scratch)

### nvinfer1::INocopy

```c
class INoCopy
{
protected:
    INoCopy() = default;
    virtual ~INoCopy() = default;
    INoCopy(INoCopy const& other) = delete;
    INoCopy& operator=(INoCopy const& other) = delete;
    INoCopy(INoCopy&& other) = delete;
    INoCopy& operator=(INoCopy&& other) = delete;
};
```

由 TensorRT 库实现的所有 TensorRT 接口的基类,这些类的对象不可移动或复制，只能通过指针进行操作。

<figure><img src="https://m7jsqg9xk0.feishu.cn/space/api/box/stream/download/asynccode/?code=MjRiMGM2NjVjYTUwMWJmZjBiMjdmNDZmYjZjZTRhZDZfcmJJVnZTZW8xWTkwRmZWRTRIblBFajVjbjdjVjViZG5fVG9rZW46REZPV2JhdkpIbzJvZlJ4RUlhaWNDNmt0bkpjXzE3MTgwMzg4OTI6MTcxODA0MjQ5Ml9WNA" alt=""><figcaption></figcaption></figure>

### nvinfer1::ITensor

`nvinfer1::ITensor`类的公共成员方法如下：`ITensor`类提供了一系列方法用于设置和获取张量的属性，以及执行一些操作。以下是每个成员方法的作用和返回值类型：

<figure><img src="https://m7jsqg9xk0.feishu.cn/space/api/box/stream/download/asynccode/?code=ZTIxMDY0ZWJmZjljMjQ4ZmU3MDliYzNjY2NiYWVmNDJfQmtXWTVYMnVVS3ZKMm9nd21OeXNkRXdva3NXWXRLYnlfVG9rZW46VGtGNmJvb3Jrb29Oakx4a1Y0bGNHbGRkbkZmXzE3MTgwMzg4OTI6MTcxODA0MjQ5Ml9WNA" alt="" width="188"><figcaption></figcaption></figure>

1. `setName(char const* name) noexcept`: 设置张量的名称。参数`name`是一个指向以null结尾的C风格字符串，表示张量的名称。该方法没有返回值。

\


2. `getName() const noexcept`: 获取张量的名称。返回一个指向以null结尾的C风格字符串的指针，表示张量的名称。

\


3. `setDimensions(Dims dimensions) noexcept`: 设置张量的维度。参数`dimensions`是一个`Dims`结构，表示张量的维度信息。该方法没有返回值。

\


4. `getDimensions() const noexcept`: 获取张量的维度。返回一个`Dims`结构，表示张量的维度信息。

\


5. `setType(DataType type) noexcept`: 设置张量的数据类型。参数`type`是一个`DataType`枚举值，表示张量的数据类型。该方法没有返回值。

\


6. `getType() const noexcept`: 获取张量的数据类型。返回一个`DataType`枚举值，表示张量的数据类型。

\


7. `setDynamicRange(float min, float max) noexcept`: 设置张量的动态范围。参数`min`和`max`分别表示动态范围的最小值和最大值。返回一个`bool`值，表示动态范围是否设置成功。

\


8. `isNetworkInput() const noexcept`: 检查张量是否是网络的输入。返回一个`bool`值，表示张量是否是网络的输入。

\


9. `isNetworkOutput() const noexcept`: 检查张量是否是网络的输出。返回一个`bool`值，表示张量是否是网络的输出。

\


10. `setBroadcastAcrossBatch(bool broadcastAcrossBatch) noexcept`: 设置是否在批处理中广播张量。参数`broadcastAcrossBatch`是一个`bool`值，表示是否在批处理中广播张量。该方法没有返回值。

\


11. `getBroadcastAcrossBatch() const noexcept`: 检查张量是否在批处理中广播。返回一个`bool`值，表示张量是否在批处理中广播。

\


12. `getLocation() const noexcept`: 获取张量的存储位置。返回一个`TensorLocation`枚举值，表示张量的存储位置。

\


13. `setLocation(TensorLocation location) noexcept`: 设置张量的存储位置。参数`location`是一个`TensorLocation`枚举值，表示张量的存储位置。该方法没有返回值。

\


14. `dynamicRangeIsSet() const noexcept`: 查询张量是否设置了动态范围。返回一个`bool`值，表示张量是否设置了动态范围。

\


15. `resetDynamicRange() noexcept`: 取消设置张量的动态范围。该方法没有返回值。

\


16. `getDynamicRangeMin() const noexcept`: 获取张量的动态范围的最小值。返回一个`float`值，表示张量的动态范围的最小值。

\


17. `getDynamicRangeMax() const noexcept`: 获取张量的动态范围的最大值。返回一个`float`值，表示张量的动态范围的最大值。

\


18. `setAllowedFormats(TensorFormats formats) noexcept`: 设置张量允许的格式。参数`formats`是一个`TensorFormats`枚举值，表示张量允许的格式。该方法没有返回值。

\


19. `getAllowedFormats() const noexcept`: 获取张量允许的格式。返回一个`TensorFormats`枚举值，表示张量允许的格式。

\


20. `isShapeTensor() const noexcept`: 检查张量是否是形状张量。返回一个`bool`值，表示张量是否是形状张量。

\


21. `isExecutionTensor() const noexcept`: 检查张量是否是执行张量。返回一个`bool`值，表示张量是否是执行张量。

\


22. `setDimensionName(int32_t index, char const* name) noexcept`: 为输入张量的维度命名。参数`index`表示维度的索引，参数`name`是一个指向以null结尾的C风格字符串，表示维度的名称。该方法没有返回值。

\


23. `getDimensionName(int32_t index) const noexcept`: 获取输入张量的维度名称。参数`index`表示维度的索引。返回一个指向以null结尾的C风格字符串的指针，表示维度的名称。

\


### `nvinfer1::ILayer`

这个类是TensorRT中表示网络中各层的基类。\


<figure><img src="https://m7jsqg9xk0.feishu.cn/space/api/box/stream/download/asynccode/?code=ZTNhMDJlNWJhZjliZGRmYjQ4NDcyYTQxYTQyZjQ0YWVfZHRVSWdBQ1hzOVVyUnduN3dRVExGenBPcGM4bUdBVWdfVG9rZW46QWgySmI5UGZibzhXalN4RGlwVGNvaFo1bk9nXzE3MTgwMzg4OTI6MTcxODA0MjQ5Ml9WNA" alt=""><figcaption></figcaption></figure>

1. `getType()`: 返回当前层的类型，即 `LayerType` 枚举类型。

\


2. `setName(char const* name)`: 设置当前层的名称。

\


3. `getName()`: 获取当前层的名称。

\


4. `getNbInputs()`: 获取当前层的输入数量。

\


5. `getInput(int32_t index)`: 获取当前层指定索引的输入张量，返回值是 `ITensor*` 类型指针。

\


6. `getNbOutputs()`: 获取当前层的输出数量。

\


7. `getOutput(int32_t index)`: 获取当前层指定索引的输出张量，返回值是 `ITensor*` 类型指针。

\


8. `setInput(int32_t index, ITensor& tensor)`: 将当前层指定索引的输入张量替换为指定的张量。

\


9. `setPrecision(DataType dataType)`: 设置当前层的计算精度，参数是 `DataType` 枚举类型。

\


10. `getPrecision()`: 获取当前层的计算精度，返回值是 `DataType` 枚举类型。

\


11. `precisionIsSet()`: 检查当前层的计算精度是否已设置，返回值是 `bool` 类型。

\


12. `resetPrecision()`: 重置当前层的计算精度。

\


13. `setOutputType(int32_t index, DataType dataType)`: 设置当前层指定索引的输出类型，参数是 `DataType` 枚举类型。

\


14. `getOutputType(int32_t index)`: 获取当前层指定索引的输出类型，返回值是 `DataType` 枚举类型。

\


15. `outputTypeIsSet(int32_t index)`: 检查当前层指定索引的输出类型是否已设置，返回值是 `bool` 类型。

\


16. `resetOutputType(int32_t index)`: 重置当前层指定索引的输出类型。

\
\
\


### **nvinfer1::Dims32**

\


<figure><img src="https://m7jsqg9xk0.feishu.cn/space/api/box/stream/download/asynccode/?code=YjhkZWQxZTgxMGYwM2Q0NTJjY2NhYzA2ZDRkZTRhOGZfdTJFSEVGbjExRkw0Vm15SWZjMUpMdG1WcTE5cGl1OFFfVG9rZW46TWNidGJyQmt2b3BzQmh4WEV3eGNDUFhHbjNkXzE3MTgwMzg4OTI6MTcxODA0MjQ5Ml9WNA" alt=""><figcaption></figcaption></figure>

```c
//!
//! \class Dims
//! \brief Structure to define the dimensions of a tensor.
//!
//! TensorRT can also return an invalid dims structure. This structure is represented by nbDims == -1
//! and d[i] == 0 for all d.
//!
//! TensorRT can also return an "unknown rank" dims structure. This structure is represented by nbDims == -1
//! and d[i] == -1 for all d.
//!
class Dims32
{
public:
    //! The maximum rank (number of dimensions) supported for a tensor.
    static constexpr int32_t MAX_DIMS{8};
    //! The rank (number of dimensions).
    int32_t nbDims;
    //! The extent of each dimension.
    int32_t d[MAX_DIMS];
};

//!
//! Alias for Dims32.
//!
//! \warning: This alias might change in the future.
//!
using Dims = Dims32;
```

`Dims32` 是一个用于定义张量维度的结构体类。在 TensorRT 中，它用于描述张量的形状。

1. **MAX\_DIMS**:
   * **描述**: 这是一个静态常量，表示张量支持的最大维度数。这个值是8，意味着这个结构体最多可以表示8维的张量。
   * **类型**: `int32_t`
   * **作用**: 确定张量的最大维度数。

\


2. **nbDims**:
   * **描述**: 这是一个整数，表示张量的维度数（也称为秩）。
   * **类型**: `int32_t`
   * **作用**: 存储张量的实际维度数。如果 `nbDims` 为 -1，表示这个结构体是无效的维度结构。

\


3. **d\[MAX\_DIMS]**:
   * **描述**: 这是一个数组，存储每个维度的大小（extent）。
   * **类型**: `int32_t[MAX_DIMS]`
   * **作用**: 保存张量每个维度的大小信息。如果 `nbDims` 为 -1 并且所有 `d[i]` 为 0，表示这是一个无效的维度结构。如果 `nbDims` 为 -1 并且所有 `d[i]` 为 -1，表示这是一个 "unknown rank" 的维度结构。

\


### nvinfer1::Weights

```c
//!
//! \class Weights
//!
//! \brief An array of weights used as a layer parameter.
//!
//! When using the DLA, the cumulative size of all Weights used in a network
//! must be less than 512MB in size. If the build option kGPU_FALLBACK is specified,
//! then multiple DLA sub-networks may be generated from the single original network.
//!
//! The weights are held by reference until the engine has been built. Therefore the data referenced
//! by \p values field should be preserved until the build is complete.
//!
//! The term "empty weights" refers to Weights with weight coefficients ( \p count == 0 and \p values == nullptr).
//!
class Weights
{
public:
    DataType type;      //!< The type of the weights.
    void const* values; //!< The weight values, in a contiguous array.
    int64_t count;      //!< The number of weights in the array.
};
```

这个 `Weights` 类表示用于层参数的权重数组，权重是用于训练和推理的核心参数。

1. **DataType type**：
   * **描述**: 权重的数据类型。
   * **类型**: `DataType` 是一个枚举类型，表示权重数据的类型，如 `kFLOAT`, `kHALF`, `kINT8` 等。
   * **作用**: 确定权重的存储和计算类型。

\


2. **void const\* values**：
   * **描述**: 指向权重值的指针。
   * **类型**: `void const*` 表示一个指向常量数据的指针，数据类型不确定。
   * **作用**: 存储权重值的实际数据，数据是连续的数组。

\


3. **int64\_t count**：
   * **描述**: 权重数组中的权重数量。
   * **类型**: `int64_t`
   * **作用**: 指示权重数组的长度，即有多少个权重值。

\


* **DLA 使用限制**：当使用 DLA（深度学习加速器）时，网络中所有权重的累积大小必须小于 512MB。 确保在构建网络时遵循硬件限制，避免超过 DLA 的内存限制。

\


* **权重引用**：在引擎构建完成之前，权重通过引用保持。因此，在构建完成之前，应保留 `values` 字段引用的数据。确保在引擎构建期间，权重数据不会被意外修改或释放。

\


* **空权重**："空权重" 指的是 `count == 0` 且 `values == nullptr` 的权重。提供一种表示没有权重的方式，可能用于占位符或初始化状态。

\


### nvinfer1::IBuilder

\
`IBuilder` 类用于根据网络定义构建 TensorRT 引擎。该类包含多个方法，用于设置和获取构建过程中的各种配置和属性。\


<figure><img src="https://m7jsqg9xk0.feishu.cn/space/api/box/stream/download/asynccode/?code=ZTA2MzI1NGI1YmE3MTJjNTcwOGM2YTg3NzAyOTkzMGJfcTZ2dmcxWXJ0d25UaFZSSGtjS3dsb0hWU2hZRnZ0ZlZfVG9rZW46RUxqWmJhcTBhb3djYWx4a01KaWN6NHJtbjhiXzE3MTgwMzg4OTI6MTcxODA0MjQ5Ml9WNA" alt=""><figcaption></figcaption></figure>

* `void setMaxBatchSize(int32_t batchSize) noexcept`
  * 设置最大批次大小。在 TensorRT 8.4 中已弃用。
  * 参数：`batchSize` 是最大批次大小。
  * \

* `int32_t getMaxBatchSize() const noexcept`
  * 获取最大批次大小。在 TensorRT 8.4 中已弃用。
  * 返回值：最大批次大小。

\


* `bool platformHasFastFp16() const noexcept`
  * 检测平台是否支持快速原生 FP16 计算。

\


* `bool platformHasFastInt8() const noexcept`
  * 检测平台是否支持快速原生 INT8 计算。

\


* `int32_t getMaxDLABatchSize() const noexcept`
  * 获取 DLA 能够支持的最大批次大小。

\


* `int32_t getNbDLACores() const noexcept`
  * 返回可用的 DLA 引擎数量。
  * 返回值：DLA 引擎数量。

\


* `void setGpuAllocator(IGpuAllocator* allocator) noexcept`
  * 设置用于构建器的 GPU 分配器。如果传递 nullptr，则使用默认分配器。
  * 参数：`allocator` 是分配器对象。

\


* `nvinfer1::IBuilderConfig* createBuilderConfig() noexcept`
  * 创建构建器配置对象。
  * 返回值：构建器配置对象。

\


* `nvinfer1::INetworkDefinition* createNetworkV2(NetworkDefinitionCreationFlags flags) noexcept`
  * 创建网络定义对象，支持动态形状和显式批次维度。
  * 参数：`flags` 是指定网络属性的标志位。
  * 返回值：网络定义对象。

\


* `nvinfer1::IOptimizationProfile* createOptimizationProfile() noexcept`
  * 创建新的优化配置文件。
  * 返回值：优化配置文件对象。

\


* `void reset() noexcept`
  * 重置构建器状态为默认值。

\


* `bool platformHasTf32() const noexcept`
  * 检测平台是否支持 TF32。

\


* `nvinfer1::IHostMemory* buildSerializedNetwork(INetworkDefinition& network, IBuilderConfig& config) noexcept`
  * 构建并序列化网络，而无需创建引擎。
  * 参数：`network` 是网络定义，`config` 是构建器配置。
  * 返回值：包含序列化网络的主机内存对象。

\


* `ILogger* getLogger() const noexcept`
  * 获取用于创建构建器的日志记录器。
  * 返回值：日志记录器对象。

\


* `bool setMaxThreads(int32_t maxThreads) noexcept`
  * 设置构建器可以使用的最大线程数。
  * 参数：`maxThreads` 是最大线程数。

\


* `int32_t getMaxThreads() const noexcept`
  * 获取构建器可以使用的最大线程数。

\


### `nvinfer1::IOptimizationProfile`

`IOptimizationProfile` 类用于为具有动态输入维度和形状张量的网络定义优化配置文件。当从具有动态可调整大小输入的网络定义（至少一个输入张量的一个或多个维度指定为 -1）构建 ICudaEngine 时，用户需要指定至少一个优化配置文件。优化配置文件的索引从 0 开始。第一个定义的优化配置文件（索引为 0）将在未显式选择优化配置文件时由 ICudaEngine 使用。如果没有动态输入，除非用户显式提供，否则默认优化配置文件将自动生成。\


<figure><img src="https://m7jsqg9xk0.feishu.cn/space/api/box/stream/download/asynccode/?code=OGY4ZmIxM2MzMzAzMDMwNWRhMmEwOGE3MGVjY2RkMjlfWEtzR0VDdXpkSERMY3c4WW9ScnRQQWhCc1FHNGFIS1JfVG9rZW46S1FOTmJyWWZLb3FleHh4TXRHeWNqWUE5bmdiXzE3MTgwMzg4OTI6MTcxODA0MjQ5Ml9WNA" alt=""><figcaption></figcaption></figure>

1. **`bool setDimensions(char const* inputName, OptProfileSelector select, Dims dims) noexcept`**
   1. 设置动态输入张量的最小、最佳和最大维度。
   2. 该函数必须针对具有动态维度的每个网络输入张量调用三次（分别用于最小、最佳和最大维度）。
   3. **参数：**
      * `inputName`: 输入张量名称。
      * `select`: 设置最小、最佳或最大维度。
      * `dims`: 最小、最佳或最大维度。
   4. **返回值：**
      * `true` 如果没有检测到不一致。
      * `false` 如果检测到不一致（例如维度不匹配）。

\


2. **`Dims getDimensions(char const* inputName, OptProfileSelector select) const noexcept`**
   1. 获取动态输入张量的最小、最佳或最大维度。
   2. 如果维度尚未通过 `setDimensions()` 设置，则返回 `nbDims == -1` 的无效 `Dims`。
   3. **参数：**
      * `inputName`: 输入张量名称。
      * `select`: 获取最小、最佳或最大维度。
   4. **返回值：**
      * 返回指定的维度。

\


3. **`bool setShapeValues(char const* inputName, OptProfileSelector select, int32_t const* values, int32_t nbValues) noexcept`**
   1. 设置输入形状张量的最小、最佳和最大值。
   2. 该函数必须针对每个形状张量调用三次。
   3. **参数：**
      * `inputName`: 输入张量名称。
      * `select`: 设置最小、最佳或最大值。
      * `values`: 最小、最佳或最大形状张量元素的数组。
      * `nbValues`: 值数组的长度，必须等于形状张量元素的数量（≥ 1）。
   4. **返回值：**
      * `true` 如果没有检测到不一致。
      * `false` 如果检测到不一致（例如 `nbValues` 与之前的调用不匹配）。

\


4. **`int32_t getNbShapeValues(char const* inputName) const noexcept`**
   1. 获取输入形状张量的值数量。
   2. 如果之前未通过 `setShapeValues()` 设置，则返回 `-1`。
   3. **参数：**
      * `inputName`: 输入张量名称。
   4. **返回值：**
      * 返回形状值的数量。

\


5. **`int32_t const* getShapeValues(char const* inputName, OptProfileSelector select) const noexcept`**
   1. 获取输入形状张量的最小、最佳或最大值。
   2. 如果之前未通过 `setShapeValues()` 设置，则返回 `nullptr`。
   3. **参数：**
      * `inputName`: 输入张量名称。
      * `select`: 获取最小、最佳或最大值。
   4. **返回值：**
      * 返回指定的形状值。

\


6. **`bool isValid() const noexcept`**
   1. 检查优化配置文件是否可以传递给 IBuilderConfig 对象。
   2. 执行部分验证，例如检查最小、最佳和最大维度是否已设置且具有相同的维度数，以及检查最佳维度始终不小于最小维度，最大维度不小于最佳维度。
   3. **返回值：**
      * `true` 如果优化配置文件有效。
      * `false` 如果优化配置文件无效。

\


### `nvinfer1::IBuilderConfig`

`IBuilderConfig` 类用于配置构建器，以便生成 TensorRT 引擎\


<figure><img src="https://m7jsqg9xk0.feishu.cn/space/api/box/stream/download/asynccode/?code=N2NjMjQ4ZDdhNjQ4ZjU3NmY5NmRhMzIxOTIyY2MxZWJfdmtVTEp5N2xodWRlY0M3bFI0OEd3aXdvcms5UFExYUpfVG9rZW46QU1Zc2JRREZmb3lkTVB4eE93eGNoQ2lQbmhoXzE3MTgwMzg4OTI6MTcxODA0MjQ5Ml9WNA" alt=""><figcaption></figcaption></figure>

**定时迭代相关方法**

\


* `void setAvgTimingIterations(int32_t avgTiming) noexcept`
  * 设置在定时层时用于平均的迭代次数。
  * 参数：`avgTiming` 是平均迭代次数。

\


* `int32_t getAvgTimingIterations() const noexcept`
  * 查询平均迭代次数。
  * 返回值：平均迭代次数。

\


**INT8 校准相关方法**

\


* `void setInt8Calibrator(IInt8Calibrator* calibrator) noexcept`
  * 设置 INT8 校准接口，用于最小化 INT8 量化过程中的信息丢失。
  * 参数：`calibrator` 是校准器对象。

\


* `IInt8Calibrator* getInt8Calibrator() const noexcept`
  * 获取 INT8 校准接口。
  * 返回值：校准器对象。

\


**最大工作空间相关方法**

\


* `void setMaxWorkspaceSize(std::size_t workspaceSize) noexcept`
  * 设置最大工作空间大小。在 TensorRT 8.3 中已弃用，被 `setMemoryPoolLimit()` 取代。
  * 参数：`workspaceSize` 是最大工作空间大小。

\


* `std::size_t getMaxWorkspaceSize() const noexcept`
  * 获取最大工作空间大小。在 TensorRT 8.3 中已弃用，被 `getMemoryPoolLimit()` 取代。
  * 返回值：最大工作空间大小。

\


**`BuilderFlag`相关方法**

* `void setFlags(BuilderFlags builderFlags) noexcept`
  * 设置构建模式标志，标志在 `BuilderFlags` 枚举中列出。
  * 参数：`builderFlags` 是构建标志。

\


* `BuilderFlags getFlags() const noexcept`
  * 获取构建模式标志。
  * 返回值：构建标志。

\


* `void clearFlag(BuilderFlag builderFlag) noexcept`
  * 清除单个构建模式标志。
  * 参数：`builderFlag` 是要清除的构建标志。

\


* `void setFlag(BuilderFlag builderFlag) noexcept`
  * 设置单个构建模式标志。
  * 参数：`builderFlag` 是要设置的构建标志。

\


* `bool getFlag(BuilderFlag builderFlag) const noexcept`
  * 查询构建模式标志是否已设置。
  * 参数：`builderFlag` 是要查询的构建标志。
  * 返回值：`true` 表示标志已设置，`false` 表示未设置。

\


**设备类型相关方法**

\


* `void setDeviceType(ILayer const* layer, DeviceType deviceType) noexcept`
  * 设置层必须在其上执行的设备类型。
  * 参数：`layer` 是层对象，`deviceType` 是设备类型。

\


* `DeviceType getDeviceType(ILayer const* layer) const noexcept`
  * 获取层执行的设备类型。
  * 参数：`layer` 是层对象。
  * 返回值：设备类型。

\


* `bool isDeviceTypeSet(ILayer const* layer) const noexcept`
  * 查询是否已为层显式设置设备类型。
  * 参数：`layer` 是层对象。
  * 返回值：`true` 表示设备类型已设置，`false` 表示未设置。

\


* `void resetDeviceType(ILayer const* layer) noexcept`
  * 重置层的设备类型。
  * 参数：`layer` 是层对象。

\


* `bool canRunOnDLA(ILayer const* layer) const noexcept`
  * 检查层是否可以在 DLA 上运行。
  * 参数：`layer` 是层对象。
  * 返回值：`true` 表示可以在 DLA 上运行，`false` 表示不能。

\


* `void setDLACore(int32_t dlaCore) noexcept`
  * 设置用于网络的 DLA 核心。默认为 -1。
  * 参数：`dlaCore` 是 DLA 核心索引。

\


* `int32_t getDLACore() const noexcept`
  * 获取引擎执行的 DLA 核心。
  * 返回值：DLA 核心索引，未设置或 DLA 不存在时为 -1。

\


* `void setDefaultDeviceType(DeviceType deviceType) noexcept`
  * 设置构建器使用的默认设备类型。
  * 参数：`deviceType` 是设备类型。

\


*   `DeviceType getDefaultDeviceType() const noexcept`

    * 获取默认设备类型。
    * 返回值：设备类型。



### enum class DeviceType

```c
//!
//! \enum DeviceType
//! \brief The device that this layer/network will execute on.
//!
//!
enum class DeviceType : int32_t
{
    kGPU, //!< GPU Device
    kDLA, //!< DLA Core
};
```

\


**重置builder配置**

\


* `void reset() noexcept`
  * 重置builder配置为默认值。

\


**CUDA 流相关方法**

\


* `void setProfileStream(const cudaStream_t stream) noexcept`
  * 设置用于配置文件的 CUDA 流。
  * 参数：`stream` 是 CUDA 流。

\


* `cudaStream_t getProfileStream() const noexcept`
  * 获取用于配置文件的 CUDA 流。
  * 返回值：CUDA 流。

\


**优化配置文件相关方法**

\


* `int32_t addOptimizationProfile(IOptimizationProfile const* profile) noexcept`
  * 添加优化配置文件。对于具有动态或形状输入张量的网络，必须至少调用一次此函数。
  * 参数：`profile` 是优化配置文件对象。
  * 返回值：优化配置文件的索引，输入无效时为 -1。

\


* `int32_t getNbOptimizationProfiles() const noexcept`
  * 获取优化配置文件的数量。
  * 返回值：优化配置文件的数量。

\
\


**算法选择器相关方法**

\


* `void setAlgorithmSelector(IAlgorithmSelector* selector) noexcept`
  * 设置算法选择器。
  * 参数：`selector` 是算法选择器对象。

\


* `IAlgorithmSelector* getAlgorithmSelector() const noexcept`
  * 获取算法选择器。
  * 返回值：算法选择器对象。

\


**校准配置文件相关方法**

\


* `bool setCalibrationProfile(IOptimizationProfile const* profile) noexcept`
  * 设置校准配置文件。
  * 参数：`profile` 是校准配置文件对象。
  * 返回值：`true` 表示设置成功，`false` 表示失败。

\


* `IOptimizationProfile const* getCalibrationProfile() noexcept`
  * 获取当前校准配置文件。
  * 返回值：校准配置文件对象。

\


**量化标志相关方法**

\


* `void setQuantizationFlags(QuantizationFlags flags) noexcept`
  * 设置量化标志，标志在 `QuantizationFlag` 枚举中列出。
  * 参数：`flags` 是量化标志。

\


* `QuantizationFlags getQuantizationFlags() const noexcept`
  * 获取量化标志。
  * 返回值：量化标志。

\


* `void clearQuantizationFlag(QuantizationFlag flag) noexcept`
  * 清除单个量化标志。
  * 参数：`flag` 是要清除的量化标志。

\


* `void setQuantizationFlag(QuantizationFlag flag) noexcept`
  * 设置单个量化标志。
  * 参数：`flag` 是要设置的量化标志。

\


* `bool getQuantizationFlag(QuantizationFlag flag) const noexcept`
  * 查询量化标志是否已设置。
  * 参数：`flag` 是要查询的量化标志。
  * 返回值：`true` 表示标志已设置，`false` 表示未设置。

\


### enum class QuantizationFlag

```c
//!
//! \enum QuantizationFlag
//!
//! \brief List of valid flags for quantizing the network to int8
//!
//! \see IBuilderConfig::setQuantizationFlag(), IBuilderConfig::getQuantizationFlag()
//!
enum class QuantizationFlag : int32_t
{
    //! Run int8 calibration pass before layer fusion. Only valid for IInt8LegacyCalibrator and
    //! IInt8EntropyCalibrator. The builder always runs the int8 calibration pass before layer fusion for
    //! IInt8MinMaxCalibrator and IInt8EntropyCalibrator2. Disabled by default.
    kCALIBRATE_BEFORE_FUSION = 0
};
```

* 在层融合之前运行 int8 校准过程。
* 该标志仅对 `IInt8LegacyCalibrator` 和 `IInt8EntropyCalibrator` 有效。对于 `IInt8MinMaxCalibrator` 和 `IInt8EntropyCalibrator2`，构建器总是在层融合之前运行 int8 校准过程。
* 默认情况下，该标志是禁用的。

**作用**:

* 当启用该标志时，在进行层融合之前，构建器会运行 int8 校准过程。这意味着在对网络中的层进行任何融合操作之前，构建器会先校准这些层，以确保量化的准确性和有效性。

**使用场景**:

* 在使用 `IInt8LegacyCalibrator` 和 `IInt8EntropyCalibrator` 进行量化时，如果希望在层融合之前先进行校准，可以启用该标志。这可以确保在层融合之前，网络的每一层都经过精确的量化校准，从而可能提高最终模型的精度。

\


**策略源相关方法**

\


* `bool setTacticSources(TacticSources tacticSources) noexcept`
  * 设置策略源，控制 TensorRT 用于策略选择的策略源。策略来源是指 TensorRT 在构建深度学习推理引擎时所能使用的一组算法库或实现方法。不同的策略来源可能对应不同的底层计算库，例如 cuBLAS 和 cuBLASLt。这些库提供了不同的实现，TensorRT 可以从中选择最优的实现（策略）以达到最佳的性能。
  * 参数：`tacticSources` 是策略源，是一个位掩码（bitmask），用于指定可以使用的策略来源。不同的策略来源可以通过按位或操作（bitwise OR）组合在一起。例如，要启用 cuBLAS 和 cuBLASLt 作为策略来源，可以使用：

```C
1U << static_cast<uint32_t>(TacticSource::kCUBLAS) | 1U << static_cast<uint32_t>(TacticSource::kCUBLAS_LT)
```

* `TacticSources getTacticSources() const noexcept`
  * 获取策略源。
  * 返回值：策略源。

假设我们要启用 cuBLAS 和 cuBLASLt 作为策略来源，可以这样使用这个函数：

```c
// 定义策略来源的位掩码
TacticSources sources = (1U << static_cast<uint32_t>(TacticSource::kCUBLAS)) | 
                        (1U << static_cast<uint32_t>(TacticSource::kCUBLAS_LT));

// 调用 setTacticSources 函数设置策略来源
bool result = config->setTacticSources(sources);

if (result) {
    std::cout << "Tactic sources updated successfully." << std::endl;
} else {
    std::cout << "Failed to update tactic sources." << std::endl;
}
```

**定时缓存相关方法**

\


* `nvinfer1::ITimingCache* createTimingCache(void const* blob, std::size_t size) const noexcept`
  * 从序列化原始数据创建 `ITimingCache` 实例。
  * 参数：`blob` 是包含序列化定时缓存的原始数据指针，`size` 是序列化定时缓存的大小。
  * 返回值：创建的定时缓存对象。

\


* `bool setTimingCache(ITimingCache const& cache, bool ignoreMismatch) noexcept`
  * 将定时缓存附加到 `IBuilderConfig`。
  * 参数：`cache` 是定时缓存对象，`ignoreMismatch` 表示是否忽略不匹配。
  * 返回值：`true` 表示设置成功，`false` 表示失败。

\


* `nvinfer1::ITimingCache const* getTimingCache() const noexcept`
  * 获取 `IBuilderConfig` 中的定时缓存对象。
  * 返回值：定时缓存对象。

\
策略计时信息（Tactic Timing Information）是指在使用 TensorRT 构建深度学习推理引擎时，对不同的计算策略（tactics）进行计时和性能评估的信息。这些策略用于优化神经网络层的执行，以选择最快或最有效的计算方法。\
在 TensorRT 中，策略是指执行某个操作（例如卷积、矩阵乘法等）的特定实现方式。不同的策略可能使用不同的算法、数据布局、内存访问模式等。为了选择最佳策略，TensorRT 在构建引擎时会对多种策略进行测试和计时，然后选择性能最优的策略用于最终的推理引擎。\


#### 计时信息的收集和使用

\


1. **收集策略计时信息**
   1. 当 TensorRT 构建引擎时，会针对每个网络层尝试多个策略，并对这些策略进行计时。这些计时信息包括每个策略的执行时间、所需的内存大小等。
   2. TensorRT 会根据这些计时信息，选择每个层的最佳策略（最小化推理时间或内存占用等）。

\


2. **使用计时缓存**
   1. TensorRT 提供了计时缓存（Timing Cache）的机制，可以将收集到的计时信息保存下来，以加速后续的引擎构建过程。
   2. 当构建新引擎时，可以加载之前保存的计时缓存，这样可以避免重新进行所有策略的测试和计时，从而加快构建速度。

\


#### 具体实现和作用

\
通过 `ITimingCache` 类，程序可以序列化、合并和重置计时缓存，具体方法包括：\


* **序列化计时缓存**：将当前的计时信息保存到 `IHostMemory` 对象中，可以写入文件或传输给其他实例。

```C
nvinfer1::IHostMemory* serialize() const noexcept
```

* **合并计时缓存**：将另一个计时缓存中的信息合并到当前缓存中，可以共享不同设备或不同构建过程的计时信息。

```C
bool combine(ITimingCache const& inputCache, bool ignoreMismatch) noexcept
```

细节：将输入缓存中的条目追加到本地缓存中，冲突的条目将被跳过。如果输入缓存是由不同版本的 TensorRT 生成的，合并将被跳过并返回 `false`。如果将 `ignoreMismatch` 设置为 `true`，则可以合并由不同设备创建的计时缓存。

* **重置计时缓存**：清空当前的计时缓存。

```C
bool reset() noexcept
```

以下是如何在程序中使用计时缓存的示例：

```c
// 创建或初始化计时缓存
ITimingCache* timingCache = builderConfig->createTimingCache(nullptr, 0);

// 在构建引擎时使用计时缓存
builderConfig->setTimingCache(*timingCache, true);

// 构建引擎
ICudaEngine* engine = builder->buildEngineWithConfig(network, *builderConfig);

// 序列化计时缓存以便在下次使用
nvinfer1::IHostMemory* serializedCache = timingCache->serialize();
std::ofstream cacheFile("timing_cache.bin", std::ios::binary);
cacheFile.write(reinterpret_cast<const char*>(serializedCache->data()), serializedCache->size());
cacheFile.close();

// 下次构建时加载计时缓存
std::ifstream cacheFile("timing_cache.bin", std::ios::binary);
std::vector<char> buffer((std::istreambuf_iterator<char>(cacheFile)), std::istreambuf_iterator<char>());
ITimingCache* newTimingCache = builderConfig->createTimingCache(buffer.data(), buffer.size());
builderConfig->setTimingCache(*newTimingCache, true);

// 构建新引擎时使用已保存的计时缓存
ICudaEngine* newEngine = builder->buildEngineWithConfig(network, *builderConfig);
```

\


### nvinfer1::IHostMemory

```c
class IHostMemory : public INoCopy
{
public:
    virtual ~IHostMemory() noexcept = default;

    //! A pointer to the raw data that is owned by the library.
    void* data() const noexcept
    {
        return mImpl->data();
    }

    //! The size in bytes of the data that was allocated.
    std::size_t size() const noexcept
    {
        return mImpl->size();
    }

    //! The type of the memory that was allocated.
    DataType type() const noexcept
    {
        return mImpl->type();
    }
    //!
    //! Destroy the allocated memory.
    //!
    //! \deprecated Deprecated in TRT 8.0. Superseded by `delete`.
    //!
    //! \warning Calling destroy on a managed pointer will result in a double-free error.
    //!
    TRT_DEPRECATED void destroy() noexcept
    {
        delete this;
    }

protected:
    apiv::VHostMemory* mImpl;
};
```

该类主要用于管理由库分配的内存。用户可以通过该类访问和操作分配的内存，但不需要自己负责内存的分配和释放。\


1. **`void* data() const noexcept`**：
   * **描述**: 返回指向原始数据的指针，该数据由库拥有。
   * **返回类型**: `void*`
   * **作用**: 获取分配的内存数据。

\


2. **`std::size_t size() const noexcept`**：
   * **描述**: 返回分配数据的字节大小。
   * **返回类型**: `std::size_t`
   * **作用**: 获取分配数据的大小。

\


3. **`DataType type() const noexcept`**：
   * **描述**: 返回分配内存的数据类型。
   * **返回类型**: `DataType`，表示数据类型。
   * **作用**: 获取分配数据的类型。

\
\


### **`nvinfer1::ICudaEngine`**

`ICudaEngine`用于执行构建网络的推理引擎上的推理。\


<figure><img src="https://m7jsqg9xk0.feishu.cn/space/api/box/stream/download/asynccode/?code=ZWVhNjQ4ZmFjMzRlZjg1NGE3YTY4MDdkMDQzMzhkNDhfWnJSYmtIVjBWb1FSOEU1RHYzaHJMNE94YTNKTmF2VFFfVG9rZW46VmdNN2J1SDhGb0RZWG14ZUxNOWNXdkhHbmZmXzE3MTgwMzg4OTI6MTcxODA0MjQ5Ml9WNA" alt=""><figcaption></figcaption></figure>

1. `getNbBindings() const noexcept`: 获取绑定索引的数量。返回类型为`int32_t`，表示绑定索引的数量。

\


2. `getBindingIndex(char const* name) const noexcept`: 根据给定的张量名获取绑定索引。返回类型为`int32_t`，表示绑定索引。

\


3. `getBindingName(int32_t bindingIndex) const noexcept`: 根据给定的绑定索引获取绑定的名称。返回类型为`char const*`，表示绑定的名称。

\


4. `bindingIsInput(int32_t bindingIndex) const noexcept`: 确定绑定是否为输入绑定。返回类型为`bool`，表示绑定是否为输入绑定。

\


5. `getBindingDimensions(int32_t bindingIndex) const noexcept`: 获取绑定的维度。返回类型为`Dims`，表示绑定的维度。

\


6. `getTensorShape(char const* tensorName) const noexcept`: 获取输入或输出张量的形状。返回类型为`Dims`，表示张量的形状。

\


7. `getTensorDataType(char const* tensorName) const noexcept`: 根据张量名确定缓冲区的数据类型。返回类型为`DataType`，表示数据类型。

\


8. `getNbLayers() const noexcept`: 获取网络中的层数。返回类型为`int32_t`，表示网络中的层数。

\


9. `serialize() const noexcept`: 将网络序列化为流。返回类型为`IHostMemory*`，表示包含序列化引擎的主机内存对象。

\


10. `createExecutionContext() noexcept`: 创建执行上下文。返回类型为`IExecutionContext*`，表示执行上下文对象。

\


11. `getTensorLocation(char const* tensorName) const noexcept`: 获取输入或输出张量所需的设备位置。返回类型为`TensorLocation`，表示张量所需的设备位置。

\


12. `isShapeInferenceIO(char const* tensorName) const noexcept`: 确定张量是否作为形状推断的输入或输出。返回类型为`bool`，表示张量是否作为形状推断的输入或输出。

\


13. `getTensorIOMode(char const* tensorName) const noexcept`: 确定张量是输入还是输出张量。返回类型为`TensorIOMode`，表示张量的输入输出模式。

\


14. `createExecutionContextWithoutDeviceMemory() noexcept`: 创建一个没有分配任何设备内存的执行上下文。返回类型为`IExecutionContext*`，表示执行上下文对象。

\


15. `getDeviceMemorySize() const noexcept`: 返回执行上下文所需的设备内存量。返回类型为`size_t`，表示设备内存的大小。

\


16. `isRefittable() const noexcept`: 返回引擎是否可以重新适配的布尔值。返回类型为`bool`，表示引擎是否可以重新适配。

\


17. `getTensorBytesPerComponent(char const* tensorName) const noexcept`: 返回一个元素的每个组件的字节大小。返回类型为`int32_t`，表示每个组件的字节大小。

\


18. `getTensorComponentsPerElement(char const* tensorName) const noexcept`: 返回一个元素中包含的组件数。返回类型为`int32_t`，表示每个元素中包含的组件数。

\


19. `getTensorFormat(char const* tensorName) const noexcept`: 返回绑定格式。返回类型为`TensorFormat`，表示张量格式。

\


20. `getTensorFormatDesc(char const* tensorName) const noexcept`: 返回张量格式的人类可读描述。返回类型为`char const*`，表示张量格式的描述。

```c
    //!
    //! \brief Return the human readable description of the tensor format, or empty string if the provided name does not
    //! map to an input or output tensor.
    //!
    //! The description includes the order, vectorization, data type, and strides.
    //! Examples are shown as follows:
    //!   Example 1: kCHW + FP32
    //!     "Row major linear FP32 format"
    //!   Example 2: kCHW2 + FP16
    //!     "Two wide channel vectorized row major FP16 format"
    //!   Example 3: kHWC8 + FP16 + Line Stride = 32
    //!     "Channel major FP16 format where C % 8 == 0 and H Stride % 32 == 0"
    //!
    //! \param tensorName The name of an input or output tensor.
    //!
    //! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including the terminator.
    //!
    char const* getTensorFormatDesc(char const* tensorName) const noexcept
    {
        return mImpl->getTensorFormatDesc(tensorName);
    }
```

21. `getTensorVectorizedDim(char const* tensorName) const noexcept`: 返回缓冲区向量化的维度索引。返回类型为`int32_t`，表示向量化的维度索引。

\


22. `getName() const noexcept`: 返回与引擎关联的网络的名称。返回类型为`char const*`，表示网络的名称。

\


23. `getNbOptimizationProfiles() const noexcept`: 获取为该引擎定义的优化配置文件的数量。返回类型为`int32_t`，表示优化配置文件的数量。

\


24. `getProfileShape(char const* tensorName, int32_t profileIndex, OptProfileSelector select) const noexcept`: 获取给定优化配置文件下输入张量的最小/最佳/最大维度。返回类型为`Dims`，表示输入张量的维度。

\


25. `getEngineCapability() const noexcept`: 查询引擎的执行能力。返回类型为`EngineCapability`，表示引擎的执行能力。

\


26. `setErrorRecorder(IErrorRecorder* recorder) noexcept`: 设置此接口的错误记录器。返回类型为`void`。

\


27. `getErrorRecorder() const noexcept`: 获取分配给此接口的错误记录器。返回类型为`IErrorRecorder*`，表示分配给此接口的错误记录器。

\


28. `hasImplicitBatchDimension() const noexcept`: 查询引擎是否具有隐式批次维度。返回类型为`bool`，表示引擎是否具有隐式批次维度。

\


29. `getTacticSources() const noexcept`: 返回引擎所需的策略源。返回类型为`TacticSources`，表示引擎所需的策略源。

\


30. `getProfilingVerbosity() const noexcept`: 返回构建引擎时生成的详细程度。返回类型为`ProfilingVerbosity`，表示构建引擎时生成的详细程度。

\


31. `createEngineInspector() const noexcept`: 创建一个新的引擎检查器，用于打印引擎或执行上下文中的层信息。返回类型为`IEngineInspector*`，表示引擎检查器对象。

\


32. `getNbIOTensors() const noexcept`: 返回IO张量的数量。返回类型为`int32_t`，表示IO张量的数量。

\


33. `getIOTensorName(int32_t index) const noexcept`: 返回IO张量的名称。返回类型为`char const*`，表示IO张量的名称。

\


### nvinfer1::IRuntime

`IRuntime` 类用于反序列化一个引擎。\


<figure><img src="https://m7jsqg9xk0.feishu.cn/space/api/box/stream/download/asynccode/?code=M2Q1MmQ3YmJiODZlOTg3YWU5Y2VhNmZiMWZlZGU0MThfdlRxRmVIeWY5bGRkam45TUl6OWI3TWcxS0U1MzhxUmFfVG9rZW46UVRTbWJDM2N5b0ZpOW94UkFmaGNWdEs4blZjXzE3MTgwMzg4OTI6MTcxODA0MjQ5Ml9WNA" alt=""><figcaption></figcaption></figure>

**引擎反序列化相关方法**

\


* `ICudaEngine* deserializeCudaEngine(void const* blob, std::size_t size) noexcept`
  * 从流中反序列化引擎。如果已为运行时设置了错误记录器，它也将传递给引擎。
  * 参数：
    * `blob`：保存序列化引擎的内存。
    * `size`：内存大小，以字节为单位。
  * 返回值：反序列化后的引擎，反序列化失败则返回 nullptr。

\


**DLA 核心相关方法**

\


* `void setDLACore(int32_t dlaCore) noexcept`
  * 设置网络使用的 DLA 核心。默认值为 -1。
  * 参数：`dlaCore` 是 DLA 核心索引，范围在 \[0, getNbDLACores())。
  * 警告：如果 `getNbDLACores()` 返回 0，则此方法无效。

\


* `int32_t getDLACore() const noexcept`
  * 获取引擎执行的 DLA 核心。
  * 返回值：分配的 DLA 核心索引，DLA 不存在或未设置时为 -1。

\


* `int32_t getNbDLACores() const noexcept`
  * 返回可访问的 DLA 硬件核心数量，如果 DLA 不可用，则返回 0。
  * 返回值：DLA 硬件核心数量。

\


**销毁方法**

\


* `TRT_DEPRECATED void destroy() noexcept`
  * 销毁对象。在 TensorRT 8.0 中已弃用，改为使用 `delete`。
  * 警告：在托管指针上调用销毁会导致双重释放错误。

\


**日志记录器相关方法**

\


* `ILogger* getLogger() const noexcept`
  * 获取创建运行时时使用的日志记录器。
  * 返回值：日志记录器。

\


**线程相关方法**

\


* `bool setMaxThreads(int32_t maxThreads) noexcept`
  * 设置运行时可以使用的最大线程数。默认值为 1，包括当前线程。大于 1 的值允许 TensorRT 使用多线程算法。小于 1 的值会触发 `kINVALID_ARGUMENT` 错误。
  * 参数：`maxThreads` 是最大线程数。
  * 返回值：成功返回 true，否则返回 false。

\


* `int32_t getMaxThreads() const noexcept`
  * 获取运行时可以使用的最大线程数。
  * 返回值：最大线程数。

\


### nvinfer1::IExecutionContext

`IExecutionContext` 类用于执行通过 `ICudaEngine` 创建的推理上下文。多个执行上下文可以存在于一个 `ICudaEngine` 实例中，允许使用相同的引擎同时执行多个批次的推理。\


<figure><img src="https://m7jsqg9xk0.feishu.cn/space/api/box/stream/download/asynccode/?code=MDkwZmNmOTU5NWYwYzU2NzU4NGI3YzI3OGU0Yzg0Y2FfQWswNlJyeDFTcjhnZWpWYmVZR0E2a2pCcElXVktWY0ZfVG9rZW46WGJLa2JzMDVpb3RDdGt4OVNlemNiUDFBbkRPXzE3MTgwMzg4OTI6MTcxODA0MjQ5Ml9WNA" alt=""><figcaption></figcaption></figure>

**执行推理**

\


* `bool executeV2(void* const* bindings) noexcept`
  * 同步执行推理，仅适用于完整维度网络。
  * 参数：
    * `bindings`：输入和输出缓冲区的指针数组。
  * 返回值：执行成功返回 `true`，否则返回 `false`。

\


* `TRT_DEPRECATED bool enqueueV2(void* const* bindings, cudaStream_t stream, cudaEvent_t* inputConsumed) noexcept`
  * 异步执行推理，仅适用于完整维度网络。
  * 参数：
    * `bindings`：输入和输出缓冲区的指针数组。
    * `stream`：用于推理内核的 CUDA 流。
    * `inputConsumed`：可选事件，当输入缓冲区可以重新填充新数据时发出信号。
  * 返回值：内核成功排队返回 `true`，否则返回 `false`。

\


* `bool enqueueV3(cudaStream_t stream) noexcept`
  * 异步执行推理。
  * 参数：
    * `stream`：用于推理内核的 CUDA 流。
  * 返回值：内核成功排队返回 `true`，否则返回 `false`。

\


**优化配置文件**

\


* `TRT_DEPRECATED bool setOptimizationProfile(int32_t profileIndex) noexcept`
  * 选择当前上下文的优化配置文件。
  * 参数：
    * `profileIndex`：配置文件索引。
  * 返回值：调用成功返回 `true`，否则返回 `false`。

\


* `int32_t getOptimizationProfile() const noexcept`
  * 获取当前选择的优化配置文件索引。
  * 返回值：配置文件索引。

\


* `bool setOptimizationProfileAsync(int32_t profileIndex, cudaStream_t stream) noexcept`
  * 异步选择当前上下文的优化配置文件。
  * 参数：
    * `profileIndex`：配置文件索引。
    * `stream`：用于数据复制的 CUDA 流。
  * 返回值：调用成功返回 `true`，否则返回 `false`。

\


**输入和输出设置**

\


* `TRT_DEPRECATED bool setBindingDimensions(int32_t bindingIndex, Dims dimensions) noexcept`
  * 设置输入绑定的动态维度。
  * 参数：
    * `bindingIndex`：输入张量的索引。
    * `dimensions`：输入张量的维度。
  * 返回值：调用成功返回 `true`，否则返回 `false`。

\


* `bool setInputShape(char const* tensorName, Dims const& dims) noexcept`
  * 设置给定输入的形状。
  * 参数：
    * `tensorName`：输入张量的名称。
    * `dims`：输入张量的形状。
  * 返回值：调用成功返回 `true`，否则返回 `false`。

\


* `TRT_DEPRECATED Dims getBindingDimensions(int32_t bindingIndex) const noexcept`
  * 获取绑定的动态维度。
  * 参数：
    * `bindingIndex`：绑定索引。
  * 返回值：当前选择的绑定维度。

\


* `Dims getTensorShape(char const* tensorName) const noexcept`
  * 获取给定输入或输出的形状。
  * 参数：
    * `tensorName`：输入或输出张量的名称。
  * 返回值：输入或输出张量的形状。

\


* `TRT_DEPRECATED bool setInputShapeBinding(int32_t bindingIndex, int32_t const* data) noexcept`
  * 设置输入张量的形状绑定值。
  * 参数：
    * `bindingIndex`：输入张量的索引。
    * `data`：输入张量的形状值。
  * 返回值：调用成功返回 `true`，否则返回 `false`。

\


* `TRT_DEPRECATED bool getShapeBinding(int32_t bindingIndex, int32_t* data) const noexcept`
  * 获取输入张量的形状绑定值。
  * 参数：
    * `bindingIndex`：输入或输出张量的索引。
    * `data`：存储形状值的指针。
  * 返回值：获取成功返回 `true`，否则返回 `false`。

\


**名称和引擎**

\


* `void setName(char const* name) noexcept`
  * 设置执行上下文的名称。
  * 参数：
    * `name`：名称字符串。

\


* `char const* getName() const noexcept`
  * 获取执行上下文的名称。
  * 返回值：名称字符串。

\


* `ICudaEngine const& getEngine() const noexcept`
  * 获取关联的引擎。
  * 返回值：引擎引用。

\


**张量地址**

\


* `bool setTensorAddress(char const* tensorName, void* data) noexcept`
  * 设置给定输入或输出张量的内存地址。
  * 参数：
    * `tensorName`：张量名称。
    * `data`：数据指针。
  * 返回值：调用成功返回 `true`，否则返回 `false`。

\


* `void const* getTensorAddress(char const* tensorName) const noexcept`
  * 获取给定输入或输出张量的内存地址。
  * 参数：
    * `tensorName`：张量名称。
  * 返回值：数据指针。

\


* `bool setInputTensorAddress(char const* tensorName, void const* data) noexcept`
  * 设置给定输入的内存地址。
  * 参数：
    * `tensorName`：输入张量名称。
    * `data`：数据指针。
  * 返回值：调用成功返回 `true`，否则返回 `false`。

\


* `void* getOutputTensorAddress(char const* tensorName) const noexcept`
  * 获取给定输出的内存地址。
  * 参数：
    * `tensorName`：输出张量名称。
* 返回值：数据指针。

\


**最大输出大小**

\


* `int64_t getMaxOutputSize(char const* tensorName) const noexcept`
  * 获取输出张量的最大大小（以字节为单位）。
  * 参数：
    * `tensorName`：输出张量名称。
  * 返回值：输出张量的最大大小（以字节为单位）。

\
\


### nvinfer1::IAlgorithmIOInfo

> 可以参看官方教程：[https://github.com/NVIDIA/TensorRT/tree/release/8.6/samples/sampleAlgorithmSelector](https://github.com/NVIDIA/TensorRT/tree/release/8.6/samples/sampleAlgorithmSelector)学习这几个API的使用

`IAlgorithmIOInfo` 类提供了有关算法输入或输出的信息。与 `IAlgorithmVariant` 一起，这些信息描述了算法的变化，可以用于通过 `IAlgorithmSelector::selectAlgorithms()`重现算法。\


1. **`TensorFormat getTensorFormat() const noexcept`**
   1. 返回算法输入/输出的张量格式。
   2. **返回值**：`TensorFormat` 类型，表示张量格式。

\


2. **`DataType getDataType() const noexcept`**
   1. 返回算法输入/输出的数据类型。
   2. **返回值**：`DataType` 类型，表示数据类型。

\


3. **`Dims getStrides() const noexcept`**
   1. 返回算法输入/输出张量的步幅。
   2. **返回值**：`Dims` 类型，表示张量的步幅。

\


### nvinfer1::IAlgorithmVariant

\
`IAlgorithmVariant` 类提供了一个唯一的 128 位标识符，该标识符与输入和输出信息一起描述算法的变化，可以用于通过 `IAlgorithmSelector::selectAlgorithms()` 重现算法。\


1. **`int64_t getImplementation() const noexcept`**
   1. 返回算法的实现。
   2. **返回值**：`int64_t` 类型，表示算法的实现。

\


2. **`int64_t getTactic() const noexcept`**
   1. 返回算法的策略。
   2. **返回值**：`int64_t` 类型，表示算法的策略。

\


### nvinfer1::IAlgorithmContext

\
`IAlgorithmContext` 类描述了可以由一个或多个 `IAlgorithm` 实例满足的上下文。\


1. **`char const* getName() const noexcept`**
   1. 返回算法节点的名称，这是 `IAlgorithmContext` 的唯一标识符。
   2. **返回值**：`char const*` 类型，表示算法节点的名称。

\


2. **`Dims getDimensions(int32_t index, OptProfileSelector select) const noexcept`**
   1. 获取输入或输出张量的最小/最优/最大维度。
   2. **参数**：
      * `index`：算法输入或输出的索引。
      * `select`：要查询的最小、最优或最大维度。
   3. **返回值**：`Dims` 类型，表示所选维度的张量。

\


3. **`int32_t getNbInputs() const noexcept`**
   1. 返回算法的输入数量。
   2. **返回值**：`int32_t` 类型，表示输入数量。

\


4. **`int32_t getNbOutputs() const noexcept`**
   1. 返回算法的输出数量。
   2. **返回值**：`int32_t` 类型，表示输出数量。

\


### nvinfer1::IAlgorithm

\
`IAlgorithm` 类描述了一层的执行变体。算法由 `IAlgorithmVariant` 和每个输入和输出的 `IAlgorithmIOInfo` 表示，可以通过 `AlgorithmSelector::selectAlgorithms()` 重现。\


1. **`IAlgorithmIOInfo const& getAlgorithmIOInfo(int32_t index) const noexcept`**
   1. 返回算法输入或输出的格式。
   2. **参数**：
      * `index`：算法输入或输出的索引。
   3. **返回值**：`IAlgorithmIOInfo` 类型的引用，表示所选索引的算法输入/输出信息。

\


2. **`IAlgorithmVariant const& getAlgorithmVariant() const noexcept`**
   1. 返回算法变体。
   2. **返回值**：`IAlgorithmVariant` 类型的引用，表示算法变体。

\


3. **`float getTimingMSec() const noexcept`**
   1. 返回算法的执行时间，以毫秒为单位。
   2. **返回值**：`float` 类型，表示执行时间。

\


4. **`std::size_t getWorkspaceSize() const noexcept`**
   1. 返回算法在执行时使用的 GPU 临时内存大小，以字节为单位。
   2. **返回值**：`std::size_t` 类型，表示内存大小。

\


5. **`IAlgorithmIOInfo const* getAlgorithmIOInfoByIndex(int32_t index) const noexcept`**
   1. 返回算法输入或输出的格式。
   2. **参数**：
      * `index`：算法输入或输出的索引。
   3. **返回值**：`IAlgorithmIOInfo` 类型的指针，表示所选索引的算法输入/输出信息。如果索引超出范围，则返回 nullptr。

\


### nvinfer1::IAlgorithmSelector

\
`IAlgorithmSelector` 类提供了一个接口，应用程序可以实现该接口来选择构建器为层提供的算法。\


1. **`int32_t selectAlgorithms(IAlgorithmContext const& context, IAlgorithm const* const* choices, int32_t nbChoices, int32_t* selection) noexcept`**
   1. 从给定的算法选择列表中选择算法。
   2. **参数**：
      * `context`：算法选择上下文。
      * `choices`：可选算法列表。
      * `nbChoices`：算法选择的数量。
      * `selection`：用户在此缓冲区中写入所选选择的索引。
   3. **返回值**：返回选择的算法数量，范围在 `[0, nbChoices-1]`。

\


2. **`void reportAlgorithms(IAlgorithmContext const* const* algoContexts, IAlgorithm const* const* algoChoices, int32_t nbAlgorithms) noexcept`**
   1. 由 TensorRT 调用以报告其做出的选择。
   2. **参数**：
      * `algoContexts`：所有算法上下文的列表。
      * `algoChoices`：TensorRT 做出的算法选择列表。
      * `nbAlgorithms`：`algoContexts` 和 `algoChoices` 的大小。

\


### nvinfer1::IInt8Calibrator

`IInt8Calibrator 类及其派生类提供了用于校准 INT8 推理的接口和实现。每个派生类使用不同的校准算法，并实现了相应的方法来获取校准批次、读取和写入校准缓存，以及获取校准算法类型。IInt8LegacyCalibrator 还提供了读取和写入直方图缓存的方法，以及获取分位数和回归截止值的方法。`\


<figure><img src="https://m7jsqg9xk0.feishu.cn/space/api/box/stream/download/asynccode/?code=ODk2MWQ3NDdmMWYyZTEyYWUyOWFlNDdjNmY4MjY3MDFfbEM0MmFRUktvZmhuc0FSRzFSTnFib2FRTlVDVmtsWTJfVG9rZW46VThWSWJWUmdzbzRZSFF4YnB3Z2M4TTlubmdoXzE3MTgwMzg4OTI6MTcxODA0MjQ5Ml9WNA" alt=""><figcaption></figcaption></figure>

* `virtual int32_t getBatchSize() const noexcept = 0;`
  * 获取用于校准批次的批量大小。
  * 返回值：批量大小（`int32_t`）。

\


* `virtual bool getBatch(void* bindings[], char const* names[], int32_t nbBindings) noexcept = 0;`
  * 获取用于校准的批次输入。
  * 参数：
    * `bindings`：指向设备内存的指针数组，需要更新以指向包含每个网络输入数据的设备内存。
    * `names`：绑定数组中每个指针的网络输入名称。
    * `nbBindings`：绑定数组中的指针数量。
  * 返回值：如果没有更多批次用于校准，则返回 `false`，否则返回 `true`。

\


* `virtual void const* readCalibrationCache(std::size_t& length) noexcept = 0;`
  * 读取校准缓存。
  * 参数：
    * `length`：缓存数据的长度，应由被调用函数设置。如果没有数据，应为零。
  * 返回值：指向缓存的指针，如果没有数据则为 `nullptr`。

\


* `virtual void writeCalibrationCache(void const* ptr, std::size_t length) noexcept = 0;`
  * 保存校准缓存。
  * 参数：
    * `ptr`：指向要缓存的数据的指针。
    * `length`：要缓存的数据长度（以字节为单位）。

\


* `virtual CalibrationAlgoType getAlgorithm() noexcept = 0;`
  * 获取此校准器使用的算法。
  * 返回值：校准器使用的算法（`CalibrationAlgoType`）。

\
\


### **nvinfer1::ILogger**

`ILogger` 是一个用于应用程序实现日志记录接口的类。该类提供了一个抽象接口，允许应用程序定义自己的日志记录方法。它主要用于 TensorRT 的builder、运行时系统。

```c
//!
//! \class ILogger
//!
//! \brief Application-implemented logging interface for the builder, refitter and runtime.
//!
//! The logger used to create an instance of IBuilder, IRuntime or IRefitter is used for all objects created through
//! that interface. The logger should be valid until all objects created are released.
//!
//! The Logger object implementation must be thread safe. All locking and synchronization is pushed to the
//! interface implementation and TensorRT does not hold any synchronization primitives when calling the interface
//! functions.
//!
class ILogger
{
public:
    //!
    //! \enum Severity
    //!
    //! The severity corresponding to a log message.
    //!
    enum class Severity : int32_t
    {
        //! An internal error has occurred. Execution is unrecoverable.
        kINTERNAL_ERROR = 0,
        //! An application error has occurred.
        kERROR = 1,
        //! An application error has been discovered, but TensorRT has recovered or fallen back to a default.
        kWARNING = 2,
        //!  Informational messages with instructional information.
        kINFO = 3,
        //!  Verbose messages with debugging information.
        kVERBOSE = 4,
    };

    //!
    //! A callback implemented by the application to handle logging messages;
    //!
    //! \param severity The severity of the message.
    //! \param msg A null-terminated log message.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when multiple execution contexts are used during runtime, or if the same logger is used
    //!                  for multiple runtimes, builders, or refitters.
    //!
    virtual void log(Severity severity, AsciiChar const* msg) noexcept = 0;

    ILogger() = default;
    virtual ~ILogger() = default;

protected:
// @cond SuppressDoxyWarnings
    ILogger(ILogger const&) = default;
    ILogger(ILogger&&) = default;
    ILogger& operator=(ILogger const&) & = default;
    ILogger& operator=(ILogger&&) & = default;
// @endcond
};
```

1. **抽象接口**: `ILogger` 提供了一个抽象接口，强制应用程序实现 `log` 方法，以便处理不同严重性的日志消息。
2. **线程安全**: 日志记录方法 `log` 需要是线程安全的，因为可能会被多个线程同时调用。

* 简单示例

```c
#include <iostream>
#include <string>

// 自定义的日志记录类，继承自 ILogger
class CustomLogger : public ILogger {
public:
    // 实现纯虚函数 log
    void log(Severity severity, AsciiChar const* msg) noexcept override {
        std::string severityStr;
        switch (severity) {
            case Severity::kINTERNAL_ERROR:
                severityStr = "INTERNAL ERROR";
                break;
            case Severity::kERROR:
                severityStr = "ERROR";
                break;
            case Severity::kWARNING:
                severityStr = "WARNING";
                break;
            case Severity::kINFO:
                severityStr = "INFO";
                break;
            case Severity::kVERBOSE:
                severityStr = "VERBOSE";
                break;
        }
        std::cout << "[" << severityStr << "] " << msg << std::endl;
    }
};

int main() {
    CustomLogger logger;
    logger.log(ILogger::Severity::kINFO, "This is an informational message.");
    logger.log(ILogger::Severity::kERROR, "This is an error message.");
    return 0;
}
```

\


### `nvinfer1::Ilayer`

`ILayer` 类是所有网络层类的基类，用于定义网络中的层。这个类提供了许多用于操作和查询层的方法，如获取层类型、设置和获取层名称、输入和输出张量的数量、设置和获取计算精度等。\


<figure><img src="https://m7jsqg9xk0.feishu.cn/space/api/box/stream/download/asynccode/?code=Mjg2YjVkODVkNzIxZTFiM2FlMzllZWZhYTYxNzJmNTlfQmVYb2Z5N21rQlBTQ0FSbmlkWEJtWVp2aXo0S0dDV29fVG9rZW46VGFYd2JMS2I2b0czblN4WDhsemNqMDBOblFoXzE3MTgwMzg4OTI6MTcxODA0MjQ5Ml9WNA" alt=""><figcaption></figcaption></figure>

1. **`LayerType getType() const noexcept`**：
   1. **描述**：返回层的类型。
   2. **返回类型**：`LayerType`，表示层的类型。

\


2. **`void setName(char const* name) noexcept`**：
   1. **描述**：设置层的名称，并复制该名称字符串。
   2. **参数**：`name`，层的名称字符串。
   3. **警告**：名称字符串必须是以 null 结尾的，并且长度不超过 4096 字节（包括终止符）。

\


3. **`char const* getName() const noexcept`**：
   1. **描述**：返回层的名称。
   2. **返回类型**：`char const*`，层的名称字符串。

\


4. **`int32_t getNbInputs() const noexcept`**：
   1. **描述**：返回层的输入数量。
   2. **返回类型**：`int32_t`，输入数量。

\


5. **`ITensor* getInput(int32_t index) const noexcept`**：
   1. **描述**：返回指定索引的输入张量。
   2. **参数**：`index`，输入张量的索引。
   3. **返回类型**：`ITensor*`，输入张量指针。如果索引超出范围或张量是可选的，则返回 `nullptr`。

\


6. **`int32_t getNbOutputs() const noexcept`**：
   1. **描述**：返回层的输出数量。
   2. **返回类型**：`int32_t`，输出数量。

\


7. **`ITensor* getOutput(int32_t index) const noexcept`**：
   1. **描述**：返回指定索引的输出张量。
   2. **参数**：`index`，输出张量的索引。
   3. **返回类型**：`ITensor*`，输出张量指针。如果索引超出范围或张量是可选的，则返回 `nullptr`。

\


8. **`void setInput(int32_t index, ITensor& tensor) noexcept`**：
   1. **描述**：用指定的张量替换层的输入。
   2. **参数**：`index`，要修改的输入索引；`tensor`，新的输入张量。
   3. **注意**：对于大多数层，不能改变输入的数量，索引必须小于 `getNbInputs()` 的值。

\


9. **`void setPrecision(DataType dataType) noexcept`**：
   1. **描述**：设置层的计算精度。
   2. **参数**：`dataType`，计算精度类型。
   3. **说明**：设置计算精度后，TensorRT 会选择在此精度下运行的实现，或者选择最快的实现。

\


10. **`DataType getPrecision() const noexcept`**：
    1. **描述**：返回层的计算精度。
    2. **返回类型**：`DataType`，计算精度类型。

\


11. **`bool precisionIsSet() const noexcept`**：
    1. **描述**：检查层的计算精度是否已设置。
    2. **返回类型**：`bool`，表示计算精度是否已显式设置。

\


12. **`void resetPrecision() noexcept`**：
    1. **描述**：重置层的计算精度。

\


13. **`void setOutputType(int32_t index, DataType dataType) noexcept`**：
    1. **描述**：设置层的输出类型。
    2. **参数**：`index`，输出索引；`dataType`，输出数据类型。
    3. **说明**：设置输出类型后，TensorRT 会选择生成该类型数据的实现。

\


14. **`DataType getOutputType(int32_t index) const noexcept`**：
    1. **描述**：返回层的输出类型。
    2. **参数**：`index`，输出索引。
    3. **返回类型**：`DataType`，输出数据类型。

\


15. **`bool outputTypeIsSet(int32_t index) const noexcept`**：
    1. **描述**：检查层的输出类型是否已设置。
    2. **参数**：`index`，输出索引。
    3. **返回类型**：`bool`，表示输出类型是否已显式设置。

\


16. **`void resetOutputType(int32_t index) noexcept`**：
    1. **描述**：重置层的输出类型。
    2. **参数**：`index`，输出索引。

\
\


### **`nvinfer1::INetworkDefinition`**

`INetworkDefinition` 是一个网络定义类，用于构建器的输入。以下API功能我剔除了类似addXXLayer这种的（实在太多了），功能就是给网络添加特定的层结构，所以我只列出了几个重点API：

<figure><img src="https://m7jsqg9xk0.feishu.cn/space/api/box/stream/download/asynccode/?code=OGY4MDc3NWM5NmMwMjE1MDA1OTNmMjdiMjZmZjQyMWNfTVZLWmRFbXlNRFNRNWVmR2duZmVCQ1VxQlE1QTFpVmdfVG9rZW46TUVET2JlZEtGb21ZQ1F4Z2E1S2NlWUlHbjljXzE3MTgwMzg4OTI6MTcxODA0MjQ5Ml9WNA" alt=""><figcaption></figcaption></figure>

> **addXXLayer API的学习参看代码实例**

1. **addInput(char const\* name, DataType type, Dims dimensions)**：
   1. 作用：向网络中添加一个输入张量。
   2. 返回值类型：ITensor\*（指向添加的新张量的指针）。

\


2. **markOutput(ITensor& tensor)**：
   1. 作用：将一个张量标记为网络输出。
   2. 返回值类型：void。

\


3. **getNbLayers()**：
   1. 作用：获取网络中层的数量。
   2. 返回值类型：int32\_t（表示网络中层的数量）。

\


4. **getLayer(int32\_t index)**：
   1. 作用：获取指定索引处的层。
   2. 返回值类型：ILayer\*（指向获取的层的指针）。

\


5. **getNbInputs()**：
   1. 作用：获取网络中输入张量的数量。
   2. 返回值类型：int32\_t（表示网络中输入张量的数量）。

\


6. **getInput(int32\_t index)**：
   1. 作用：获取指定索引处的输入张量。
   2. 返回值类型：ITensor\*（指向获取的输入张量的指针）。

\


7. **getNbOutputs()**：
   1. 作用：获取网络中输出张量的数量。
   2. 返回值类型：int32\_t（表示网络中输出张量的数量）。

\


8. **getOutput(int32\_t index)**：
   1. 作用：获取指定索引处的输出张量。
   2. 返回值类型：ITensor\*（指向获取的输出张量的指针）。

\


9. **destroy()**：
   1. 作用：销毁当前的 `INetworkDefinition` 对象。
   2. 返回值类型：void。

\


10. **removeTensor(ITensor& tensor)**：
    1. 作用：从网络定义中移除一个张量。
    2. 返回值类型：void。
    3. \

11. **unmarkOutput(ITensor& tensor)**：
    1. 作用：取消将一个张量标记为网络输出。
    2. 返回值类型：void。

\


12. **addPluginV2(ITensor\* const\* inputs, int32\_t nbInputs, IPluginV2& plugin)**：
    1. 作用：向网络中添加一个插件层。
    2. 返回值类型：IPluginV2Layer\*（指向添加的新插件层的指针）。

\


13. **setName(char const\* name)**：
    1. 作用：设置网络的名称。
    2. 返回值类型：void。

\


14. **getName()**：
    1. 作用：获取网络的名称。
    2. 返回值类型：char const\*（指向网络名称的指针）。

\


15. **hasImplicitBatchDimension()**：
    1. 作用：查询网络是否具有隐式批量维度。
    2. 返回值类型：bool（true 表示具有隐式批量维度，false 表示没有）。

\


16. **markOutputForShapes(ITensor& tensor)**：
    1. 作用：将张量标记为用于计算形状的网络输出。
    2. 返回值类型：bool（true 表示成功，false 表示失败）。

\


17. **unmarkOutputForShapes(ITensor& tensor)**：
    1. 作用：取消将张量标记为用于计算形状的网络输出。
    2. 返回值类型：bool（true 表示成功，false 表示失败）。

\


18. **hasExplicitPrecision()**：
    1. 作用：查询网络是否具有显式精度。
    2. 返回值类型：bool（true 表示具有显式精度，false 表示没有）。

\


#### 一个小细节

注意`nvinfer1::ICudaEngine`和`nvinfer1::INetworkDefinition`都有`getNbLayers()`这个方法，但是有所不同：

1. `nvinfer1::ICudaEngine.getNbLayers()`: 这个函数返回的是已经构建好的引擎（`ICudaEngine`对象）中的层数。这个引擎是通过TensorRT的`IBuilder`构建得到的，可能经过了优化、融合等操作，因此这个层数反映了实际在硬件上执行的网络层数，是优化后的结果。
2. `nvinfer1::INetworkDefinition.getNbLayers()`: 这个函数返回的是网络定义（`INetworkDefinition`对象）中的层数。这个网络定义是在构建引擎之前定义的，可能还没有经过优化、融合等操作，所以这个层数反映的是网络在构建之前的原始状态，不考虑优化后的情况。

\
\


### enum class BuilderFlag

\
`BuilderFlag` 枚举类定义了一组构建引擎时可启用的模式，适用于从网络定义创建引擎的构建器。每个枚举值表示一种特定的构建模式或选项。\


#### 枚举值和解释

\


1. **kFP16 = 0**
   1. 启用 FP16 层选择，并在需要时回退到 FP32。
   2. 在可能的情况下使用半精度浮点数进行计算，但如果某些操作不支持 FP16，则回退到标准的单精度浮点数 (FP32)。

\


2. **kINT8 = 1**
   1. 启用 Int8 层选择，并在需要时回退到 FP32 或 FP16（如果同时指定了 kFP16）。
   2. 允许在 8 位整数 (Int8) 精度下执行计算，以提高性能和减少内存使用。

\


3. **kDEBUG = 2**
   1. 启用层调试，通过在每个层之后进行同步来调试层。
   2. 对每个层进行调试，以帮助排查网络中的问题。

\


4. **kGPU\_FALLBACK = 3**
   1. 启用 GPU 回退功能，如果某些层无法在 DLA 上执行，则回退到 GPU 执行。
   2. 允许在无法在深度学习加速器 (DLA) 上执行时，将计算任务转移到 GPU。

\


5. **kREFIT = 5**
   1. 启用构建可重新适配的引擎。
   2. 允许构建后对网络进行重新训练或重新调整权重。

\


6. **kDISABLE\_TIMING\_CACHE = 6**
   1. 禁用跨相同层重用计时信息。
   2. 防止在构建过程中使用先前构建中收集的层计时数据。

\


7. **kTF32 = 7**
   1. 允许（但不强制）对类型为 DataType::kFLOAT 的张量使用 TF32 进行计算。
   2. TF32 通过将输入舍入为 10 位尾数后进行乘法运算，然后使用 23 位尾数累加和。
   3. 默认情况下启用。

\


8. **kSPARSE\_WEIGHTS = 8**
   1. 允许构建器检查权重并在权重具有适当稀疏性时使用优化的函数。
   2. 利用稀疏矩阵运算优化性能。

\


9. **kSAFETY\_SCOPE = 9**
   1. 将 EngineCapability::kSTANDARD 流程中的允许参数更改为与 EngineCapability::kSAFETY 对 DeviceType::kGPU 的检查相匹配，或与 EngineCapability::kDLA\_STANDALONE 对 DeviceType::kDLA 的检查相匹配。
   2. 如果未设置且在构建时指定了 EngineCapability::kSAFETY，该标志强制为 true。
   3. 仅在 NVIDIA Drive 产品中支持。

\


10. **kOBEY\_PRECISION\_CONSTRAINTS = 10**
    1. 要求层在指定的精度下执行。否则构建失败。
    2. 强制执行精度约束，确保每层都在指定的精度下运行。

\


11. **kPREFER\_PRECISION\_CONSTRAINTS = 11**
    1. 优先在指定的精度下执行层。如果构建失败，则回退到其他精度（并发出警告）。
    2. 尽量在指定精度下运行层，如果无法满足，则回退到其他可用的精度。

\


12. **kDIRECT\_IO = 12**
    1. 要求不在层与网络 I/O 张量之间插入重格式化操作。
    2. 如果为了功能正确性需要重格式化，则构建失败。

\


13. **kREJECT\_EMPTY\_ALGORITHMS = 13**
    1. 如果 IAlgorithmSelector::selectAlgorithms 返回空的算法集合，则构建失败。
    2. 确保选择算法时不会返回空集，以避免构建失败。

\


14. **kENABLE\_TACTIC\_HEURISTIC = 14**
    1. 启用基于启发式的策略选择，以缩短引擎生成时间。生成的引擎可能没有使用基于配置文件的构建器时那样高效。
    2. 仅在 NVIDIA Ampere 及更高版本的 GPU 上支持。

\


#### 相关方法

\


* `IBuilderConfig::setFlag(BuilderFlag flag)`
  * 设置指定的构建标志。
  * \

* `IBuilderConfig::getFlag(BuilderFlag flag)`
  * 获取指定的构建标志的状态。

\
\


### _`enum`_ _`class`_` ``DataType`

```c
enum class DataType : int32_t
{
    //! 32-bit floating point format.
    kFLOAT = 0,

    //! IEEE 16-bit floating-point format.
    kHALF = 1,

    //! Signed 8-bit integer representing a quantized floating-point value.
    kINT8 = 2,

    //! Signed 32-bit integer format.
    kINT32 = 3,

    //! 8-bit boolean. 0 = false, 1 = true, other values undefined.
    kBOOL = 4,
    
    kUINT8 = 5

};
```

这段代码定义了一个枚举类型 `DataType`，表示权重和张量的数据类型。枚举类型 `DataType` 包含以下枚举值：

1. `kFLOAT`: 表示32位浮点数格式。
2. `kHALF`: 表示IEEE 16位浮点数格式。
3. `kINT8`: 表示一个量化浮点值的带符号8位整数。
4. `kINT32`: 表示带符号32位整数格式。
5. `kBOOL`: 表示8位布尔值。0表示假，1表示真，其他值未定义。
6. `kUINT8`: 表示无符号8位整数格式。

\


### `enum class LayerType`

```c
//!
//! \enum LayerType
//!
//! \brief The type values of layer classes.
//!
//! \see ILayer::getType()
//!
enum class LayerType : int32_t
{
    kCONVOLUTION = 0,         //!< Convolution layer.
    kFULLY_CONNECTED = 1,     //!< Fully connected layer.
    kACTIVATION = 2,          //!< Activation layer.
    kPOOLING = 3,             //!< Pooling layer.
    kLRN = 4,                 //!< LRN layer.
    kSCALE = 5,               //!< Scale layer.
    kSOFTMAX = 6,             //!< SoftMax layer.
    kDECONVOLUTION = 7,       //!< Deconvolution layer.
    kCONCATENATION = 8,       //!< Concatenation layer.
    kELEMENTWISE = 9,         //!< Elementwise layer.
    kPLUGIN = 10,             //!< Plugin layer.
    kUNARY = 11,              //!< UnaryOp operation Layer.
    kPADDING = 12,            //!< Padding layer.
    kSHUFFLE = 13,            //!< Shuffle layer.
    kREDUCE = 14,             //!< Reduce layer.
    kTOPK = 15,               //!< TopK layer.
    kGATHER = 16,             //!< Gather layer.
    kMATRIX_MULTIPLY = 17,    //!< Matrix multiply layer.
    kRAGGED_SOFTMAX = 18,     //!< Ragged softmax layer.
    kCONSTANT = 19,           //!< Constant layer.
    kRNN_V2 = 20,             //!< RNNv2 layer.
    kIDENTITY = 21,           //!< Identity layer.
    kPLUGIN_V2 = 22,          //!< PluginV2 layer.
    kSLICE = 23,              //!< Slice layer.
    kSHAPE = 24,              //!< Shape layer.
    kPARAMETRIC_RELU = 25,    //!< Parametric ReLU layer.
    kRESIZE = 26,             //!< Resize Layer.
    kTRIP_LIMIT = 27,         //!< Loop Trip limit layer
    kRECURRENCE = 28,         //!< Loop Recurrence layer
    kITERATOR = 29,           //!< Loop Iterator layer
    kLOOP_OUTPUT = 30,        //!< Loop output layer
    kSELECT = 31,             //!< Select layer.
    kFILL = 32,               //!< Fill layer
    kQUANTIZE = 33,           //!< Quantize layer
    kDEQUANTIZE = 34,         //!< Dequantize layer
    kCONDITION = 35,          //!< Condition layer
    kCONDITIONAL_INPUT = 36,  //!< Conditional Input layer
    kCONDITIONAL_OUTPUT = 37, //!< Conditional Output layer
    kSCATTER = 38,            //!< Scatter layer
    kEINSUM = 39,             //!< Einsum layer
    kASSERTION = 40,          //!< Assertion layer
    kONE_HOT = 41,            //!< OneHot layer
    kNON_ZERO = 42,           //!< NonZero layer
    kGRID_SAMPLE = 43,        //!< Grid sample layer
    kNMS = 44,                //!< NMS layer
};
```

\


### `enum class TensorLocation`

```c
enum class TensorLocation : int32_t
{
    kDEVICE = 0, //!< Data stored on device.
    kHOST = 1,   //!< Data stored on host.
};
```

用于表示张量数据存储的位置，可以是设备上（GPU）或主机上（CPU）。枚举值包括 `kDEVICE` 表示设备上的数据，值为 `0`，以及 `kHOST` 表示主机上的数据，值为 `1`。\


### `enum class TensorFormat`

```c
//!
//! \enum TensorFormat
//!
//! \brief Format of the input/output tensors.
//!
//! This enum is used by both plugins and network I/O tensors.
//!
//! \see IPluginV2::supportsFormat(), safe::ICudaEngine::getBindingFormat()
//!
//! For more information about data formats, see the topic "Data Format Description" located in the
//! TensorRT Developer Guide.
//!
enum class TensorFormat : int32_t
{
    //! Row major linear format.
    //! For a tensor with dimensions {N, C, H, W} or {numbers, channels,
    //! columns, rows}, the dimensional index corresponds to {3, 2, 1, 0}
    //! and thus the order is W minor.
    //!
    //! For DLA usage, the tensor sizes are limited to C,H,W in the range [1,8192].
    //!
    kLINEAR = 0,

    //! Two wide channel vectorized row major format. This format is bound to
    //! FP16. It is only available for dimensions >= 3.
    //! For a tensor with dimensions {N, C, H, W},
    //! the memory layout is equivalent to a C array with dimensions
    //! [N][(C+1)/2][H][W][2], with the tensor coordinates (n, c, h, w)
    //! mapping to array subscript [n][c/2][h][w][c%2].
    kCHW2 = 1,

    //! Eight channel format where C is padded to a multiple of 8. This format
    //! is bound to FP16. It is only available for dimensions >= 3.
    //! For a tensor with dimensions {N, C, H, W},
    //! the memory layout is equivalent to the array with dimensions
    //! [N][H][W][(C+7)/8*8], with the tensor coordinates (n, c, h, w)
    //! mapping to array subscript [n][h][w][c].
    kHWC8 = 2,

    //! Four wide channel vectorized row major format. This format is bound to
    //! INT8 or FP16. It is only available for dimensions >= 3.
    //! For INT8, the C dimension must be a build-time constant.
    //! For a tensor with dimensions {N, C, H, W},
    //! the memory layout is equivalent to a C array with dimensions
    //! [N][(C+3)/4][H][W][4], with the tensor coordinates (n, c, h, w)
    //! mapping to array subscript [n][c/4][h][w][c%4].
    //!
    //! Deprecated usage:
    //!
    //! If running on the DLA, this format can be used for acceleration
    //! with the caveat that C must be equal or lesser than 4.
    //! If used as DLA input and the build option kGPU_FALLBACK is not specified,
    //! it needs to meet line stride requirement of DLA format. Column stride in bytes should
    //! be a multiple of 32 on Xavier and 64 on Orin.
    kCHW4 = 3,

    //! Sixteen wide channel vectorized row major format. This format is bound
    //! to FP16. It is only available for dimensions >= 3.
    //! For a tensor with dimensions {N, C, H, W},
    //! the memory layout is equivalent to a C array with dimensions
    //! [N][(C+15)/16][H][W][16], with the tensor coordinates (n, c, h, w)
    //! mapping to array subscript [n][c/16][h][w][c%16].
    //!
    //! For DLA usage, this format maps to the native feature format for FP16,
    //! and the tensor sizes are limited to C,H,W in the range [1,8192].
    //!
    kCHW16 = 4,

    //! Thirty-two wide channel vectorized row major format. This format is
    //! only available for dimensions >= 3.
    //! For a tensor with dimensions {N, C, H, W},
    //! the memory layout is equivalent to a C array with dimensions
    //! [N][(C+31)/32][H][W][32], with the tensor coordinates (n, c, h, w)
    //! mapping to array subscript [n][c/32][h][w][c%32].
    //!
    //! For DLA usage, this format maps to the native feature format for INT8,
    //! and the tensor sizes are limited to C,H,W in the range [1,8192].
    kCHW32 = 5,

    //! Eight channel format where C is padded to a multiple of 8. This format
    //! is bound to FP16, and it is only available for dimensions >= 4.
    //! For a tensor with dimensions {N, C, D, H, W},
    //! the memory layout is equivalent to an array with dimensions
    //! [N][D][H][W][(C+7)/8*8], with the tensor coordinates (n, c, d, h, w)
    //! mapping to array subscript [n][d][h][w][c].
    kDHWC8 = 6,

    //! Thirty-two wide channel vectorized row major format. This format is
    //! bound to FP16 and INT8 and is only available for dimensions >= 4.
    //! For a tensor with dimensions {N, C, D, H, W},
    //! the memory layout is equivalent to a C array with dimensions
    //! [N][(C+31)/32][D][H][W][32], with the tensor coordinates (n, c, d, h, w)
    //! mapping to array subscript [n][c/32][d][h][w][c%32].
    kCDHW32 = 7,

    //! Non-vectorized channel-last format. This format is bound to either FP32 or UINT8,
    //! and is only available for dimensions >= 3.
    kHWC = 8,

    //! DLA planar format. For a tensor with dimension {N, C, H, W}, the W axis
    //! always has unit stride. The stride for stepping along the H axis is
    //! rounded up to 64 bytes.
    //!
    //! The memory layout is equivalent to a C array with dimensions
    //! [N][C][H][roundUp(W, 64/elementSize)] where elementSize is
    //! 2 for FP16 and 1 for Int8, with the tensor coordinates (n, c, h, w)
    //! mapping to array subscript [n][c][h][w].
    kDLA_LINEAR = 9,

    //! DLA image format. For a tensor with dimension {N, C, H, W} the C axis
    //! always has unit stride. The stride for stepping along the H axis is rounded up
    //! to 32 bytes on Xavier and 64 bytes on Orin. C can only be 1, 3 or 4.
    //! If C == 1, it will map to grayscale format.
    //! If C == 3 or C == 4, it will map to color image format. And if C == 3,
    //! the stride for stepping along the W axis needs to be padded to 4 in elements.
    //!
    //! When C is {1, 3, 4}, then C' is {1, 4, 4} respectively,
    //! the memory layout is equivalent to a C array with dimensions
    //! [N][H][roundUp(W, 32/C'/elementSize)][C'] on Xavier and [N][H][roundUp(W, 64/C'/elementSize)][C'] on Orin
    //! where elementSize is 2 for FP16
    //! and 1 for Int8. The tensor coordinates (n, c, h, w) mapping to array
    //! subscript [n][h][w][c].
    kDLA_HWC4 = 10,

    //! Sixteen channel format where C is padded to a multiple of 16. This format
    //! is bound to FP16. It is only available for dimensions >= 3.
    //! For a tensor with dimensions {N, C, H, W},
    //! the memory layout is equivalent to the array with dimensions
    //! [N][H][W][(C+15)/16*16], with the tensor coordinates (n, c, h, w)
    //! mapping to array subscript [n][h][w][c].
    kHWC16 = 11
};
```

\


1. **kLINEAR(不涉及向量化操作)**:
   1. **描述**: 行主线性格式。
   2. **适用范围**: 所有维度都可以，但最常见于4维张量。
   3. **布局**: 数据按行排列，主要用于内存连续存储。
   4. \

2. **kCHW2**:
   1. **描述**: 两通道宽度向量化的行主格式，绑定于 FP16 数据类型。
   2. **适用范围**: 维度大于等于3。
   3. **布局**: 每两个通道向量化为一组，适用于特定的计算优化。

\


3. **kHWC8**:
   1. **描述**: 通道数被填充至8的格式，绑定于 FP16 数据类型。
   2. **适用范围**: 维度大于等于3。
   3. **布局**: 通道数被填充至8的倍数，适用于某些硬件加速器。

\


4. **kCHW4**:
   1. **描述**: 四通道宽度向量化的行主格式，绑定于 INT8 或 FP16 数据类型。
   2. **适用范围**: 维度大于等于3。
   3. **布局**: 每四个通道向量化为一组，适用于特定的计算优化。

\


5. **kCHW16**:
   1. **描述**: 十六通道宽度向量化的行主格式，绑定于 FP16 数据类型。
   2. **适用范围**: 维度大于等于3。
   3. **布局**: 每十六个通道向量化为一组，适用于特定的计算优化。

\


6. **kCHW32**:
   1. **描述**: 三十二通道宽度向量化的行主格式。
   2. **适用范围**: 维度大于等于3。
   3. **布局**: 每三十二个通道向量化为一组，适用于特定的计算优化。

\


7. **kDHWC8**:
   1. **描述**: 通道数被填充至8的格式，绑定于 FP16 数据类型。
   2. **适用范围**: 维度大于等于4。
   3. **布局**: 通道数被填充至8的倍数，适用于某些硬件加速器。

\


8. **kCDHW32**:
   1. **描述**: 三十二通道深度宽度向量化的行主格式，绑定于 FP16 或 INT8 数据类型。
   2. **适用范围**: 维度大于等于4。
   3. **布局**: 每三十二个通道向量化为一组，适用于特定的计算优化。

\


9. **kHWC**:
   1. **描述**: 非向量化的通道末尾格式，绑定于 FP32 或 UINT8 数据类型。
   2. **适用范围**: 维度大于等于3。
   3. **布局**: 通道数在最后，适用于一些特定场景。

\


10. **kDLA\_LINEAR**:

* **描述**: DLA 线性格式，适用于 DL 加速器。
* **适用范围**: 维度大于等于3。
* **布局**: 通道数在最前，W 轴有64字节对齐，适用于特定硬件架构。

\


11. **kDLA\_HWC4**:

* **描述**: DLA 图像格式，适用于 DL 加速器。
* **适用范围**: 维度大于等于3。
* **布局**: 通道数在最后，W 轴有32或64字节对齐，适用于特定硬件架构。

\


12. **kHWC16**:

* **描述**: 十六通道宽度向量化的通道末尾格式，绑定于 FP16 数据类型。
* **适用范围**: 维度大于等于3。
* **布局**: 通道数被填充至16的倍数，适用于特定的计算优化。

\
\


### `enum class TensorIOMode`

```c
//!
//! \enum TensorIOMode
//!
//! \brief Definition of tensor IO Mode.
//!
enum class TensorIOMode : int32_t
{
    //! Tensor is not an input or output.
    kNONE = 0,

    //! Tensor is input to the engine.
    kINPUT = 1,

    //! Tensor is output by the engine.
    kOUTPUT = 2
};
```

\
\


### **`nvinfer1::ProfilingVerbosity`**

```c
//!
//! \enum ProfilingVerbosity
//!
//! \brief List of verbosity levels of layer information exposed in NVTX annotations and in IEngineInspector.
//!
//! \see IBuilderConfig::setProfilingVerbosity(),
//!      IBuilderConfig::getProfilingVerbosity(),
//!      IEngineInspector
//!
enum class ProfilingVerbosity : int32_t
{
    kLAYER_NAMES_ONLY = 0, //!< Print only the layer names. This is the default setting.
    kNONE = 1,             //!< Do not print any layer information.
    kDETAILED = 2,         //!< Print detailed layer information including layer names and layer parameters.

    //! \deprecated Deprecated in TensorRT 8.0. Superseded by kLAYER_NAMES_ONLY.
    kDEFAULT TRT_DEPRECATED_ENUM = kLAYER_NAMES_ONLY,
    //! \deprecated Deprecated in TensorRT 8.0. Superseded by kDETAILED.
    kVERBOSE TRT_DEPRECATED_ENUM = kDETAILED
};
```

`ProfilingVerbosity` 是一个枚举类型，用于设置在NVTX注解和IEngineInspector中暴露的层信息的详细程度。枚举值如下：

* `kLAYER_NAMES_ONLY`: 仅打印层名称。这是默认设置。
* `kNONE`: 不打印任何层信息。
* `kDETAILED`: 打印详细的层信息，包括层名称和层参数。

可以根据需要选择不同的详细程度，从而更好地理解和分析网络模型的层信息。\


### `nvinfer1::IEngineInspector`

```c
//!
//! \class IEngineInspector
//!
//! \brief An engine inspector which prints out the layer information of an engine or an execution context.
//!
//! The amount of printed information depends on the profiling verbosity setting of the builder config when the engine
//! is built:
//! - ProfilingVerbosity::kLAYER_NAMES_ONLY: only layer names will be printed.
//! - ProfilingVerbosity::kNONE: no layer information will be printed.
//! - ProfilingVerbosity::kDETAILED: layer names and layer parameters will be printed.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
//! \see ProfilingVerbosity, IEngineInspector
//!
class IEngineInspector : public INoCopy
{
public:
    virtual ~IEngineInspector() noexcept = default;

    //!
    //! \brief Set an execution context as the inspection source.
    //!
    //! Setting the execution context and specifying all the input shapes allows the inspector
    //! to calculate concrete dimensions for any dynamic shapes and display their format information.
    //! Otherwise, values dependent on input shapes will be displayed as -1 and format information
    //! will not be shown.
    //!
    //! Passing nullptr will remove any association with an execution context.
    //!
    //! \return Whether the action succeeds.
    //!
    bool setExecutionContext(IExecutionContext const* context) noexcept
    {
        return mImpl->setExecutionContext(context);
    }

    //!
    //! \brief Get the context currently being inspected.
    //!
    //! \return The pointer to the context currently being inspected.
    //!
    //! \see setExecutionContext()
    //!
    IExecutionContext const* getExecutionContext() const noexcept
    {
        return mImpl->getExecutionContext();
    }

    //!
    //! \brief Get a string describing the information about a specific layer in the current engine or the execution
    //!        context.
    //!
    //! \param layerIndex the index of the layer. It must lie in range [0, engine.getNbLayers()).
    //!
    //! \param format the format the layer information should be printed in.
    //!
    //! \return A null-terminated C-style string describing the information about a specific layer in the current
    //!         engine or the execution context.
    //!
    //! \warning The content of the returned string may change when another execution context has
    //!          been set, or when another getLayerInformation() or getEngineInformation() has been called.
    //!
    //! \warning In a multi-threaded environment, this function must be protected from other threads changing the
    //!          inspection source. If the inspection source changes, the data that is being pointed to can change.
    //!          Copy the string to another buffer before releasing the lock in order to guarantee consistency.
    //!
    //! \see LayerInformationFormat
    //!
    char const* getLayerInformation(int32_t layerIndex, LayerInformationFormat format) const noexcept
    {
        return mImpl->getLayerInformation(layerIndex, format);
    }

    //!
    //! \brief Get a string describing the information about all the layers in the current engine or the execution
    //!        context.
    //!
    //! \param layerIndex the index of the layer. It must lie in range [0, engine.getNbLayers()).
    //!
    //! \param format the format the layer information should be printed in.
    //!
    //! \return A null-terminated C-style string describing the information about all the layers in the current
    //!         engine or the execution context.
    //!
    //! \warning The content of the returned string may change when another execution context has
    //!          been set, or when another getLayerInformation() or getEngineInformation() has been called.
    //!
    //! \warning In a multi-threaded environment, this function must be protected from other threads changing the
    //!          inspection source. If the inspection source changes, the data that is being pointed to can change.
    //!          Copy the string to another buffer before releasing the lock in order to guarantee consistency.
    //!
    //! \see LayerInformationFormat
    //!
    char const* getEngineInformation(LayerInformationFormat format) const noexcept
    {
        return mImpl->getEngineInformation(format);
    }

    //!
    //! \brief Set the ErrorRecorder for this interface
    //!
    //! Assigns the ErrorRecorder to this interface. The ErrorRecorder will track all errors during execution.
    //! This function will call incRefCount of the registered ErrorRecorder at least once. Setting
    //! recorder to nullptr unregisters the recorder with the interface, resulting in a call to decRefCount if
    //! a recorder has been registered.
    //!
    //! If an error recorder is not set, messages will be sent to the global log stream.
    //!
    //! \param recorder The error recorder to register with this interface.
    //
    //! \see getErrorRecorder()
    //!
    void setErrorRecorder(IErrorRecorder* recorder) noexcept
    {
        mImpl->setErrorRecorder(recorder);
    }

    //!
    //! \brief Get the ErrorRecorder assigned to this interface.
    //!
    //! Retrieves the assigned error recorder object for the given class. A nullptr will be returned if
    //! an error handler has not been set.
    //!
    //! \return A pointer to the IErrorRecorder object that has been registered.
    //!
    //! \see setErrorRecorder()
    //!
    IErrorRecorder* getErrorRecorder() const noexcept
    {
        return mImpl->getErrorRecorder();
    }

protected:
    apiv::VEngineInspector* mImpl;
}; // class IEngineInspector
```

这个类是一个引擎检查器，用于打印引擎或执行上下文中的层信息。\


1. `bool setExecutionContext(IExecutionContext const* context) noexcept`: 设置要检查的执行上下文。如果指定了所有输入形状，则检查器可以计算任何动态形状的具体维度并显示其格式信息。返回操作是否成功的布尔值。

\


2. `IExecutionContext const* getExecutionContext() const noexcept`: 获取当前被检查的执行上下文的指针。

\


3. `char const* getLayerInformation(int32_t layerIndex, LayerInformationFormat format) const noexcept`: 获取关于当前引擎或执行上下文中特定层信息的字符串描述。参数 `layerIndex` 表示层的索引，`format` 表示信息格式。返回描述信息的C风格字符串。

* 输出示例(手动去掉了输出字符串末尾的)：

```c
[info]layer_info: reshape_before_/linear/MatMul
[info]layer_info: /linear/MatMul
[info]layer_info: reshape_after_/linear/MatMul
```

4. `char const* getEngineInformation(LayerInformationFormat format) const noexcept`: 获取关于当前引擎或执行上下文中所有层信息的字符串描述。参数 `format` 表示信息格式。返回描述信息的C风格字符串。

* 输出示例：

```c

[info]layer_info: Layers:
reshape_before_/linear/MatMul
/linear/MatMul
reshape_after_/linear/MatMul

Bindings:
input0
output0
```

5. `void setErrorRecorder(IErrorRecorder* recorder) noexcept`: 设置错误记录器。参数 `recorder` 是要注册到该接口的错误记录器。这个函数不返回任何值。

\


6. `IErrorRecorder* getErrorRecorder() const noexcept`: 获取分配给该接口的错误记录器。返回注册的错误记录器对象的指针。

\


### `nvinfer1::LayerInformationFormat`

```c
enum class LayerInformationFormat : int32_t
{
    kONELINE = 0, //!< Print layer information in one line per layer.
    kJSON = 1,    //!< Print layer information in JSON format.
};
```

\
这些类是 NVIDIA TensorRT 中用于描述张量维度的类，它们继承自 `Dims` 类，并提供了各种维度的数据描述。以下是对这些类的详细解释：\


### `Dims`

`Dims` 类是一个基类，用于表示张量的维度。它包含了以下主要成员：

<figure><img src="https://m7jsqg9xk0.feishu.cn/space/api/box/stream/download/asynccode/?code=YjU3ZGY3OGIxZTFiOTdlYTg2MzI3ZDZlNWI5Y2Y4OTdfTmthQ011SDZRNTV5NGNscWJVQ2p0N0Y1aGtPMk01V3RfVG9rZW46SDlHZWJ4a2RTb0FUZ2N4QXdzN2NseHh1blhiXzE3MTgwMzg4OTI6MTcxODA0MjQ5Ml9WNA" alt=""><figcaption></figcaption></figure>

* `nbDims`：表示维度的数量。
* `d`：一个数组，存储每个维度的大小。
* `MAX_DIMS`：一个常量，表示支持的最大维度数。

\


### `Dims2`

* `Dims2()`：默认构造函数，创建一个空的 `Dims2` 对象，维度大小为 `{0, 0}`。
* `Dims2(int32_t d0, int32_t d1)`：构造一个包含两个维度的 `Dims2` 对象，维度大小为 `{d0, d1}`。

\


### `DimsHW`

`DimsHW` 类继承自 `Dims2`，用于描述二维空间数据，通常用于表示图像的高度和宽度。

* `int32_t& h()`：返回高度的引用。
* `int32_t h() const`：返回高度的常量引用。
* `int32_t& w()`：返回宽度的引用。
* `int32_t w() const`：返回宽度的常量引用。

\


### `Dims3`

`Dims3` 类继承自 `Dims2`，用于描述三维数据。\


### `Dims4`

`Dims4` 类继承自 `Dims3`，用于描述四维数据。\
\
