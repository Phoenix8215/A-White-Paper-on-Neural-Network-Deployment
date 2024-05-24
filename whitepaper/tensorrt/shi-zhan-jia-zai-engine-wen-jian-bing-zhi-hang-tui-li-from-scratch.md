# 🐨 实战:加载engine文件并执行推理|from scratch

### 推理流程

Tensor RT的推理流程表达为下图：

<figure><img src="../../.gitbook/assets/图片 (176).png" alt=""><figcaption></figcaption></figure>

1. 使用 TensorRT 进行模型推理可大致分为三个关键阶段：输入数据的预处理（preprocess）、模型推理（inference）和推理结果的后处理（postprocess）。
2. 在模型推理阶段，通常需要将 Onnx 等格式的模型编译（build）成推理引擎（engine）。这个引擎可以被序列化（serialize）以永久保存。而序列化后的 engine 可以通过反序列化（deserialize）加载使用。
3. 在进行正式推理之前，需要手动分配 CUDA 内存来存储输入和输出数据流。数据会以流（stream）的方式进行传输。在推理开始之前，需要将输入数据（input\_data）从主机（host）转移到 CUDA 设备（device）中。推理结束后，推理结果需要从 CUDA 设备（device）转移到主机（host）内存中。
4. 编译模型时，必须定义日志记录器（logger），并使用解析器（parser）对模型进行解析。配置（config）用于调整模型，如通过配置文件设置模型的动态输入尺寸。通过构建器（builder）可以创建序列化的引擎。
5. 反序列化序列化后的引擎后，可以创建上下文（context）用于推理的进行。

### 实战代码 <a href="#h_665645167_1" id="h_665645167_1"></a>

整个代码目录如下：

```sh
├── config
│   └── Makefile.config
├── Makefile
├── models
│   ├── engine
│   │   └── sample.engine
│   └── onnx
│       └── sample.onnx
└── src
    ├── cpp
    │   ├── main.cpp
    │   ├── model.cpp
    │   ├── model.hpp
    │   ├── utils.cpp
    │   └── utils.hpp
    └── python
        └── generate_onnx.py
```

#### `Makefile.config`

```makefile
# 根据当前的环境修改gcc和cuda的版本
CXX                         :=  g++
CUDA_VER                    :=  11.4

# opencv和TensorRT的安装目录
TENSORRT_INSTALL_DIR        :=  /usr/include/aarch64-linux-gnu

# 大家根据自己的机器的型号选择一下ARCH，这个是nvcc需要用到的参数
# GeForce RTX 3070, 3080, 3090
# ARCH= -gencode arch=compute_86,code=[sm_86,compute_86]

# Kepler GeForce GTX 770, GTX 760, GT 740
# ARCH= -gencode arch=compute_30,code=sm_30

# Tesla A100 (GA100), DGX-A100, RTX 3080
# ARCH= -gencode arch=compute_80,code=[sm_80,compute_80]

# Tesla V100
# ARCH= -gencode arch=compute_70,code=[sm_70,compute_70]

# GeForce RTX 2080 Ti, RTX 2080, RTX 2070, Quadro RTX 8000, Quadro RTX 6000, Quadro RTX 5000, Tesla T4, XNOR Tensor Cores
# ARCH= -gencode arch=compute_75,code=[sm_75,compute_75]

# Jetson XAVIER
# ARCH= -gencode arch=compute_72,code=[sm_72,compute_72]

# GTX 1080, GTX 1070, GTX 1060, GTX 1050, GTX 1030, Titan Xp, Tesla P40, Tesla P4
# ARCH= -gencode arch=compute_61,code=sm_61 -gencode arch=compute_61,code=compute_61

# GP100/Tesla P100 - DGX-1
# ARCH= -gencode arch=compute_60,code=sm_60

# For Jetson TX1, Tegra X1, DRIVE CX, DRIVE PX - uncomment:
# ARCH= -gencode arch=compute_53,code=[sm_53,compute_53]

# For Jetson Tx2 or Drive-PX2 uncomment:
ARCH= -gencode arch=compute_62,code=[sm_62,compute_62]

# For Tesla GA10x cards, RTX 3090, RTX 3080, RTX 3070, RTX A6000, RTX A40 uncomment:
# ARCH= -gencode arch=compute_86,code=[sm_86,compute_86]


#--------------------------------------------------------------------------------------
# Compile options
DEBUG                       :=  0
SHOW_WARNING                :=  0

# Compile applications
APP				                  :=  trt-infer

```

#### `main.cpp`

```c
#include <iostream>
#include <memory>

#include "model.hpp"
#include "utils.hpp"

using namespace std;

int main(int argc, char const *argv[])
{
    Model model("models/onnx/sample.onnx");
    if(!model.build()){
        LOGE("fail in building model");
        return 0;
    }
    if(!model.infer()){
        LOGE("fail in infering model");
        return 0;
    }
    return 0;
}

```

#### `model.cpp`

```c
#include <memory>
#include <iostream>
#include <string>
#include <type_traits>

#include "model.hpp"
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "utils.hpp"
#include "cuda_runtime.h"

using namespace std;

class Logger : public nvinfer1::ILogger{
public:
    virtual void log (Severity severity, const char* msg) noexcept override{
        string str;
        switch (severity){
            case Severity::kINTERNAL_ERROR: str = RED    "[fatal]: " CLEAR;
            case Severity::kERROR:          str = RED    "[error]: " CLEAR;
            case Severity::kWARNING:        str = BLUE   "[warn]: "  CLEAR;
            case Severity::kINFO:           str = YELLOW "[info]: "  CLEAR;
            case Severity::kVERBOSE:        str = PURPLE "[verb]: "  CLEAR;
        }
        if (severity <= Severity::kINFO)
            cout << str << string(msg) << endl;
    }
};

struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
        delete obj;
    }
};

template <typename T>
using make_unique = std::unique_ptr<T, InferDeleter>;

Model::Model(string onnxPath){
    if (!fileExists(onnxPath)) {
        LOGE("%s not found. Program terminated", onnxPath.c_str());
        exit(1);
    }
    mOnnxPath   = onnxPath;
    mEnginePath = getEnginePath(mOnnxPath);
}

bool Model::build(){
    if (fileExists(mEnginePath)){
        LOG("%s has been generated!", mEnginePath.c_str());
        return true;
    } else {
        LOG("%s not found. Building engine...", mEnginePath.c_str());
    }
    Logger logger;
    auto builder       = make_unique<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    auto network       = make_unique<nvinfer1::INetworkDefinition>(builder->createNetworkV2(1));
    auto config        = make_unique<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    auto parser        = make_unique<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));

    config->setMaxWorkspaceSize(1<<28);

    if (!parser->parseFromFile(mOnnxPath.c_str(), 1)){
        LOGE("ERROR: failed to %s", mOnnxPath.c_str());
        return false;
    }

    auto engine        = make_unique<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config));
    auto plan          = builder->buildSerializedNetwork(*network, *config);
    auto runtime       = make_unique<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));

    auto f = fopen(mEnginePath.c_str(), "wb");
    fwrite(plan->data(), 1, plan->size(), f);
    fclose(f);

    mEngine            = shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()), InferDeleter());
    mInputDims         = network->getInput(0)->getDimensions();
    mOutputDims        = network->getOutput(0)->getDimensions();
    return true;
};

bool Model::infer(){
    /*
        我们在infer需要做的事情
        1. 读取model => 创建runtime, engine, context
        2. 把数据进行host->device传输
        3. 使用context推理
        4. 把数据进行device->host传输
    */

    /* 1. 读取model => 创建runtime, engine, context */
    if (!fileExists(mEnginePath)) {
        LOGE("ERROR: %s not found", mEnginePath.c_str());
        return false;
    }

    /* 反序列化从文件中读取的数据以unsigned char的vector保存*/
    vector<unsigned char> modelData;
    modelData = loadFile(mEnginePath);
    
    Logger logger;
    auto runtime     = make_unique<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    auto engine      = make_unique<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(modelData.data(), modelData.size()));
    auto context     = make_unique<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    

    // Binding [[input[0], input[1]...],[output[0], output[1]...]]
    auto input_dims   = context->getBindingDimensions(0);
    auto output_dims  = context->getBindingDimensions(1);

    LOG("input dim shape is:  %s", printDims(input_dims).c_str());
    LOG("output dim shape is: %s", printDims(output_dims).c_str());

    /* 2. host->device的数据传递 */
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    /* host memory上的数据*/
    float input_host[]{0.0193, 0.2616, 0.7713, 0.3785, 0.9980, 0.9008, 0.4766, 0.1663, 0.8045, 0.6552};
    float output_host[5];

    /* device memory上的数据*/
    float* input_device = nullptr;
    float* weight_device = nullptr;
    float* output_device = nullptr;

    int input_size = 10;
    int output_size = 5;

    /* 分配空间, 并传送数据从host到device*/
    cudaMalloc(&input_device, sizeof(input_host));
    cudaMalloc(&output_device, sizeof(output_host));
    cudaMemcpyAsync(input_device, input_host, sizeof(input_host), cudaMemcpyKind::cudaMemcpyHostToDevice, stream);

    /* 3. 模型推理, 最后做同步处理 */
    float* bindings[] = {input_device, output_device};
    bool success = context->enqueueV2((void**)bindings, stream, nullptr);

    /* 4. device->host的数据传递 */
    cudaMemcpyAsync(output_host, output_device, sizeof(output_host), cudaMemcpyKind::cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    LOG("input data is:  %s", printTensor(input_host, input_size).c_str());
    LOG("output data is: %s", printTensor(output_host, output_size).c_str());
    LOG("finished inference");
    return true;
}

```

> <mark style="color:red;">如果输入有两个，则可以使用以下方法获取绑定维度：</mark>
>
> ```cpp
> auto input_dims_0 = context->getBindingDimensions(0);
> auto input_dims_1 = context->getBindingDimensions(1);
> ```
>
> 这样可以分别获取第一个输入和第二个输入的绑定维度。

#### `model.hpp`

```c
#ifndef __MODEL_HPP__
#define __MODEL_HPP__

// TensorRT related
#include "NvOnnxParser.h"
#include "NvInfer.h"

#include <string>
#include <memory>


class Model{
public:
    Model(std::string onnxPath);
    bool build();
    bool infer();
private:
    std::string mOnnxPath;
    std::string mEnginePath;
    nvinfer1::Dims mInputDims;
    nvinfer1::Dims mOutputDims;
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
    bool constructNetwork();
    bool preprocess();
};

#endif // __MODEL_HPP__
```

#### `utils.cpp`

```c
#include "utils.hpp"
#include <experimental/filesystem>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include "NvInfer.h"

using namespace std;

bool fileExists(const string fileName) {
    if (!experimental::filesystem::exists(
            experimental::filesystem::path(fileName))) {
        return false;
    } else {
        return true;
    }
}

/**
 * @brief 获取engine的大小size，并将engine的信息载入到data中，
 * 
 * @param path engine的路径
 * @param data 存储engine数据的vector
 * @param size engine的大小
 * @return true 文件读取成功
 * @return false 文件读取失败
 */
bool fileRead(const string &path, vector<unsigned char> &data, size_t &size){
    stringstream trtModelStream;
    ifstream cache(path);
    if(!cache.is_open()) {
        cerr << "Unable to open file😅: " << path << endl;
        return false;
    }

    /* 将engine的内容写入trtModelStream中*/
    trtModelStream.seekg(0, trtModelStream.beg);
    trtModelStream << cache.rdbuf();
    cache.close();

    /* 计算model的大小*/
    trtModelStream.seekg(0, ios::end);
    size = trtModelStream.tellg();
    data.resize(size);

    // vector<uint8_t> tmp;
    trtModelStream.seekg(0, ios::beg);
    // tmp.resize(size);

    // read方法将从trtModelStream读取的数据写入以data[0]为起始地址的内存位置
    trtModelStream.read(reinterpret_cast<char *>(data.data()), size);
    return true;
}


vector<unsigned char> loadFile(const string &file){
    ifstream in(file, ios::in | ios::binary);
    if (!in.is_open())
        return {};

    in.seekg(0, ios::end);
    size_t length = in.tellg();

    vector<unsigned char> data;
    if (length > 0){
        in.seekg(0, ios::beg);
        data.resize(length);
        in.read(reinterpret_cast<char*>(data.data()), length);
    }
    in.close();
    return data;
}

string printDims(const nvinfer1::Dims dims){
    int n = 0;
    char buff[100];
    string result;

    n += snprintf(buff + n, sizeof(buff) - n, "[ ");
    for (int i = 0; i < dims.nbDims; i++){
        n += snprintf(buff + n, sizeof(buff) - n, "%d", dims.d[i]);
        if (i != dims.nbDims - 1) {
            n += snprintf(buff + n, sizeof(buff) - n, ", ");
        }
    }
    n += snprintf(buff + n, sizeof(buff) - n, " ]");
    result = buff;
    return result;
}

string printTensor(float* tensor, int size){
    int n = 0;
    char buff[100];
    string result;
    n += snprintf(buff + n, sizeof(buff) - n, "[ ");
    for (int i = 0; i < size; i++){
        n += snprintf(buff + n, sizeof(buff) - n, "%8.4lf", tensor[i]);
        if (i != size - 1){
            n += snprintf(buff + n, sizeof(buff) - n, ", ");
        }
    }
    n += snprintf(buff + n, sizeof(buff) - n, " ]");
    result = buff;
    return result;
}

// models/onnx/sample.onnx
string getEnginePath(string onnxPath){
    int name_l = onnxPath.rfind("/");
    int name_r = onnxPath.rfind(".");

    int dir_r  = onnxPath.find("/");

    string enginePath;
    enginePath = onnxPath.substr(0, dir_r);
    enginePath += "/engine";
    enginePath += onnxPath.substr(name_l, name_r - name_l);
    enginePath += ".engine";
    return enginePath;
}


// string getEnginePath(string onnxPath) {
//     // 使用 std::filesystem 来处理文件路径
//     experimental::filesystem::path filePath(onnxPath);

//     // 获取父目录路径
//     experimental::filesystem::path parentPath = filePath.parent_path();

//     // 获取文件名（不包括扩展名）
//     std::string filename = filePath.stem().string();

//     // 构造引擎文件路径
//     experimental::filesystem::path enginePath = parentPath / "engine" / (filename + ".engine");

//     return enginePath.string();
// }

```

#### `utils.hpp`

```c
#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <ostream>
#include <string>
#include "NvInfer.h"
#include <stdarg.h>
#include <vector>

#define CUDA_CHECK(call)             __cudaCheck(call, __FILE__, __LINE__)
#define LAST_KERNEL_CHECK(call)      __kernelCheck(__FILE__, __LINE__)

#define LOG(...)                     __log_info(Level::INFO, __VA_ARGS__)
#define LOGV(...)                    __log_info(Level::VERB, __VA_ARGS__)
#define LOGE(...)                    __log_info(Level::ERROR, __VA_ARGS__)

#define DGREEN    "\033[1;36m"
#define BLUE      "\033[1;34m"
#define PURPLE    "\033[1;35m"
#define GREEN     "\033[1;32m"
#define YELLOW    "\033[1;33m"
#define RED       "\033[1;31m"
#define CLEAR     "\033[0m"

enum struct Level {
    ERROR,
    INFO,
    VERB
};

static void __cudaCheck(cudaError_t err, const char* file, const int line) {
    if (err != cudaSuccess) {
        printf("ERROR: %s:%d, ", file, line);
        printf("code:%s, reason:%s\n", cudaGetErrorName(err), cudaGetErrorString(err));
        exit(1);
    }
}

static void __kernelCheck(const char* file, const int line) {
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        printf("ERROR: %s:%d, ", file, line);
        printf("code:%s, reason:%s\n", cudaGetErrorName(err), cudaGetErrorString(err));
        exit(1);
    }
}

static void __log_info(Level level, const char* format, ...) {
    char msg[1000];
    va_list args;
    va_start(args, format);
    int n = 0;

    if (level == Level::INFO) {
        n += snprintf(msg + n, sizeof(msg) - n, YELLOW "[info]:" CLEAR);
    } else if (level == Level::VERB) {
        n += snprintf(msg + n, sizeof(msg) - n, PURPLE "[verb]:" CLEAR);
    } else {
        n += snprintf(msg + n, sizeof(msg) - n, RED "[error]:" CLEAR);
    }
    n += vsnprintf(msg + n, sizeof(msg) - n, format, args);

    fprintf(stdout, "%s\n", msg);
    va_end(args);
}

bool fileExists(const std::string fileName);
bool fileRead(const std::string &path, std::vector<unsigned char> &data, size_t &size);
std::vector<unsigned char> loadFile(const std::string &path);
std::string printDims(const nvinfer1::Dims dims);
std::string printTensor(float* tensor, int size);
std::string getEnginePath(std::string onnxPath);

#endif //__UTILS_HPP__

```

#### `generate_onnx.py`

```python
import torch
import torch.nn as nn
import torch.onnx
import onnxsim
import onnx
import os

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=10, out_features=5, bias=False)
    
    def forward(self, x):
        x = self.linear(x)
        return x

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def export_norm_onnx():
    current_path = os.path.dirname(__file__)
    file = current_path + "/../../models/onnx/sample.onnx"

    input   = torch.rand(1, 10)
    model   = Model()
    torch.onnx.export(
        model         = model, 
        args          = (input,),
        f             = file,
        input_names   = ["input0"],
        output_names  = ["output0"],
        opset_version = 15)
    print("Finished normal onnx export")

    # check the exported onnx model
    model_onnx = onnx.load(file)
    onnx.checker.check_model(model_onnx)

    # use onnx-simplifier to simplify the onnx
    print(f"Simplifying with onnx-simplifier {onnxsim.__version__}...")
    model_onnx, check = onnxsim.simplify(model_onnx)
    assert check, "assert check failed"
    onnx.save(model_onnx, file)

def infer():
    setup_seed(1)
    model  = Model()
    input  = torch.tensor([[0.0193, 0.2616, 0.7713, 0.3785, 0.9980, 0.9008, 0.4766, 0.1663, 0.8045, 0.6552]])
    output = model(input)
    print(input)
    print(output)

if __name__ == "__main__":
    export_norm_onnx()
    infer()

```

#### `Makefile`

```makefile
CONFIG_LOCAL  :=  ./config/Makefile.config

include $(CONFIG_LOCAL)

BUILD_PATH    :=  build
SRC_PATH      :=  src/cpp
INC_PATH      :=  include
CUDA_DIR      :=  /usr/local/cuda-$(CUDA_VER)

CXX_SRC       :=  $(wildcard $(SRC_PATH)/*.cpp)
KERNELS_SRC   :=  $(wildcard $(SRC_PATH)/*.cu)

APP_OBJS      :=  $(patsubst $(SRC_PATH)%, $(BUILD_PATH)%, $(CXX_SRC:.cpp=.cpp.o))
APP_OBJS      +=  $(patsubst $(SRC_PATH)%, $(BUILD_PATH)%, $(KERNELS_SRC:.cu=.cu.o))  

APP_MKS       :=  $(APP_OBJS:.o=.mk)

APP_DEPS      :=  $(CXX_SRC)
APP_DEPS      +=  $(KERNELS_SRC)
APP_DEPS      +=  $(wildcard $(SRC_PATH)/*.h)
# -----------------------------------------------------

CUCC          :=  $(CUDA_DIR)/bin/nvcc
CXXFLAGS      :=  -std=c++11 -pthread -fPIC
CUDAFLAGS     :=  --shared -Xcompiler -fPIC 


INCS          :=  -I $(CUDA_DIR)/include \
                  -I $(SRC_PATH) \
									-I $(TENSORRT_INSTALL_DIR)/include \
									-I $(INC_PATH) \
									`pkg-config --cflags opencv4 2>/dev/null || pkg-config --cflags opencv`

LIBS          :=  -L "$(CUDA_DIR)/lib64" \
									-L "$(TENSORRT_INSTALL_DIR)/lib" \
                  -lcudart -lcublas -lcudnn \
									-lnvinfer -lnvonnxparser\
									-lstdc++fs \
									`pkg-config --libs opencv4 2>/dev/null || pkg-config --libs opencv`


ifeq ($(DEBUG),1)
CUDAFLAGS     +=  -g -O0 -G
CXXFLAGS      +=  -g -O0
else
CUDAFLAGS     +=  -O3
CXXFLAGS      +=  -O3
endif

ifeq ($(SHOW_WARNING),1)
CUDAFLAGS     +=  -Wall -Wunused-function -Wunused-variable -Wfatal-errors
CXXFLAGS      +=  -Wall -Wunused-function -Wunused-variable -Wfatal-errors
else
CUDAFLAGS     +=  -w
CXXFLAGS      +=  -w
endif

.PHONY: all update show clean $(APP)
all: 
	$(MAKE) $(APP)

update: $(APP)
	@echo finished updating 😎😎😎$<

$(APP): $(APP_DEPS) $(APP_OBJS)
	@$(CXX) $(APP_OBJS) -o $@ $(LIBS) $(INCS)
	@echo finished building $@. Have fun!!🥰🥰🥰

show: 
	@echo $(BUILD_PATH)
	@echo $(APP_DEPS)
	@echo $(INCS)
	@echo $(APP_OBJS)
	@echo $(APP_MKS)

clean:
	rm -rf $(APP) 😭
	rm -rf build 😭
	

ifneq ($(MAKECMDGOALS), clean)
-include $(APP_MKS)
endif

# Compile CXX
$(BUILD_PATH)/%.cpp.o: $(SRC_PATH)/%.cpp 
	@echo Compile CXX $@
	@mkdir -p $(BUILD_PATH)
	@$(CXX) -o $@ -c $< $(CXXFLAGS) $(INCS)
$(BUILD_PATH)/%.cpp.mk: $(SRC_PATH)/%.cpp
	@echo Compile Dependence CXX $@
	@mkdir -p $(BUILD_PATH)
	@$(CXX) -M $< -MF $@ -MT $(@:.cpp.mk=.cpp.o) $(CXXFLAGS) $(INCS) 

# Compile CUDA
$(BUILD_PATH)/%.cu.o: $(SRC_PATH)/%.cu
	@echo Compile CUDA $@
	@mkdir -p $(BUILD_PATH)
	@$(CUCC) -o $@ -c $< $(CUDAFLAGS) $(INCS)
$(BUILD_PATH)/%.cu.mk: $(SRC_PATH)%.cu
	@echo Compile Dependence CUDA $@
	@mkdir -p $(BUILD_PATH)
	@$(CUCC) -M $< -MF $@ -MT $(@:.cu.mk=.cu.o) $(CUDAFLAGS)


```

### reference

* [https://blog.csdn.net/weixin\_43863869/article/details/128651343](https://blog.csdn.net/weixin\_43863869/article/details/128651343)
