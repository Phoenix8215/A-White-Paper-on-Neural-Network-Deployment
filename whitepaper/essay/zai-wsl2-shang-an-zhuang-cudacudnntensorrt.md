# 在WSL2上安装CUDA\_cuDNN\_TensorRT

### 安装Toolkit Driver

使用 CUDA 前，要求 GPU 驱动与 `cuda` 的版本要匹配，匹配关系如下：

<figure><img src="https://image.aayu.today/uploads/2023/03/01/202303012238954.png" alt="" width="563"><figcaption></figcaption></figure>

> 参考：[https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions\_\_table-cuda-toolkit-driver-versions](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions\_\_table-cuda-toolkit-driver-versions)

输入 `nvidia-smi` 命令，查看 GPU 驱动版本

```shell
Thu Mar 16 18:15:53 2023
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 515.67       Driver Version: 517.00       CUDA Version: 11.7     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  On   | 00000000:01:00.0  On |                  N/A |
| N/A   56C    P3    26W /  N/A |   1125MiB /  8192MiB |     55%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      6465      G   /Xwayland                       N/A      |
+-----------------------------------------------------------------------------+
```

可以看到当前安装的驱动版本是 `517.00`

### 安装CUDA

在 Nvidia 官网选择对应版本：[https://developer.nvidia.com/cuda-toolkit-archive](https://developer.nvidia.com/cuda-toolkit-archive)。比如我选择的是 11.7 版本，选择 Linux ， x86\_64 ， WSL-Ubuntu ， 2.0 ， deb(local) ，如下

<figure><img src="../../.gitbook/assets/图片.png" alt=""><figcaption></figcaption></figure>

使用提示的安装命令直接安装即可

> CUDA默认安装路径为`/usr/local/cuda/`

### 安装nvcc

```shell
sudo apt install nvidia-cuda-toolkit
```

输入`nvcc –version`查看输出结果

```shell
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2019 NVIDIA Corporation
Built on Sun_Jul_28_19:07:16_PDT_2019
Cuda compilation tools, release 10.1, V10.1.243
```

### 安装cuDNN

官网[CUDA 深度神经网络库 (cuDNN) | NVIDIA Developer](https://developer.nvidia.com/zh-cn/cudnn)，下载`tar`

<figure><img src="../../.gitbook/assets/图片 (1).png" alt="" width="563"><figcaption></figcaption></figure>

```shell
tar -xvf cudnn-linux-x86_64-8.x.x.x_cudaX.Y-archive.tar.xz
sudo cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include 
sudo cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64 
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

`X.Y` 和 `v8.x.x.x` 分别用你本机 CUDA 和cuDNN 的版本信息替代

#### 安装TensorRT

访问：[https://developer.nvidia.com/nvidia-tensorrt-8x-download](https://developer.nvidia.com/nvidia-tensorrt-8x-download) 下载`deb`版本的 TensorRT

> GA为稳定版本，下载稳定版本。推荐以压缩包的形式下载TensorRT，这样就可以下载多个版本的TensorRT，方便程序调用。

```shell
os="ubuntuxx04"
tag="8.x.x-cuda-x.x"
sudo dpkg -i nv-tensorrt-local-repo-${os}-${tag}_1.0-1_amd64.deb
sudo cp /var/nv-tensorrt-local-repo-${os}-${tag}/*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get install tensorrt
```

> 将os和tag替换成自己的版本信息

* 验证

```shell
(base) root@LAPTOP-476JT8H0:~/workstation/# dpkg -l | grep TensorRT
ii  libnvinfer-bin                                    8.6.0.12-1+cuda12.0               amd64        TensorRT binaries
ii  libnvinfer-dev                                    8.6.0.12-1+cuda12.0               amd64        TensorRT development libraries
ii  libnvinfer-dispatch-dev                           8.6.0.12-1+cuda12.0               amd64        TensorRT development dispatch runtime libraries
ii  libnvinfer-dispatch8                              8.6.0.12-1+cuda12.0               amd64        TensorRT dispatch runtime library
ii  libnvinfer-headers-dev                            8.6.0.12-1+cuda12.0               amd64        TensorRT development headers
ii  libnvinfer-headers-plugin-dev                     8.6.0.12-1+cuda12.0               amd64        TensorRT plugin headers
ii  libnvinfer-lean-dev                               8.6.0.12-1+cuda12.0               amd64        TensorRT lean runtime libraries
ii  libnvinfer-lean8                                  8.6.0.12-1+cuda12.0               amd64        TensorRT lean runtime library
ii  libnvinfer-plugin-dev                             8.6.0.12-1+cuda12.0               amd64        TensorRT plugin libraries
ii  libnvinfer-plugin8                                8.6.0.12-1+cuda12.0               amd64        TensorRT plugin libraries
ii  libnvinfer-samples                                8.6.0.12-1+cuda12.0               all          TensorRT samples
ii  libnvinfer-vc-plugin-dev                          8.6.0.12-1+cuda12.0               amd64        TensorRT vc-plugin library
ii  libnvinfer-vc-plugin8                             8.6.0.12-1+cuda12.0               amd64        TensorRT vc-plugin library
ii  libnvinfer8                                       8.6.0.12-1+cuda12.0               amd64        TensorRT runtime libraries
ii  libnvonnxparsers-dev                              8.6.0.12-1+cuda12.0               amd64        TensorRT ONNX libraries
ii  libnvonnxparsers8                                 8.6.0.12-1+cuda12.0               amd64        TensorRT ONNX libraries
ii  libnvparsers-dev                                  8.6.0.12-1+cuda12.0               amd64        TensorRT parsers libraries
ii  libnvparsers8                                     8.6.0.12-1+cuda12.0               amd64        TensorRT parsers libraries
ii  tensorrt                                          8.6.0.12-1+cuda12.0               amd64        Meta package for TensorRT
```
