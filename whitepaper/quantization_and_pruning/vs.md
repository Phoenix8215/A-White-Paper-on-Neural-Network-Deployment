# 🫴 quantization from scratch|python

### 一般过程

**计算Scale=>量化=>截断=>反量化**

$$
\begin{aligned}
\text{Scale}& =\frac{(Rmax-Rmin)}{(Qmax-Qmin)}  \\
\text{Q}& =Round(\frac R{Scale})  \\
\text{Q}& =Clip(Q,-128,127)  \\
\text{R'}& =Q*Scale 
\end{aligned}
$$

> 和上一节的公式表达的意思是一样的，回字的四种写法而已。

* `from scratch`

```python
import numpy as np

# 截断操作
def saturete(x, int_max, int_min):
    return np.clip(x, int_min, int_max)

# 计算scale
def scale_cal(x, int_max, int_min):
    scale = (x.max() - x.min()) / (int_max - int_min)
    return scale

# 量化
def quant_float_data(x, scale, int_max, int_min):
    xq = saturete(np.round(x/scale), int_max, int_min)
    return xq

# 反量化
def dequant_data(xq, scale):
    x = ((xq)*scale).astype('float32')
    return x

if __name__ == "__main__":
    np.random.seed(8215)
    data_float32 = np.random.randn(3).astype('float32')
    int_max = 127
    int_min = -128
    print(f"input = {data_float32}")

    scale= scale_cal(data_float32, int_max, int_min)
    print(f"scale = {scale}")
    data_int8 = quant_float_data(data_float32, scale, int_max, int_min)
    print(f"quant_result = {np.round(data_float32 / scale)}")
    print(f"saturete_result = {data_int8}")
    data_dequant_float = dequant_data(data_int8, scale)
    print(f"dequant_result = {data_dequant_float}")
    
    print(f"diff = {data_dequant_float - data_float32}")

```

输出结果如下：

```bash
input = [ 0.21535037  1.2962357  -1.0161772 ]
scale = 0.00906828525019627
quant_result = [  24.  143. -112.]
saturete_result = [  24.  127. -112.]
dequant_result = [ 0.21763885  1.1516722  -1.0156479 ]
diff = [ 0.00228848 -0.14456344  0.00052929]
```

可以看到`1.2962357`经过`int8`量化后的结果，误差是十分大的相较于其他数值

### 对称量化

$$
\begin{aligned}
\text{Scale}& =\frac{|Rmax|}{|Qmax|}  \\
\text{Q}& =Round(\frac R{Scale})  \\
\text{Q}& =Clip(Q,-127,127)  \\
\text{R'}& =Q*Scale 
\end{aligned}
$$

* `from scratch`

```python
import numpy as np

def saturete(x):
    return np.clip(x, -127, 127)

def scale_cal(x):
    max_val = np.max(np.abs(x))
    return max_val / 127

def quant_float_data(x, scale):
    xq = saturete(np.round(x/scale))
    return xq

def dequant_data(xq, scale):
    x = (xq * scale).astype('float32')
    return x

if __name__ == "__main__":
    np.random.seed(8215)
    data_float32 = np.random.randn(3).astype('float32')
    print(f"input = {data_float32}")

    scale = scale_cal(data_float32)
    print(f"scale = {scale}")

    data_int8 = quant_float_data(data_float32, scale)
    print(f"quant_result = {data_int8}")
    data_dequant_float = dequant_data(data_int8, scale)
    print(f"dequant_result = {data_dequant_float}")

    print(f"diff = {data_dequant_float - data_float32}")

```

输出结果：

```bash
input = [ 0.21535037  1.2962357  -1.0161772 ]
scale = 0.01020658016204834
quant_result = [  21.  127. -100.]
dequant_result = [ 0.21433818  1.2962357  -1.020658  ]
diff = [-0.00101219  0.         -0.00448084]
```

可以看到量化误差相较于之前大大减少了。

### 非对称量化

$$
\begin{aligned}
\text{Scale}& =\frac{(Rmax-Rmin)}{(Qmax-Qmin)}  \\
\text{Z}& =Qmax-Round(\frac{Rmax}{Scale})  \\
\text{Q}& =Round(\frac R{Scale}+Z)  \\
\text{Q}& =Clip(Q,-128,127)  \\
\text{R'}& =(Q-Z)*Scale 
\end{aligned}
$$

* `from scratch`

```python
import numpy as np

def saturete(x, int_max, int_min):
    return np.clip(x, int_min, int_max)

def scale_z_cal(x, int_max, int_min):
    scale = (x.max() - x.min()) / (int_max - int_min)
    z = int_max - np.round((x.max() / scale))
    return scale, z

def quant_float_data(x, scale, z, int_max, int_min):
    xq = saturete(np.round(x/scale + z), int_max, int_min)
    return xq

def dequant_data(xq, scale, z):
    x = ((xq - z)*scale).astype('float32')
    return x

if __name__ == "__main__":
    np.random.seed(8215)
    data_float32 = np.random.randn(3).astype('float32')
    int_max = 127
    int_min = -128
    print(f"input = {data_float32}")

    scale, z = scale_z_cal(data_float32, int_max, int_min)
    print(f"scale = {scale}")
    print(f"z = {z}")
    data_int8 = quant_float_data(data_float32, scale, z, int_max, int_min)
    print(f"quant_result = {data_int8}")
    data_dequant_float = dequant_data(data_int8, scale, z)
    print(f"dequant_result = {data_dequant_float}")
    
    print(f"diff = {data_dequant_float - data_float32}")

```

输出结果：

```bash
input = [ 0.21535037  1.2962357  -1.0161772 ]
scale = 0.00906828525019627
z = -16.0
quant_result = [   8.  127. -128.]
dequant_result = [ 0.21763885  1.2967647  -1.0156479 ]
diff = [0.00228848 0.00052905 0.00052929]
```

在tensorRT中INT8量化使用的方法就是对称量化。对称量化方法不用计算偏移量$$Z$$，计算量小，是一种非饱和量化。如下图所示，红色的区域由于对称量化被浪费掉，没有做真实值的一个表达。&#x20;

<figure><img src="../../.gitbook/assets/图片 (2) (1) (1) (1) (1) (1) (1) (1).png" alt=""><figcaption></figcaption></figure>

在对称量化中还存在一个问题，比如目前原始数组中有1000个点分布在\[-1,1]之间，突然有个离散点分布在100处，此时做对称量化时Scale会被调整得很大，使得上下限超出\[-127,127]的范围，从而导致量化误差增大，对精度的影响也会相应增大。 因此，在对称量化中，需要谨慎处理数据中的极端值，以免对量化精度造成不利影响。

### reference

* https://blog.csdn.net/qq\_40672115/article/details/129812691
