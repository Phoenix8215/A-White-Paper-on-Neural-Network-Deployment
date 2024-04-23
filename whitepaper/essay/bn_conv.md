# 🎃 融合BN和Conv层

层融合可以减少启动kernel的开销与memory操作，从而提高效率 同时，有些计算可以通过层融合优化后，跟其他计算合并

* `Vertical layer fusion` (垂直层融合)：用的比较常见，针对`conv + BN + ReLU`进行融合

<figure><img src="../../.gitbook/assets/图片 (52).png" alt=""><figcaption></figcaption></figure>

* `Horizontal layer fusion` (水平层融合)

<figure><img src="../../.gitbook/assets/图片 (53).png" alt=""><figcaption></figcaption></figure>

回顾一下`Batch Normalization`的公式,其中

<figure><img src="../../.gitbook/assets/图片 (54).png" alt="" width="339"><figcaption></figcaption></figure>

$$
\begin{aligned}\mu_B&=\frac{1}{B}\sum_{i=1}^{B}x_i\\\sigma_B^2&=\frac{1}{B}\sum_{i=1}^{B}(x_i\mu_B)^2+\epsilon\end{aligned}\begin{aligned}\widehat{x_i}&=\frac{x_i-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}     \quad  \\     \quad y_i&=\gamma*\widehat{x_i}+\beta\end{aligned}\xrightarrow{\text{展开}} \quad y _ i = \gamma * \frac { x _ i - \mu _ B }{ \sqrt { \sigma _ B ^ 2 + \epsilon }}+\beta \frac{\text{代入}}{x_i=w*x+b}\quad y=\gamma*\frac{w*x+b-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}+\beta
$$

$$
\begin{aligned}
&y=\gamma*\frac{w*x+b-\mu_{B}}{\sqrt{\sigma_{B}^{2}+\epsilon}}+\beta  \\
&y=\frac{\gamma^{*}w}{\sqrt{\sigma_{B}^{2}+\epsilon}}*x+\frac{\gamma}{\sqrt{\sigma_{B}^{2}+\epsilon}}(b-\mu_{B})+\beta  \\
&y=(\frac{\gamma*w}{\sqrt{\sigma_{B}^{2}+\epsilon}})*x+(\frac{\gamma}{\sqrt{\sigma_{B}^{2}+\epsilon}}(b-\mu_{B})+\beta)
\end{aligned}
$$

$$
\begin{array}{l}\widehat{w}=\dfrac{\gamma*w}{\sqrt{\sigma_B^2+\epsilon}}\\\widehat{b}=\dfrac{\gamma}{\sqrt{\sigma_B^2+\epsilon}}(b-\mu_B)+\beta\end{array}
$$

这两个参数值可以提前计算出来.

{% hint style="info" %}
很多模型经常会有很多 种类的activation function，比如 GELU, Swish, Mish等等，这些激活函数往往由于计算复杂很难加速，可以尝试改成ReLU看看精度的改变和性能的提升.
{% endhint %}

把上面的公式写成矩阵的形式：

$$
对于一个形状为C\times H\times W的特征图F,记归一化结果\hat{F},计算如下:
$$

$$
\begin{aligned}&\begin{pmatrix}\hat{F}_{1,i,j}\\\hat{F}_{2,i,j}\\\vdots\\\hat{F}_{C-1,i,j}\\\hat{F}_{C,i,j}\end{pmatrix}=\begin{pmatrix}\frac{\gamma_1}{\sqrt{\sigma_1^2+\epsilon}}&0&\cdots&0\\0&\frac{\gamma_2}{\sqrt{\sigma_2^2+\epsilon}}\\\vdots&&\ddots&\vdots\\&&\frac{\gamma_{C-1}}{\sqrt{\sigma_{C-1}^2+\epsilon}}&0\\0&\cdots&0&\frac{\gamma_C}{\sqrt{\sigma_C^2+\epsilon}}\end{pmatrix}\cdot\begin{pmatrix}F_{1,i,j}\\F_{2,i,j}\\\vdots\\F_{C-1,i,j}\\F_{C,i,j}\end{pmatrix}+\begin{pmatrix}\beta_1-\gamma_1\frac{\hat{\mu}_1}{\sqrt{\sigma_1^2+\epsilon}}\\\beta_2-\gamma_2\frac{\hat{\mu}_2}{\sqrt{\sigma_2^2+\epsilon}}\\\vdots\\\beta_{C-1}-\gamma_{C-1}\frac{\hat{\mu}_C^2+\epsilon}{\sqrt{\sigma_C^2+\epsilon}}\end{pmatrix}\end{aligned}
$$

代码如下：

```python
def fuse_conv_and_bn(conv, bn):
	#
	# init
	fusedconv = torch.nn.Conv2d(
		conv.in_channels,
		conv.out_channels,
		kernel_size=conv.kernel_size,
		stride=conv.stride,
		padding=conv.padding,
		bias=True
	)
	#
	# prepare filters
	w_conv = conv.weight.clone().view(conv.out_channels, -1)
	w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps+bn.running_var)))
	fusedconv.weight.copy_( torch.mm(w_bn, w_conv).view(fusedconv.weight.size()) )
	#
	# prepare spatial bias
	if conv.bias is not None:
		b_conv = conv.bias
	else:
		b_conv = torch.zeros( conv.weight.size(0) )
	b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
	fusedconv.bias.copy_( torch.matmul(w_bn, b_conv) + b_bn )
	#
	# we're done
	return fusedconv

if __name__ == "__main__":
    import torch
    import torchvision

    torch.set_grad_enabled(False)
    x = torch.randn(16, 3, 256, 256)
    rn18 = torchvision.models.resnet18(pretrained=True)
    rn18.eval()
    net = torch.nn.Sequential(
        rn18.conv1,
        rn18.bn1
    )
    y1 = net.forward(x)
    fusedconv = fuse_conv_and_bn(net[0], net[1])
    y2 = fusedconv.forward(x)
    d = (y1 - y2).norm().div(y1.norm()).item()
    print("error: %.8f" % d)
# error: 0.00000030
```
