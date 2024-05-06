# 🤠 A White Paper on Neural Network Deployment



<div align="center">

<img src="https://readme-typing-svg.demolab.com/?font=Reenie+Beanie&#x26;size=36&#x26;pause=3000&#x26;color=F7116E&#x26;background=FFFFFFFB&#x26;center=true&#x26;vCenter=true&#x26;random=false&#x26;width=435&#x26;lines=No+performce%2C+No+algorithms!" alt="">

</div>

### 写在前面的话

随着大模型的发展，如何开展绿色AI，并且在参数量那么大的情况下如何利用有限资源进行快速推理，这些内容都离不开高效的模型部署。虽然深度学习模型的研究与开发取得了长足的进步，但将这些模型部署到实际应用中却依然面临着诸多挑战。

本书的诞生源于对市场的观察与需求的反馈。笔者注意到，在许多深度学习资料中，对于模型部署的讨论相对较少，尤其是在针对特定硬件平台的情况下更是如此。因此，笔者决定撰写这本书，以填补市场上对于深度学习模型部署方面知识的空白。

本书的目标是向读者介绍如何有效地将深度学习模型部署到英伟达（NVIDIA）相关的硬件平台上。笔者将从基础概念出发，逐步引导你了解如何在实际项目中部署深度学习模型。无论你是刚刚入门深度学习领域，还是已经在实践中积累了一定经验，相信本书都将为你提供有价值的信息与指导。

<mark style="color:red;">⚠️特别强调，本书是开源的，我希望通过共享知识，促进深度学习模型部署领域的发展。我相信，只有在开放的环境下，知识才能不断地被传播、丰富和完善。知识理应共享！</mark>

本书github地址：[https://github.com/Phoenix8215/A-White-Paper-on-Neural-Network-Deployment](https://github.com/Phoenix8215/A-White-Paper-on-Neural-Network-Deployment)

### 主要内容

神经网络部署白皮书(以下简称白皮书)仍在全力更新中，目前只更新了大概30%，后面将添加更多进阶内容，以及大量的实战案例。由于笔者平时有科研任务，所以更新的进度可能不会太快。😶‍🌫️

白皮书的部分内容为韩博提供的资料，这是他的`github`主页[https://github.com/kalfazed](https://github.com/kalfazed)， [https://github.com/kalfazed/tensorrt\_starter](https://github.com/kalfazed/tensorrt\_starter)这个是韩博写的一个开源项目，后面也会对这部分代码进行解读。

若本书对您有所帮助，请在页面右上角点个 Star ⭐ 支持一下，谢谢\~\~\~。

白皮书主要分为以下几个板块，CUDA，ONNX，TensorRT，C++，实战内容，还有一些不属于以上各类的文章我就放到杂文里边去了。

* CUDA板块：主要介绍了GPU,CPU硬件的大致结构以及程序执行的流程，介绍了一些基本概念：CUDA线程，块，网格，CUDA流，CUDA事件

