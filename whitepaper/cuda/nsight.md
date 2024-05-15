# 🫣 Nsight system和Nsight compute

### Nsight systems

1. 对kernel执行和memory进行timeline分析，尝试寻找是否可以优化&#x20;

* 隐藏memory access&#x20;
* 多流调度&#x20;
* 删除冗长的memory access&#x20;
* 融合kernel减少kernel launch的overhead&#x20;
* &#x20;CPU与GPU的overlapping

<figure><img src="../../.gitbook/assets/图片 (15).png" alt=""><figcaption></figcaption></figure>

2. 分析DRAM以及PCIe带宽的使用率 ，可以从中分析到哪些带宽没有被充分利 用，从而进行优化

<figure><img src="../../.gitbook/assets/图片 (1) (1).png" alt=""><figcaption></figcaption></figure>

3. 分析SM中warp的占用率，可以从中知道一个SM中资源是否被用满

<figure><img src="../../.gitbook/assets/图片 (2) (1).png" alt=""><figcaption></figcaption></figure>

### Nsight compute

1. roofline analysis 对核函数进行roofline analysis， 并且根据baseline进行优化比较

<figure><img src="../../.gitbook/assets/图片 (3) (1).png" alt=""><figcaption></figcaption></figure>

2. occupancy analysis 对核函数的各个指标进 行估算一个warp的占用率的变化

<figure><img src="../../.gitbook/assets/图片 (5) (1).png" alt=""><figcaption></figcaption></figure>

3. memory bindwidth analysis 针对核函数中对各个memory的数 据传输的带宽进行分析，可以比较 好的理解memory架构

<figure><img src="../../.gitbook/assets/图片 (6) (1).png" alt=""><figcaption></figcaption></figure>

4. shared memory analysis 针对核函数中对shared memory访 问以及使用效率的分析

<figure><img src="../../.gitbook/assets/图片 (7) (1).png" alt=""><figcaption></figcaption></figure>

### 如何使用(推荐使用ssh远程)

* 方法一：在host端使用Nsight进行ssh远程profiling&#x20;
* 方法二：在remote端使用Nsight进行直接profiling&#x20;
* 方法三：在remote端通过CUI获取statistics之后传输到 host端进行查看
* 方法四：在remote端直接使用CUI进行分析

<figure><img src="../../.gitbook/assets/图片 (8) (1).png" alt=""><figcaption></figcaption></figure>

### 两者的不同

<mark style="color:red;">Nsight systems ：偏重于可视化application的整体的profiling以及各个细节指标</mark>，比如说&#x20;

* PCIe bindwidth&#x20;
* DRAM bindwidth&#x20;
* &#x20;SM Warp occupancy&#x20;
* &#x20;所有核函数的调度信息&#x20;
* 所有核函数的执行时间，以及占用整体时间的比例&#x20;
* 多个Stream之间的调度信息
* 同一个stream中的多个队列的调度信息&#x20;
* CPU和GPU之间的数据传输耗时&#x20;
* Application整体上的各个核函数以及操作的消耗时间排序&#x20;
* 捕捉同一个stream中的多个event&#x20;

整体上会提供一些比较全面的信息，我们一般会从这里得到很多信息进而进行优化

<mark style="color:red;">Nsight compute ：偏重于可视化每一个CUDA kernel的profiling以及各个细节指标</mark>，比如说&#x20;

* SM中计算吞吐量
* &#x20;L1 cache数据传输吞吐量&#x20;
* &#x20;L2 cache数据传输吞吐量&#x20;
* DRAM数据传输吞吐量&#x20;
* 当前核函数属于计算密集型还是访存密集型 ，Roofline model分析&#x20;
* &#x20;核函数中的L1 cache的cache hit几率, cache miss几率的多少&#x20;
* 核函数中各个代码部分的延迟 ，精确到代码部分进行highlight&#x20;
* 核函数的load bandwidth, store bandwith, load次数, store次数&#x20;
* L1 cache/shared memory, L2 cache, global memory中的memory access scheduling&#x20;
* 设置baseline，来进行核函数的优化前后的效率对比&#x20;

<mark style="color:red;">整体上能够得到一个针对某一个kernel的非常精确的profiling，源码级别的性能捕捉，以及优化推荐。</mark>

