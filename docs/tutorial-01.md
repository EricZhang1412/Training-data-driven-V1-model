# V1 GLIF 网络入门说明

本文档面向神经科学/脑科学基础较弱的读者，解释本工程中使用的 V1 网络是什么、它和论文中的模型有什么关系，以及它如何对应到当前 PyTorch 实现。

参考论文：

- Chen, Scherr, Maass, *A data-based large-scale model for primary visual cortex enables brain-like robust and versatile visual processing*, Science Advances, 2022.
- 该论文的 supplementary materials。

## 1. 这个工程在做什么

本工程的目标是把原始 TensorFlow 代码中的大规模 V1 脉冲神经网络迁移到 PyTorch，并让它可以作为可训练模型用于视觉任务。

整体数据流可以简化为：

```text
视觉输入 / movie
-> LGN 模型或 LGN-like 输入
-> V1 GLIF recurrent network
-> V1 spike activity
-> readout classifier
-> task decision
```

它不是普通 CNN，也不是普通 RNN，而是一个带有生物结构约束的大规模 spiking recurrent neural network。

## 2. 最基础的神经科学概念

### 2.1 神经元

普通人工神经网络中的一个单元通常可以理解为：

```text
输入 x
-> 加权求和
-> activation
-> 输出
```

生物神经元更像一个会积累电流的动态系统：

```text
输入电流
-> 膜电位 membrane voltage 变化
-> 超过阈值 threshold
-> 发放 spike
-> reset / refractory
```

在代码中：

```python
z  # spike, 0/1
v  # membrane voltage
```

`z=1` 表示某个神经元在当前时间步发放了一次 spike，`z=0` 表示没有发放。

### 2.2 Spike

Spike 是神经元之间传递信息的事件。脉冲神经网络不是只计算一次前向传播，而是在多个时间步上不断更新状态：

```text
t = 0 ms: update current, voltage, spike
t = 1 ms: update current, voltage, spike
t = 2 ms: update current, voltage, spike
...
```

因此模型输入和输出通常带有时间维度：

```python
input:  [batch, time, input_units]
output: [batch, time, v1_neurons]
```

### 2.3 突触

突触是神经元之间的连接。一个突触通常包含：

```text
source neuron
target neuron
weight
delay
receptor type
```

如果 source neuron 发放 spike，它会通过这个突触影响 target neuron 的输入电流。当前工程中，突触连接主要以 sparse matrix 的形式存储。

### 2.4 兴奋性和抑制性

V1 网络中有兴奋性神经元和抑制性神经元：

```text
excitatory neuron: 让下游神经元更容易发放 spike
inhibitory neuron: 让下游神经元更不容易发放 spike
```

论文和代码中常见的细胞类别包括：

```text
Exc / E   excitatory neurons
Pvalb     inhibitory neurons
Sst       inhibitory neurons
Htr3a     inhibitory neurons
```

一个重要的生物约束是 Dale's law：

```text
一个兴奋性神经元的输出突触应该保持兴奋性；
一个抑制性神经元的输出突触应该保持抑制性。
```

所以训练时不能让权重随意变号。这也是代码中需要 sign constraint 的原因。

## 3. V1 和 LGN 是什么

视觉通路可以粗略理解为：

```text
retina
-> LGN
-> V1
-> higher visual areas
```

本工程关注的是：

```text
LGN -> V1
```

### 3.1 LGN

LGN 可以先理解为 V1 前面的视觉滤波器组。论文中使用的 LGN 模型包含：

```text
17,400 filters
```

这些 filter 模拟若干类 LGN 细胞的响应，例如：

```text
sustained ON
sustained OFF
transient ON/OFF
transient OFF/ON
```

因此当前工程中常见的输入维度是：

```text
17400
```

也就是 LGN/input units 的数量。

### 3.2 V1

V1 是 primary visual cortex，即初级视觉皮层。论文使用的是 Allen/Billeh 的 mouse V1 point-neuron model。

这个 V1 模型包含：

```text
51,978 neurons
14M+ recurrent synapses
111 GLIF3 neuron types
5 cortical layers: L1, L2/3, L4, L5, L6
4 broad cell classes: Exc, Pvalb, Sst, Htr3a
```

这和当前 full-core 测试中看到的输出一致：

```text
Number of Neurons: 51978
Number of Synapses: 14441124
```

## 4. V1 网络的结构

这个网络不是全连接网络，也不是随机 RNN。它是空间结构化的 recurrent network。

主要特点：

- 神经元分布在不同 cortical layers 中。
- 连接概率依赖 pre/post 神经元的 layer 和 cell class。
- 连接概率还会随神经元之间的空间距离衰减。
- 近距离连接更多，远距离连接更少。
- 网络同时包含 feedforward stream 和 recurrent loops。

可以粗略理解为：

```text
LGN
-> L4
-> L2/3
-> L5/L6
```

但这不是单向链路。V1 内部还有大量 recurrent connections：

```text
V1 neurons -> V1 neurons
```

这使它具有时间动态和短时记忆能力，可以处理需要时间整合的任务，例如 evidence accumulation。

## 5. GLIF3 神经元模型

GLIF 是 Generalized Leaky Integrate-and-Fire。

普通 LIF 神经元可以粗略写成：

```text
输入电流让膜电位上升；
膜电位会随时间泄漏回静息电位；
超过阈值就发放 spike。
```

GLIF3 比普通 LIF 更复杂，它除了膜电位之外，还包含两个 after-spike current 变量，用来模拟更慢的生物过程，例如 spike-frequency adaptation。

在每个时间步，V1 模型大致执行：

```text
1. LGN/input spike 转换成 input current
2. recurrent spike 通过 delay buffer 转换成 recurrent current
3. 更新 synaptic current
4. 更新 after-spike currents
5. 更新 membrane voltage
6. 判断是否超过 threshold
7. 生成 spike
8. spike 进入 delay buffer，影响未来时间步
```

这就是当前 `BillehColumnTorch` 主要实现的内容。

## 6. 为什么 sparse tensor 的形状是 4 * n_neurons

full-core 模型中：

```text
n_neurons = 51978
4 * n_neurons = 207912
```

代码里常见的 sparse tensor 形状是：

```text
recurrent_sparse: [207912, 51978]
input_sparse:     [207912, 17400]
```

原因是每个 V1 神经元有 4 类 receptor / synaptic channel。连接并不是直接投影到 neuron，而是投影到：

```python
target_index = neuron_id * 4 + receptor_type
```

因此 sparse matrix 的行数是：

```text
4 * number_of_v1_neurons
```

## 7. 论文训练了什么

原始 Allen/Billeh 网络的结构和参数来自生物数据，但它并不是天然会完成特定视觉任务。Chen 等人的论文做的是：

```text
固定网络拓扑
不新增突触
不删除突触
保持 Dale's law
训练已有突触权重
```

主要被训练的权重包括：

```text
LGN -> V1 weights
V1 -> V1 recurrent weights
```

训练方法是 spiking neural network 中常用的 surrogate gradient / BPTT。因为 spike 是离散的 0/1 事件，普通反向传播无法直接穿过 spike 函数，所以需要使用 pseudo-derivative。

当前 PyTorch 工程中，后续可以通过类似下面的开关控制训练范围：

```text
train readout only
train input weights
train recurrent weights
```

现阶段工程已经验证了 readout-only 训练、单卡训练和 DDP 多卡训练。

## 8. Readout 是什么

V1 网络本身输出的是大量神经元随时间变化的 spike。要完成分类或决策任务，需要一个 readout 机制：

```text
V1 spikes
-> decision
```

论文没有主要采用从所有神经元读出的 global linear readout，因为这太强，可能掩盖 V1 网络本体的计算贡献。

论文采用了更生物化的 readout 方式：

```text
选择 L5 excitatory pyramidal neurons
每个任务输出对应一组 readout neurons
每组通常包含 30 个神经元
谁在 response window 中 spike 更多，就表示哪个输出
```

当前工程中为了先验证训练闭环，使用了更简单的 readout：

```python
spikes.mean(dim=1) -> Linear classifier
```

这个 readout 更像机器学习中的 global rate readout。它适合做工程验证，但如果要更贴近论文，后续应该实现 L5 readout pool。

## 9. 论文中的五个任务

Chen 等人的论文使用 V1 模型完成了 5 个视觉任务：

```text
fine orientation discrimination
image classification
visual change detection on natural images
visual change detection on gratings
evidence accumulation
```

对应原始代码中的任务名大致是：

```text
ori_diff
10class
garrett / natural image VCD
vcd_grating
evidence
```

当前工程暂时没有直接复现这些任务，因为缺少原始 TensorFlow 工程依赖的数据文件，例如：

```text
alternate_small_stimuli.pkl
many_small_stimuli.pkl
EA_LGN.h5
lgn_full_col_cells_3.csv
garrett_firing_rates.pkl
additive_noise.mat
```

因此目前采用 synthetic toy task 验证 PyTorch 训练管线是合理的中间步骤。

## 10. 当前 PyTorch 工程中的模块对应关系

当前工程可以理解为下面这条链路：

```text
Allen/Billeh SONATA data
-> convert2pkl.py
-> Chen-compatible network_dat.pkl / input_dat.pkl
-> load_sparse_torch.py
-> PyTorch sparse tensors
-> BillehColumnTorch
-> readout classifier
-> single-GPU training
-> DDP multi-GPU training
```

各模块作用：

```text
convert2pkl.py
```

将 Allen/Billeh SONATA 数据转换成旧 Chen 代码兼容的 pickle 格式。

```text
load_sparse_torch.py
```

加载 V1 神经元、recurrent synapses、LGN/input synapses，并转换成 PyTorch 可用的数据结构。

```text
models_torch.py / BillehColumnTorch
```

实现 V1 GLIF recurrent dynamics。

```text
multi_train_torch.py
```

单卡训练脚本。

```text
multi_train_torch_ddp.py
```

DDP 多卡训练脚本。

## 11. 一个有用的心智模型

如果暂时不熟悉神经科学，可以先把这个系统理解成一个生物约束版 RNN。

普通 RNN：

```text
x_t, h_t
-> h_{t+1}
-> output
```

本工程中的 V1 模型：

```text
LGN input_t
V1 spike_t
membrane voltage_t
synaptic current_t
after-spike current_t
refractory state_t
-> V1 state_{t+1}
-> spike output
```

区别是：普通 RNN 的 hidden state 是抽象向量，而这里的 hidden state 对应更接近生物神经元的状态变量。

## 12. 当前工程状态

目前已经完成：

- Allen/Billeh 数据转换到 Chen-compatible pickle 格式。
- PyTorch sparse loader。
- PyTorch 版 `BillehColumnTorch` forward。
- 小模型 `1000 neurons` CPU/CUDA 测试。
- full-core `51978 neurons` CUDA forward 测试。
- readout-only 单卡训练。
- synthetic toy task 学习验证。
- DDP 多卡训练脚本。

下一步推荐：

1. 接入一个真实但容易获得的数据集，例如 MNIST/CIFAR，并转换成 `[batch, time, 17400]` 输入。
2. 实现更接近论文的 L5 readout pool。
3. 再考虑训练 `LGN -> V1` 和 `V1 -> V1` sparse weights。
4. 如果能补齐缺失文件，再复现论文中的五个原始任务。

## 13. 一句话总结

这个网络是一个基于 Allen/Billeh 鼠 V1 数据构建的大规模 GLIF3 脉冲神经网络。Chen 等人的论文把它作为一个 biologically realistic recurrent backbone，用 LGN 输入驱动它，再通过训练已有突触权重，让它完成多个视觉任务。本工程当前正在将这条 TensorFlow 主线迁移到 PyTorch，并已经打通了核心 forward、训练和 DDP 多卡流程。
