
---

# FedCausal 项目背景知识文档 v0.2

## 项目名称

**FedCausal**

## 论文标题

**Beyond Data Augmentation: Causal Frequency Intervention for Unified Robust Federated Learning**

中文标题：

**超越数据增强：面向统一鲁棒联邦学习的因果频域干预**

---

# 1. Introduction

## 1.1 研究背景

联邦学习允许多个客户端在不共享原始数据的情况下协同训练模型。但在真实部署中，联邦学习不仅面临传统的 Non-IID 数据问题，还同时面临三类更复杂的挑战：

1. **模型异构**：不同客户端可能由于算力、硬件和部署场景不同，使用完全不同的模型架构。
2. **数据损坏 / Common Corruption**：客户端数据可能受到传感器噪声、压缩伪影、天气变化、模糊、像素化等自然或人为损坏。
3. **恶意攻击 / 低质量客户端**：部分客户端可能上传错误、投毒、放大或低质量知识，导致全局模型被污染。

现有多数联邦学习方法默认客户端数据较干净，且模型结构同构，因此难以直接应对上述三重挑战。RAHFL 明确指出，在真实异构联邦环境中，客户端可能具有不同模型结构，同时私有数据还可能受到随机噪声、压缩伪影、环境因素等 corruption 影响，这会显著削弱整个联邦系统性能。

数据噪声问题并不是边缘现象。关于 FL 数据噪声影响的系统研究指出，真实边缘设备数据往往是 “in the wild” 的，会受到硬件限制、环境因素和人为错误影响；相比集中式学习，FL 对数据噪声更加敏感。其根本原因包括：服务端无法直接观察数据质量、本地训练会放大噪声影响、Non-IID 会进一步加剧噪声集中带来的性能下降。

因此，FedCausal 的核心问题不是单纯提升 clean accuracy，而是要回答：

> 在模型异构、数据损坏和潜在攻击同时存在时，是否可以通过频域因果干预获得更稳定、更轻量、更可解释的联邦鲁棒性？

---

## 1.2 与现有方法的关系

FedCausal 并不是只对比 FedProto。它需要放在以下研究脉络中理解。

---

### 1.2.1 FedAvg / FedProx 类同构联邦方法

FedAvg、FedProx 等经典方法通过聚合模型参数实现协作。这类方法的核心前提是：

* 客户端模型结构一致；
* 参数空间可以直接平均；
* 服务端可以聚合同构模型权重。

但在模型异构场景中，不同客户端可能使用 ResNet、MobileNet、ShuffleNet 或 ViT，参数维度和网络结构完全不同，因此 FedAvg / FedProx 无法直接适用。

FedCausal 的设计从一开始就不聚合模型参数，而是聚合：

* 频域掩码 (M_i)
* 类别原型 (P_i)

这使得它天然支持模型异构。

---

### 1.2.2 FedProto 类原型联邦方法

FedProto 通过类别原型而非模型参数实现异构联邦通信，是 FedCausal 的重要基础。客户端将样本映射到统一维度特征空间，然后计算每个类别的 prototype，服务端聚合 prototype 后再下发。

FedProto 的优势是：

* 支持模型异构；
* 通信成本低；
* 不需要上传模型参数。

但 FedProto 的问题也很明显：

* 如果客户端数据受到 corruption，本地原型会被噪声污染；
* 如果客户端遭受标签翻转或 prototype scaling，服务端缺乏鲁棒机制；
* 原型只描述语义结果，没有解释“输入中哪些因素是因果的”。

因此，FedCausal 可以被看成在 FedProto 之上加入了三层补强：

1. **输入端补强**：频域因果掩码过滤 corrupted frequency；
2. **表征端补强**：反事实频域干预学习不变特征；
3. **服务端补强**：基于 mask + prototype 的能量式鲁棒聚合。

---

### 1.2.3 FedMD / FedDF / RHFL / FCCL 等知识蒸馏或公共数据方法

RAHFL 论文中总结了大量异构联邦方法，例如 FedMD、FedDF、RHFL、KT-pFL、FCCL、FedProto 等。FedMD 使用公共数据上的平均类别分数进行通信，FedDF 使用无标签或合成数据进行模型融合，RHFL 使用公共无关数据对齐反馈输出，KT-pFL 使用知识系数矩阵做个性化知识迁移，FCCL 使用相关矩阵进行协作学习。

这些方法说明：**模型异构 FL 的主流路线不是参数聚合，而是知识层通信**。

FedCausal 延续这一思想，但区别在于：

* 不依赖公共数据作为通信媒介；
* 不仅传递语义知识，还传递频域物理机制；
* 不只关注模型异构，还同时关注 common corruption 和攻击鲁棒性。

---

### 1.2.4 RAHFL / AugHFL：强相关基线

RAHFL 是与你的研究最接近的工作之一。它研究的是 **model heterogeneous + data corrupted clients** 的鲁棒异构联邦学习问题。其核心做法包括：

1. 使用随机混合数据增强和一致性损失；
2. 提出 Diversity-enhanced Supervised Contrastive Learning；
3. 设计 AsymHFL，让客户端选择性地从更可靠的客户端单向学习，从而避免吸收低质量知识。

RAHFL 的动机与你非常接近：在异构模型下，corrupted samples 更难分类，不同模型对同一 corrupted image 的预测可能显著不一致，协作时会错误强化不可靠客户端的信息。

但 FedCausal 与 RAHFL 的关键区别是：

| 维度      | RAHFL                         | FedCausal                      |
| ------- | ----------------------------- | ------------------------------ |
| 本地鲁棒性来源 | mixed data augmentation + DCL | 频域因果干预                         |
| 是否依赖增强  | 是                             | 尽量不依赖传统图像增强                    |
| 通信方式    | 公共数据输出 / 非对称学习                | mask + prototype               |
| 核心机制    | 学习多样增强模式                      | 解耦 causal / spurious frequency |
| 防御逻辑    | 避免向低质量客户端学习                   | 用能量评分压低异常客户端权重                 |

RAHFL 是必须重点对比的基线，不应只和 FedProto 比。RAHFL 论文的实验中也比较了 FedMD、RHFL、FCCL、FedProto、AugHFL 等方法，并显示 RAHFL 在 corrupted heterogeneous FL 场景下具有明显优势。

---

### 1.2.5 FedERL：资源受限 Common Corruption 鲁棒 FL

FedERL 关注的是另一个与你高度相关的问题：**在客户端资源受限时，如何提升 FL 对 common corruption 的鲁棒性**。

FedERL 指出，传统 robust training 通常需要在客户端做数据增强和鲁棒训练，这会带来巨大的时间和能耗开销；因此它把鲁棒训练转移到资源更充足的服务端，通过 DART 使用公共无标签数据增强全局模型鲁棒性，从而实现客户端零额外鲁棒训练开销。

FedERL 的贡献对 FedCausal 有两点启发：

1. **鲁棒性不能只看 accuracy，还要看客户端开销**；
2. **避免把重度鲁棒训练压力放到客户端，是一个重要卖点**。

FedCausal 与 FedERL 的区别是：

| 维度          | FedERL          | FedCausal                   |
| ----------- | --------------- | --------------------------- |
| 场景          | 同构 FL + 客户端资源约束 | 异构 FL + corruption + attack |
| 鲁棒来源        | 服务端 DART        | 客户端轻量频域干预 + 服务端鲁棒聚合         |
| 是否需要公共无标签数据 | 需要              | MVP 阶段不需要                   |
| 是否支持模型异构    | 不是核心            | 是核心目标                       |
| 是否处理攻击      | 不是主要目标          | 是扩展目标                       |

FedERL 实验强调在时间和能耗预算下，FedERL 比 CleanFL 和 RobustFL 更能平衡 clean accuracy 与 robust accuracy。 因此，FedCausal 的实验也应该加入 communication cost、client-side overhead、训练时间等指标，而不能只看准确率。

---

### 1.2.6 Data Noise in FL：为什么 FedCausal 的问题是必要的

《Understanding the Impact of Data Noise in Federated Learning》提供了 FedCausal 的重要问题动机。该研究发现，FL 比集中式学习更容易受到数据噪声影响，其原因包括：

1. 服务端看不到客户端真实数据质量；
2. 本地训练会过拟合 noisy data，产生偏离的本地更新；
3. 服务端盲目聚合这些 divergent updates，导致全局模型不稳定；
4. Non-IID 会进一步放大噪声影响。

该论文还发现，当客户端数量更多、每个客户端数据更少时，本地更新更容易受噪声影响；当客户端采样率降低时，FL 对数据噪声更脆弱。

这说明 FedCausal 的 energy-based aggregation 很有必要：服务端不能盲目平均所有客户端上传的知识，而应该根据客户端上传的 mask 和 prototype 判断其可靠性。

---

## 1.3 FedCausal 的核心定位

FedCausal 的定位不是“另一个 FedProto 变体”，而是：

> 面向模型异构、数据损坏和攻击风险的频域因果鲁棒联邦框架。

它要同时对标三类方法：

1. **异构 FL 方法**：FedProto、FedMD、FedDF、RHFL、FCCL；
2. **corruption robust FL 方法**：AugHFL、RAHFL、FedERL；
3. **鲁棒聚合 / 攻击防御方法**：Median、Trimmed Mean、Krum、prototype robust aggregation。

---

# 2. Related Work

## 2.1 Heterogeneous Federated Learning

异构联邦学习研究客户端模型结构不同情况下的协作问题。典型方法包括：

* FedMD：基于公共数据上的类别分数进行蒸馏；
* FedDF：基于无标签或合成数据进行模型融合；
* RHFL：使用公共无关数据做 heterogeneous model feedback alignment；
* KT-pFL：使用知识系数矩阵进行个性化知识迁移；
* FCCL：使用特征相关矩阵进行协作学习；
* FedProto：通过类别原型进行跨模型通信。

这些方法大多假设训练数据相对干净。然而在真实部署中，客户端数据可能受到 corruption，导致本地知识本身不可靠。因此，仅做语义层对齐是不够的。

FedCausal 的改进点是：

* 不仅上传语义原型；
* 还上传频域因果 mask；
* 通过 mask 判断客户端是否学到了异常频率机制；
* 通过 prototype 判断客户端语义是否偏离全局共识。

---

## 2.2 Robust Heterogeneous FL under Corrupted Clients

RAHFL 是与 FedCausal 最接近的基线。它明确研究 corrupted clients 下的 robust asymmetric heterogeneous FL，并指出数据 corruption 会使本地模型学习错误模式，进而在协作阶段传播低质量知识。

RAHFL 的两个核心模块是：

1. **DCL / Diversity-enhanced Supervised Contrastive Learning**：使用复杂增强样本进行监督对比学习；
2. **AsymHFL**：允许客户端选择性单向学习，避免吸收表现较差客户端的低质量信息。

FedCausal 与 RAHFL 的主要区别：

* RAHFL 的鲁棒性主要来自 mixed-data augmentation；
* FedCausal 的鲁棒性来自 frequency-domain causal intervention；
* RAHFL 通过选择性学习避免低质量知识传播；
* FedCausal 通过 energy-based trust weight 直接降低异常客户端聚合权重。

因此，RAHFL 应作为 FedCausal 的强基线，而不是只比较 FedProto。

---

## 2.3 Efficient Robust FL for Common Corruptions

FedERL 研究的是在客户端时间和能耗受限条件下，如何获得 common corruption robustness。它指出传统 robust FL 会让客户端承担重度数据增强和鲁棒训练开销，而 FedERL 通过在服务端执行 DART，将鲁棒训练压力从客户端转移到服务端，实现客户端零额外计算开销。

FedCausal 需要吸收这一点：实验中除了 accuracy，还应报告：

* client-side FLOPs；
* 每轮训练时间；
* 通信量；
* 是否需要公共数据；
* 是否增加客户端显存压力。

FedCausal 的优势应该表述为：

> 相比重度增强式鲁棒训练，FedCausal 只增加轻量 FFT mask 与 prototype 通信，尽量避免在客户端进行多分支增强训练。

---

## 2.4 Data Noise Sensitivity in FL

数据噪声影响分析论文指出，FL 对数据噪声的脆弱性来自联邦流程本身：本地噪声会导致 divergent updates，服务端盲目聚合后会进一步放大全局不稳定性。

这直接支持 FedCausal 的两个设计：

1. **本地端**：用 frequency mask 和 intervention 降低 noisy data 对本地特征的污染；
2. **服务端**：用 energy-based aggregation 避免 blind aggregation。

---

## 2.5 Robust Aggregation and Byzantine Defense

传统鲁棒聚合方法包括 Median、Trimmed Mean、Krum 等。这些方法通常在参数或梯度空间检测异常。

但 FedCausal 面对的是模型异构场景，不同客户端模型参数不可比较。因此 FedCausal 在共享的低维空间中做鲁棒检测：

* physical space：(M_i)
* semantic space：(P_i)

这使得防御机制天然适配模型异构。

---

# 3. Problem Formulation

## 3.1 联邦系统

设联邦系统包含 (N) 个客户端：

[
\mathcal{C}={1,\dots,N}
]

每个客户端 (i) 拥有本地私有数据集：

[
\mathcal{D}*i={(x_j,y_j)}*{j=1}^{n_i}
]

数据分布为：

[
\mathcal{D}_i \sim P_i(X,Y)
]

不同客户端之间可能存在：

* label distribution skew；
* quantity skew；
* corruption type skew；
* corruption severity skew。

---

## 3.2 模型异构

每个客户端拥有不同本地模型：

[
f_i = c_i \circ g_i \circ h_i
]

其中：

* (h_i)：异构 backbone；
* (g_i)：projector；
* (c_i)：classifier。

要求所有客户端 projector 输出统一维度：

[
g_i(h_i(x)) \in \mathbb{R}^{D}
]

这样可以计算统一原型：

[
P_i \in \mathbb{R}^{K \times D}
]

---

## 3.3 数据损坏

客户端数据可能为 clean，也可能被 corruption function (\kappa) 污染：

[
\tilde{x} = \kappa(x, s)
]

其中 (s) 表示 severity。

测试阶段使用 CIFAR-10-C / CIFAR-100-C 中的 common corruptions，包括：

* Gaussian Noise；
* Shot Noise；
* Motion Blur；
* Defocus Blur；
* Fog；
* Snow；
* JPEG Compression；
* Pixelate。

---

## 3.4 攻击设置

初步实验关注两类攻击：

### Label Flipping

[
y \rightarrow (y+1)\mod K
]

### Prototype Scaling

[
P_i^{mal} = \rho P_i
]

其中 (\rho \in {10,100})。

后续扩展：

* backdoor trigger；
* frequency backdoor；
* representation collapse；
* adaptive prototype poisoning。

---

## 3.5 优化目标

目标是在模型异构、数据 Non-IID、未知 corruption 和恶意攻击条件下最小化 OOD 测试风险：

[
\min_f \mathbb{E}*{(X,Y)\sim P*{test}} \ell(f(X),Y)
]

---

# 4. Method

## 4.1 Causal Frequency Decomposition

FedCausal 将输入图像转换到频域：

[
Z=\mathcal{F}(X)
]

每个客户端维护可学习频域 mask：

[
M_i=\sigma(\phi_i)
]

其中：

[
M_i \in [0,1]^{C\times H\times W}
]

频域分解为：

[
Z_c = M_i \odot Z
]

[
Z_s = (1-M_i)\odot Z
]

其中：

* (Z_c)：因果频率；
* (Z_s)：环境 / spurious 频率。

然后：

[
X_c=\mathcal{F}^{-1}(Z_c)
]

分类器使用 (X_c) 进行训练，而不是原始 (X)。

---

## 4.2 Representation-Level Intervention

FedCausal 不在图像空间做增强，而是在频域进行反事实干预。

对 batch 中样本 (A)，选择标签不同的样本 (B)：

[
Y_B \neq Y_A
]

构造：

[
\tilde{Z}^{cf}*A = Z*{c,A} + Z_{s,B}
]

[
\tilde{X}^{cf}_A = \mathcal{F}^{-1}(\tilde{Z}^{cf}_A)
]

不变性损失为：

[
\mathcal{L}_{inv}
=================

\left|
\hat{f}(\tilde{X}^{cf}_A)
-------------------------

\text{sg}(\hat{f}(X_{c,A}))
\right|_2^2
]

其中 (\hat{f}) 表示 normalize 后的 feature，(\text{sg}) 表示 stop-gradient。

---

## 4.3 Prototype-based Heterogeneous Alignment

每个客户端计算本地类别原型：

[
P_{i,k}
=======

\frac{1}{|\mathcal{D}*{i,k}|}
\sum*{(x,y)\in\mathcal{D}_{i,k}}
f_i(x)
]

使用 prototype-level supervised contrastive loss：

[
\mathcal{L}_{SCL}
=================

-\frac{1}{B}
\sum_{a=1}^{B}
\log
\frac{
\exp(\cos(f_a,P_{g,y_a})/\tau_s)
}{
\sum_{k=1}^{K}
\exp(\cos(f_a,P_{g,k})/\tau_s)
}
]

---

## 4.4 Energy-based Robust Aggregation

初步版本使用两项能量：

[
E_i = d_i^{mask} + \beta d_i^{proto}
]

其中：

[
d_i^{mask}=|M_i-M_{med}|_1
]

[
d_i^{proto}=1-\cos(P_i,P_{med})
]

聚合权重：

[
\alpha_i=
\frac{\exp(-E_i/\tau)}
{\sum_j \exp(-E_j/\tau)}
]

服务端更新：

[
M_g=\sum_i \alpha_i M_i
]

[
P_g=\sum_i \alpha_i P_i
]

---

# 5. Experiments

## 5.1 Experimental Setup

初步实验使用：

* CIFAR-10；
* CIFAR-10-C subset；
* 10 个客户端；
* Dirichlet (\alpha=0.3)，之后加 (\alpha=0.1)；
* 模型混合：CNN-small、ResNet18、MobileNetV2。

---

## 5.2 Baselines

FedCausal 不应只比较 FedProto，至少应包含：

### 基础联邦类

* Local Only；
* FedAvg，同构设置下；
* FedProx，同构设置下。

### 异构联邦类

* FedProto；
* FedMD / FedDF，可选；
* RHFL / FCCL，可选。

### corruption robust heterogeneous FL

* AugHFL；
* RAHFL，强基线；
* FedERL 思路对应的 server-side robust training，可作为后续扩展基线。

### 鲁棒聚合类

* Median；
* Trimmed Mean；
* Krum；
* simple prototype median；
* FedCausal energy aggregation。

---

## 5.3 Heterogeneous FL

目标：

验证模型异构 + Non-IID 下，FedCausal 是否优于 FedProto、FedMD/FedDF/RHFL/FCCL/RAHFL 等可实现基线。

指标：

* clean accuracy；
* client average accuracy；
* client accuracy std；
* communication cost。

---

## 5.4 Robustness to Common Corruptions

训练：

* clean CIFAR-10。

测试：

* CIFAR-10-C。

初步 corruption：

* Gaussian Noise；
* Shot Noise；
* Motion Blur；
* Fog；
* JPEG Compression。

指标：

[
Drop=Acc_{clean}-Acc_{corr}
]

[
mCA = \frac{1}{|\mathcal{C}_{corr}|}\sum_c Acc_c
]

---

## 5.5 Robustness to Attacks

攻击：

* label flipping；
* prototype scaling；
* later：backdoor / frequency backdoor。

指标：

* accuracy under attack；
* malicious average weight；
* benign average weight；
* (\alpha_{mal}/\alpha_{benign})。

---

## 5.6 Ablation Study

消融：

* w/o Mask；
* w/o (L_{inv})；
* w/o (L_{SCL})；
* w/o Energy；
* w/o sparse regularization；
* Full FedCausal-MVP。

---

# 6. Limitations

FedCausal 仍有边界：

1. 如果 corruption 与语义频率高度重叠，频域 mask 难以分离；
2. 如果 (I(Z_s;Y)) 很大，spurious 可能被误判为 causal；
3. 如果 batch 中类别太少，cross-sample different-label swap 可能不稳定；
4. MVP 版本只处理 label flipping 和 prototype scaling，不能声称防御所有攻击；
5. 与 FedERL 相比，FedCausal 仍需评估客户端 FFT 计算带来的额外开销。

---

# 7. Conclusion

FedCausal 是一个面向异构、损坏和攻击场景的频域因果联邦学习框架。相比 FedProto，它不仅对齐语义原型，还通过频域 mask 过滤输入噪声；相比 RAHFL，它不依赖重度 mixed augmentation，而是通过频域干预学习不变表征；相比 FedERL，它不把鲁棒性完全转移到服务端，而是在客户端进行轻量频域因果过滤，并在服务端进行能量式鲁棒聚合。

请你认真阅读完项目背景，我现在想分步实现这个框架，并且在kaggle平台来跑实验，
你现在是我的科研代码助手。我要在 Kaggle Notebook 上实现并训练一个联邦学习研究项目，项目名为 FedCausal。

论文标题：
Beyond Data Augmentation: Causal Frequency Intervention for Unified Robust Federated Learning

研究目标：
验证 FedCausal 是否能在以下三种挑战下优于相关基线：
1. 模型异构 + Non-IID 数据；
2. Common Corruption / corrupted clients；
3. label flipping / prototype scaling 等简单攻击。

FedCausal 的核心思想：
不在图像空间做重度数据增强，而是在频域中做 causal frequency decomposition 和 representation-level intervention。
具体来说：
- 使用 FFT 将图像转到频域；
- 使用可学习 mask M 将频率分解为 causal frequency 和 spurious frequency；
- 使用 cross-sample spurious swap 构造反事实频谱；
- 使用 L_inv 强制模型学习对 spurious frequency 不敏感的特征；
- 使用 prototype-level supervised contrastive loss 做异构语义对齐；
- 服务端使用 mask + prototype 的 energy score 做鲁棒聚合。

Kaggle 环境约束：
1. 所有输入数据从 /kaggle/input/ 读取。
2. 所有输出保存到 /kaggle/working/。
3. 代码必须适合 Kaggle Notebook 分 cell 执行。
4. 不能依赖复杂的交互式命令行。
5. 所有实验参数用 Python dict 或 YAML 文件管理。
6. 训练时间要可控，默认先跑小规模 MVP。
7. 默认使用 CIFAR-10。
8. CIFAR-10-C 如果存在于 /kaggle/input/cifar10-c/，则加载；如果不存在，需要优雅跳过并提示用户上传。
9. 每一步都要保存 checkpoint 和 CSV 结果，防止 Kaggle 会话中断。
10. 代码必须模块化，但也要能在 Notebook 中直接运行。


