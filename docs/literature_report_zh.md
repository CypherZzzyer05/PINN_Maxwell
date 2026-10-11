# 麦克斯韦物理神经网络用于波导光学：调研与原型报告

调研日期：2026-10-11。目标：给定材料分布、光源和波长，计算传播后的横截面复电场与复磁场，最终支持扫参与逆设计。

## 1. 结论

这个方向有直接的论文基础，适合开展。项目需要区分三个问题：**波导支持哪些模态**、**光源激发哪些模态**、**这些模态经过纵向变化的器件如何耦合、反射与辐射**。模态本征值求解并不等于任意光路传播；只有电场模方也不足以满足 E/H 场分布的目标。

本项目从频域、分区域、界面感知的 Maxwell PINN 起步。第一版实现无损多层平板波导 TE 基模，独立验证后重建单模传播的复场。针对一族器件的快速复用，后续应研究条件网络或物理神经算子。本报告中的路线选择与推导属于工程判断；论文结论在对应条目注明来源。部分论文只有可访问摘要，不声称复现其完整方法。

## 2. 核心论文与阅读判断

| 文献 | 已核对内容、与本项目的关系 | 阅读范围 |
| --- | --- | --- |
| Raissi, Perdikaris, Karniadakis, 2019. *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations*. JCP 378, 686–707. [DOI](https://doi.org/10.1016/j.jcp.2018.10.045) | 将物理微分方程约束纳入神经训练；项目借鉴残差、边界与观测约束的组合。通用 PINN 的效果不能直接视为光学精度保证。 | 出版社摘要、元数据 |
| Lim, Psaltis, 2022；预印本 2021. *MaxwellNet: Physics-driven deep neural network training based on Maxwell's equations*. APL Photonics. [预印本](https://arxiv.org/abs/2107.06164)、[DOI](https://doi.org/10.1063/5.0071616) | 输入材料分布、输出电场，以 Maxwell 残差训练，展示微透镜散射与逆设计。正文采用 Yee 网格高阶有限差分与 PML。它是材料到场的映射；当前原型是每个器件独立训练的坐标网络。 | 10 页公开全文 |
| Piao, Gu, Wang, Qin, 2024；预印本 2023. *A Domain-adaptive Physics-informed Neural Network for Inverse Problems of Maxwell's Equations in Heterogeneous Media*. IEEE AWPL 23(10), 2905–2909. [预印本](https://arxiv.org/abs/2308.06436)、[DOI](https://doi.org/10.1109/LAWP.2024.3413851) | 介质分域，把界面位置与电磁连续条件纳入训练。项目借鉴分域及界面损失，使用已知固定界面，未实现该论文的界面反演。 | 公开正文、出版元数据 |
| Xu, Zhong, Wang, Wei, Wang, Cheng, 2025. *High precision, full-vector optical mode solving in waveguides via fourth-order derivative physics-informed neural networks*. Optics Express 33(18), 38317–38328. [DOI](https://doi.org/10.1364/OE.571156)、[摘要](https://pubmed.ncbi.nlm.nih.gov/40984242/) | 全矢量波导模态，联合求切向 E/H 与传播常数，使用四阶导数损失和初始化策略。与最终目标直接相关。原型先采用一阶方程，高阶微分训练成本需要另行对照。 | 摘要、元数据；期刊正文访问受限 |
| Hares, El-Azab, Obayya, 2026. *A stable physics-guided neural networks approach for electromagnetic problems*. JCP 548, 114568. [DOI](https://doi.org/10.1016/j.jcp.2025.114568) | 针对二维介质波导的模态稳定性，关注损失极小值与正确本征值的关系。项目必须结合归一化、模式选择和独立参考，不能仅以低残差判断正确。 | 出版社摘要 |
| Hares, El-Azab, Obayya, 2026-10-06. *Full-vectorial physics-informed neural operator for forward and inverse waveguide modeling*. Engineering with Computers 42, 187. [公开全文](https://link.springer.com/article/10.1007/s00366-026-02413-2) | 双输出头同时预测有效折射率与场，构造性本征值约束与短期偏置用于稳定模式选择。作者在其测试范围报告约 1–2% 相对误差及 GPU 推理加速，不能直接套用到本项目。是由单器件 PINN 扩展到全矢量参数化算子的重点。 | 公开正文 |
| Wang, Wang, Perdikaris, 2021；预印本 2020. *On the eigenvector bias of Fourier feature networks: From regression to solving multi-scale PDEs with physics-informed neural networks*. CMAME 384, 113938. [预印本](https://arxiv.org/abs/2012.10047) | 分析高频、多尺度学习困难，采用多尺度 Fourier 特征。对光传播载波学习有参考价值；本项目可对照频率编码与载波提取，但尚未验证这些改进。 | 摘要、作者出版列表 |
| Krishnapriyan et al., 2021. *Characterizing possible failure modes in physics-informed neural networks*. NeurIPS. [论文](https://arxiv.org/abs/2109.01050) | 展示 PINN 优化失效并讨论课程学习、分步求解。项目应记录失败、训练成本与独立场误差。论文中的其它 PDE 测试不构成 Maxwell 精度保证。 | 公开摘要 |

检索还发现 [2026 年二维介质散射预印本](https://arxiv.org/abs/2607.27349)，可作为传播散射方向补充阅读。本文以以上八篇为主要依据，覆盖直接光学应用、界面、全矢量波导、算子与训练困难；属于项目导向调研，未声称穷尽领域。

## 3. 物理任务应如何定义

时间约定 exp(-iωt)，无外加电流的传播区域满足：

\[
\nabla\times\mathbf E=i\omega\mu\mathbf H,\quad
\nabla\times\mathbf H=-i\omega\epsilon\mathbf E,
\]
\[
\nabla\cdot(\epsilon\mathbf E)=0,\quad\nabla\cdot(\mu\mathbf H)=0.
\]

完整器件还需要光源/端口、开放边界、介质界面条件。无自由面电荷、面电流时，切向 E/H 连续，法向 D/B 连续。高反差二维截面通常存在纵向分量，不能普遍令 Ez=Hz=0；线偏振入射也不保证复杂器件内部一直只有单个电场分量。

传播方向设为 z，输出面为 z=L，需保存 E(x,y,L)、H(x,y,L) 的复分量。功率流密度为 S=Re(E×H*)/2。

| 问题 | 未知量 | 解决什么 | 尚需什么 |
| --- | --- | --- | --- |
| 不变波导模态 | e(x,y)、h(x,y)、β | 端口、有效折射率、约束能力 | 激励系数与纵向变化 |
| 器件传播/散射 | E(x,y,z)、H(x,y,z) | 反射、辐射与输出面 | 单器件训练成本 |
| 条件网络/神经算子 | 材料、波长、激励到场映射 | 快速扫参与设计 | 覆盖范围、泛化与总成本 |

## 4. 第一版的方程与实现

假设材料 n=n(x)、μr=1，沿 y,z 不变，折射率为正实数。TE 电场沿 y，定义：

\[
E_y=e(x)e^{i\beta z},\quad n_{eff}=\beta/k_0,\quad q=e'/k_0,\quad k_0=2\pi/\lambda_0.
\]

Maxwell 方程化为一阶系统：

\[
e'=k_0q,\qquad q'=k_0(n_{eff}^2-n^2)e.
\]

每个有限内部介质层用独立两输出 MLP 学习 e,q。一阶自动微分计算残差，避免对材料阶跃直接求导。从同一组解重建：

\[
H_x=-n_{eff}e\,e^{i\beta z}/Z_0,\qquad H_z=-i q\,e^{i\beta z}/Z_0.
\]

Hz 通常非零且与 e 有四分之一周期相差。Ex、Ez、Hy 在此 TE 几何下为零；这是该问题的精确约化，并非适用于所有波导的标量近似。

左右外部介质延伸到无穷远，α=k0√(neff²−nclad²)。内部端点条件为 q_left=+α_left e_left/k0、q_right=−α_right e_right/k0，外部采用精确衰减指数。配置最外侧坐标仅决定导出范围，不是金属壁或 PML。

neff 用 sigmoid 限制在 max(n_left,n_right) 与 max(n_interior) 之间。e 用指数参数化为正，限定本一维问题的无节点基模；e(anchor)=1 消除零解与幅值不确定性。此约束不能直接用于有节点高阶模态。多个有限层通过 e,q 连续约束连接；对 μr=1 的 TE 问题，这满足切向 E/H 及必要法向磁感应连续条件。

损失为 L=L_PDE+10L_interface+10L_boundary+10L_anchor。先 Adam 随机采样训练，后固定点 L-BFGS；独立 513 点/层用于训练后残差检查。位置以 μm 为单位，k0、β 为 rad/μm；E 导出为 V/m，H 为 A/m。Z0 用于场尺度转换；横向功率积分乘 10^-6，得到每单位不变 y 长度的功率 W/m。

## 5. 光源与传播范围

模式光源激励学习到的基模。在无损直波导中只改变相位，|E|² 与总功率不随 z 改变，这符合单模物理。

高斯光源使用 exp[-(x/w)²] 的电场轮廓，计算对基模的 L2 投影并传播保留分量。报告的保留比例是电场形状 L2 比例，**不是严格端口耦合效率**。未求解辐射、其它模态及端面入射反射，不能把示例当作完整高斯束经过任意材料的解。

此处“多层”指横向 x 的芯层、包覆层和包层。沿 z 的材料突变、弯曲、分束、透镜、二维截面与三维光路均属于后续任务。

## 6. 验证设计

训练结束前不调用参考本征值与参考场。参考不进入损失、neff 初始化或场初始化。对称三层波导使用解析色散方程及独立多层传输矩阵；包覆波导使用传输矩阵根搜索。传输矩阵在大厚度/大层数下有数值稳定性限制，扩展时需改散射矩阵并增加 FEM/FDTD 对照。

验收包括：Maxwell 旋度符号、H 单位、复相位、正向 Poynting 功率；解析场代入网络残差验证 PDE/边界/归一化；独立参考一致性；训练后 neff 相对误差 <0.1%、E 相对 L2 <1%、H 相对 L2 <2%；独立点 PDE、界面、边界与锚点损失均 <10^-5；输出面复 E/H 的相对 L2 误差 <2%。最后一项用于检查传播常数误差引起的累计相位偏差。

实测数据见 [validation_results.md](validation_results.md)。上述阈值属于项目选择，不是通用文献精度保证。传播功率恒定由单模公式保证，不能单独证明网络准确；独立场与本征值误差才是主要证据。输出包含 CSV、复场 NPZ、训练历史与环境记录。

## 7. 后续路线与 Git 管理

| 阶段 | 目标与验收 | 建议独立分支/worktree |
| --- | --- | --- |
| 当前基线 | 一维 TE 多层基模与复 E/H；解析/传输矩阵及 Maxwell 检查 | codex/maxwell-pinn-prototype |
| TM、多模态 | 法向分量跳变、正交约束；色散和多模谱与参考一致 | codex/tm-multimode |
| 二维全矢量截面 | Ex/Ey/Ez、Hx/Hy/Hz、neff；矩形/脊形波导与 FEM/FDTD 对照 | codex/vector-cross-section |
| 纵向传播散射 | n(x,z)、端口、反射与辐射；阶跃、渐变、弯曲，R+T+损耗闭合 | codex/propagation-scattering |
| 参数化算子 | 材料图、λ、光源到场；未见参数误差与训练+推理成本 | codex/parametric-operator |

纵向变化的优先实现方案是二维 TE 频域散射：学习 Ey/Hx/Hz 的复实虚部，加入 Maxwell 一阶残差、端口、界面与 PML。应实际对照载波提取、频率特征、局部训练、残差重采样和损失平衡。模态展开是另一条可解释路线，但必须评估前后向模态和辐射的影响。

改变物理假设或求解方向前，先创建独立 worktree 与 codex/ 分支，记录适用范围和独立基准再比较。表中后续方向仅为路线，尚未实现。当前每个配置重新训练，尚无任意介质泛化或相对于传统求解器的速度优势证据。
