# PINN_Maxwell

用于波导光学的麦克斯韦物理神经网络。包含多层平板波导 TE 基模，以及**圆偏振激励的全矢量同轴分层介质光纤**；输出有效折射率、复数 E/H、材料界面两侧场和 Poynting 功率流。

[中文论文调研与方案报告](docs/literature_report_zh.md) · [验证结果](docs/validation_results.md) · [文献 BibTeX](docs/references.bib)

## 圆柱介质光纤

[模型、边界条件与范围](docs/cylindrical_fiber_zh.md) · [界面 E/H 实测结果](docs/fiber_validation_zh.md)

默认 λ=1.55 μm，圆柱计算半径 4 μm、长度 10 μm；芯层半径 0.75 μm、内包层外半径 1.5 μm，三层折射率为 1.60 / 1.45 / 1.44。材料沿径向分层，分界面是 `r=0.75` 与 `r=1.5 μm` 的两个圆柱面，沿 z 不变。外包层使用精确衰减导模边界；未采用金属壁。

```powershell
python -m pip install -e .
python -m pinn_maxwell.fiber_demo --output runs/fiber_circular --publish results/fiber_circular
python -m unittest discover -s tests -v
```

也可使用 `pinn-maxwell-fiber`。训练为每个有限径向层的四状态 MLP，加上可训练 neff；全矢量 Maxwell 一阶残差、轴心正则性和界面约束决定解，外层 K1 尾场是开放边界的精确延拓。Bessel 参考仅用于验证。

使用仓库已保存的权重直接生成完整界面数组：

```powershell
python -m pinn_maxwell.fiber_demo --checkpoint results/fiber_circular/model.pt --adam-steps 0 --lbfgs-steps 0 --output runs/fiber_saved
```

圆偏振激励是两组正交简并模式的等幅 π/2 相差叠加。保留 Ex/Ey/Ez、Hx/Hy/Hz；轴心 Ey=i Ex，局部偏振在界面附近可能呈椭圆，由 Stokes 图检查。`handedness=-1` 切换相反旋向。当前为直光纤导模激励，不包含端面自由空间耦合或纵向材料突变的反射。

| 输出文件 | 内容 |
| --- | --- |
| `interface_1.csv`、`interface_2.csv` | z=0 与 z=L 的 129 个角度；界面内外两侧六个复数 E/H 分量、模长；含 Cartesian 和 r/φ/z 坐标分量 |
| `interface_1.npz`、`interface_2.npz` | 整个分界面 `[z, phi, component]` 的复场、位置、内外侧数据 |
| `interface_1.png`、`interface_2.png` | 展开的圆柱分界面：E/H 模长与场分量相位分布 |
| `cross_section.npz`、`cross_section.png` | 输入/输出横截面 E/H、场图、圆偏振度与纵向 Ez |
| `metrics.json`、`history.json`、`model.pt` | 参考误差、界面条件、独立 Maxwell 检查、训练历史和可重载网络 |

```python
import numpy as np
from pinn_maxwell import FiberPINN
from pinn_maxwell.fiber_fields import fiber_fields
model = FiberPINN.from_checkpoint('results/fiber_circular/model.pt')
E, H = fiber_fields(model, x_um=0.75, y_um=0., z_um=10., layer=0)  # 界面芯层侧
E2, H2 = fiber_fields(model, x_um=0.75, y_um=0., z_um=10., layer=1)  # 相同坐标包层侧
data = np.load('runs/fiber_circular/interface_1.npz')
```

坐标为 μm，E 为 V/m，H 为 A/m。`electric_amplitude_v_m` 指轴心复 E 模长，各正交分量为其 1/√2；模长是复幅值范数。外侧计算圆柱 `r=4 μm` 为开放截断边界，物理外包层延伸到无穷远。每个配置重新训练；不声称任意器件泛化。验证未通过返回退出码 2，并禁止 `--publish`。

## 真实 FDTD 独立验证

已使用 Ansys Lumerical 2024 R1 对同一圆偏振光纤完成 80/60/40/30/25 nm 五档三维 FDTD。25 nm 下，输入全截面 E/H 差异为 0.80675%/0.51464%，界面两侧 E 为 0.23%–0.34%、H 为 0.57%–0.73%。传播 10 μm 后原始复场差异约 12.47%，主要受 FDTD 网格相位色散影响；相位对齐后的场形指标需单独解读。有效折射率外推偏差估计 +0.00158%，拟合敏感度约 0.00249%，尚不能精确确定该量级的网络偏差。

[完整 FDTD 报告与复现步骤](docs/fdtd_validation_zh.md)、[指标及偏差](results/fdtd_validation/comparison.json)、[两条界面 CSV](results/fdtd_validation/)、[实际求解器审计](results/fdtd_validation/solver_audit.json)。验证代码在 `fdtd_validation/`；独立分支 `codex/fdtd-fiber-validation`。运行商业 FDTD 需要本机 Lumerical 与许可证，分析工具和普通物理测试可独立安装。

## 平板波导运行

Python >=3.10。CPU 即可；建议独立虚拟环境。

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m unittest discover -s tests -v
python -m pinn_maxwell.demo --config configs/slab.json --output runs/slab
python -m pinn_maxwell.demo --config configs/coated_slab.json --output runs/coated_slab
python -m pinn_maxwell.demo --config configs/asymmetric_slab.json --output runs/asymmetric_slab
```

Linux/macOS 激活方式为 `source .venv/bin/activate`。也可使用安装后的 `pinn-maxwell` 命令。工作目录应为仓库根目录。默认训练：Adam 1500 步 + L-BFGS 最多 400 步，float64，随机种子 7，单线程 CPU。可用 `--device cuda`，但小网络不保证 GPU 更快。

训练不使用参考场标签；完成训练后才调用独立参考。验证未过时返回退出码 2，结果中 `validated=false`；`--allow-unvalidated` 仅用于诊断，不会改变验证标志。

## 输出

| 文件 | 内容 |
| --- | --- |
| `metrics.json` | 独立参考误差、残差、功率、环境、配置及验收标志 |
| `fields.png` | 横截面 E/H、传播相位图和输出电场模方 |
| `fields.npz` | `E_v_m`、`H_a_m`：复数数组 `[z, x, xyz]`；坐标单位 μm，场为 V/m 与 A/m |
| `output_cross_section.csv` | z=L 的 Ey/Hx/Hz 实虚部及 Sz；零分量 Ex/Ez/Hy 在 NPZ 中保留 |
| `model.pt` | 网络参数、几何和网络宽度 |
| `history.json` | 训练轨迹 |

```python
import numpy as np
data = np.load('runs/slab/fields.npz')
E_out = data['E_v_m'][-1]  # [x, xyz]
H_out = data['H_a_m'][-1]
```

已训练的小型权重保存在 `results/`，可直接重载，无需重新训练：

```python
import torch
from pinn_maxwell import MaxwellTEPINN
model = MaxwellTEPINN.from_checkpoint('results/slab/model.pt')
with torch.no_grad():
    e_q = model(torch.linspace(-2, 2, 1001, dtype=torch.float64))
    print(float(model.neff))
```

配置长度单位 μm；`indices` 是从左至右的实折射率，`boundaries_um` 比它多一个元素。第一和最后介质为无限包层，最外侧坐标只是导出范围。`anchor_um` 指定 e=1 的归一化点；`electric_amplitude_v_m` 是模式锚点或输入高斯光峰值的 E 幅值。

## 平板波导模型的范围

传播沿 z，TE 电场沿 y，横截面在 x 变化，沿 y 不变。网络在内部层学习 e=Ey 与 q=e'/k0，界面约束 e/q 连续，外部使用精确指数衰减。传播重建 Hx 与 Hz，并保留相位；Hz 一般不为零。无节点约束只适用于基模。

`launch=mode` 是直波导单模传播；`launch=gaussian` 只保留输入电场投影到 TE 基模的分量。保留比例是电场 L2 比例，不是严格耦合效率；未求解反射、辐射和其它模态。材料层变化在横向 x，尚未覆盖沿 z 突变的器件、二维截面全矢量场、TM、损耗、弯曲或任意光路。

## Git 工作方式

平板基线在 `codex/maxwell-pinn-prototype`；光纤方向在独立 worktree 的 `codex/cylindrical-circular-layers` 分支。下一物理方向先创建新的 worktree。

```powershell
git worktree add -b codex/fiber-axial-scattering ../PINN_Maxwell-axial codex/cylindrical-circular-layers
```

后续全矢量、传播散射及条件神经算子方向的验收标准见调研报告。
