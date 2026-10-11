# PINN_Maxwell

用于波导光学的麦克斯韦物理神经网络原型。第一版求解**无损、各向同性、多层平板波导的 TE 基模**，输出有效折射率、传播后的复数 E/H 场和 Poynting 功率流。

[中文论文调研与方案报告](docs/literature_report_zh.md) · [验证结果](docs/validation_results.md) · [文献 BibTeX](docs/references.bib)

## 运行

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

## 当前物理范围

传播沿 z，TE 电场沿 y，横截面在 x 变化，沿 y 不变。网络在内部层学习 e=Ey 与 q=e'/k0，界面约束 e/q 连续，外部使用精确指数衰减。传播重建 Hx 与 Hz，并保留相位；Hz 一般不为零。无节点约束只适用于基模。

`launch=mode` 是直波导单模传播；`launch=gaussian` 只保留输入电场投影到 TE 基模的分量。保留比例是电场 L2 比例，不是严格耦合效率；未求解反射、辐射和其它模态。材料层变化在横向 x，尚未覆盖沿 z 突变的器件、二维截面全矢量场、TM、损耗、弯曲或任意光路。

## Git 工作方式

当前方向在 `codex/maxwell-pinn-prototype` 分支和独立 worktree 开发。下一物理方向先创建新的 worktree；不覆盖其它光学项目。

```powershell
git worktree add -b codex/vector-cross-section ../PINN_Maxwell-vector codex/maxwell-pinn-prototype
```

后续全矢量、传播散射及条件神经算子方向的验收标准见调研报告。
