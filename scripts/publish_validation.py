"""Copy small, validated artifacts into version control and generate result report."""

import json
from pathlib import Path
import shutil
import sys


def main(names):
    rows = []
    for name in names:
        source = Path("runs")/name
        metrics = json.loads((source/"metrics.json").read_text(encoding="utf-8"))
        if not metrics["validated"]:
            raise ValueError(f"Cannot publish an unvalidated example: {name}")
        target = Path("results")/name
        target.mkdir(parents=True, exist_ok=True)
        for filename in ("metrics.json", "history.json", "fields.png", "output_cross_section.csv", "model.pt"):
            shutil.copy2(source/filename, target/filename)
        scores = metrics["scores"]
        rows.append(f"| {name} | {metrics['neff_pinn']:.10f} | {metrics['neff_reference']:.10f} | "
                    f"{100*scores['e_relative_l2']:.5f}% | {100*scores['h_relative_l2']:.5f}% | "
                    f"{100*scores['output_e_relative_l2']:.5f}% | {100*scores['output_h_relative_l2']:.5f}% | "
                    f"{scores['heldout_pde_loss']:.3e} |")
    text = """# 原型实测验证

测试日期：2026-10-11。CPU 单线程、float64；Python 3.10.21、PyTorch 2.3.1、NumPy 1.21.5、SciPy 1.7.3。每个配置独立训练，随机种子 7，宽度 32，Adam 1500 步 + L-BFGS 最多 400 迭代。L-BFGS 线搜索可以多次调用损失，因此历史中的调用次数可大于 400。

λ=1.55 μm，传播长度 10 μm。E/H 误差均为相对 L2，场形误差包含整个导出横向范围；输出误差比较 z=L 的复场，包含累计相位误差。E/H 场标签未参与训练。

| 配置 | PINN neff | 独立参考 neff | E 场形误差 | H 场形误差 | 输出复 E 误差 | 输出复 H 误差 | 独立点 PDE 损失 |
| --- | --- | --- | --- | --- | --- | --- | --- |
""" + "\n".join(rows) + "\n\n"
    for name in names:
        text += (f"## {name}\n\n"
                 f"[完整指标](../results/{name}/metrics.json) · "
                 f"[训练历史](../results/{name}/history.json) · "
                 f"[输出截面 CSV](../results/{name}/output_cross_section.csv) · "
                 f"[已训练权重](../results/{name}/model.pt)\n\n"
                 f"![E/H 与传播结果](../results/{name}/fields.png)\n\n")
    text += """## 适用范围与解释

三层对称波导参考为解析色散方程和独立传输矩阵；两者的本征值与场一致至浮点精度。包覆与非对称波导参考为独立传输矩阵。所有案例使用精确的无限包层衰减边界。

功率在单模传播中保持恒定；这是传播公式保证的结果，不能单独证明神经求解精度。五层包覆示例只传播高斯轮廓的基模投影，L2 保留比例不是严格端口耦合效率。

测试还验证了 Maxwell 旋度符号、解析模式残差、H 单位、相位、界面连续性、非法配置处理。未训练网络的示例被验收门拒绝，CLI 返回退出码 2。保存权重重新加载后与原有场数组一致。

这些结果展示三个配置的正确性，不证明任意几何泛化。原型仍是一维截面 TE 基模，尚未覆盖纵向结构变化、TM、高阶模式或二维全矢量器件。

完整传播数组 `fields.npz` 位于本地 `runs/<配置>/`，避免把大数组放进 Git；可通过 README 命令重新生成。仓库保留小型训练权重、输出截面、图片、指标与历史。
"""
    Path("docs/validation_results.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:] or ["slab", "coated_slab", "asymmetric_slab"])
