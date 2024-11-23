import torch
import matplotlib.pyplot as plt
from manifold import StiefelManifold
import numpy as np

# 设置随机种子以保证可重复性
torch.manual_seed(42)

# 创建Stiefel流形实例 (例如6x3的矩阵)
rdim, kdim = 100, 10
manifold = StiefelManifold(rdim, kdim)

# 生成随机正交矩阵作为基点
x = torch.nn.init.orthogonal_(torch.empty(rdim, kdim))

# 生成不同模长的切向量并计算误差
norms = np.logspace(-20, 1, 1000)  # 从10^-3到10^1的对数刻度
num_trials = 3  # 重复次数

errors_svd = np.zeros_like(norms)
errors_qr = np.zeros_like(norms)
errors_cayley = np.zeros_like(norms)
iterations = np.zeros_like(norms)

for trial in range(num_trials):
    # 每次循环生成新的随机切向量
    v = torch.randn(rdim, kdim)
    v = manifold.project(x, v)
    v = v / v.flatten().norm()

    for i, norm in enumerate(norms):
        # 计算指数映射（作为参考）
        exp_map = manifold.exponential_map(x, norm * v).reshape(rdim, kdim)
        iterations[i] += manifold.velocity.call_times
        manifold.velocity.empty_cache()

        # 计算三种收缩映射
        svd_retract = manifold.svd_retract(x, v, t=norm)
        qr_retract = manifold.qr_retract(x, v, t=norm)
        cayley_retract = manifold.cayley_retract(x, v, t=norm)

        # 累加Frobenius范数误差
        errors_svd[i] += torch.norm(exp_map - svd_retract).item()
        errors_qr[i] += torch.norm(exp_map - qr_retract).item()
        errors_cayley[i] += torch.norm(exp_map - cayley_retract).item()

# 计算平均误差和迭代次数
errors_svd /= num_trials
errors_qr /= num_trials
errors_cayley /= num_trials
iterations /= num_trials

# 创建图形和主坐标轴
fig, ax1 = plt.subplots(figsize=(10, 6), constrained_layout=True)

# 绘制误差曲线（使用左侧y轴）
ax1.loglog(norms, errors_svd, "b-", label="SVD Retraction")
ax1.loglog(norms, errors_qr, "r--", label="QR Retraction")
ax1.loglog(norms, errors_cayley, "g-.", label="Cayley Retraction")
ax1.grid(True)
ax1.set_xlabel("Tangent Vector Norm")
ax1.set_ylabel("Average Error (Frobenius Norm)")

# 创建右侧y轴并绘制迭代次数
ax2 = ax1.twinx()
ax2.semilogx(norms, iterations, "k-", label="Iterations")
ax2.set_ylabel("Average Number of Iterations")

# 合并两个坐标轴的图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2)

plt.title("Comparison of Retraction Methods and Iterations")
plt.savefig("retraction_comparison.png")
plt.close()
