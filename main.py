from optimizer import *
from manifold import *

import torch
import matplotlib.pyplot as plt

# Initialize parameters
rdim = 1000
kdim = 5
max_iter = 200

eye_mat = torch.eye(rdim)[:, :kdim]
symmetric_mat = torch.randn(rdim, rdim)
symmetric_mat = (symmetric_mat + symmetric_mat.T) / 2

G = torch.randn(rdim, kdim)


# Create a simple test function
def test_function(x):
    return torch.trace(x.T @ symmetric_mat @ x) + 2 * torch.trace(x.T @ G)


# Create Stiefel manifold
manifold = StiefelManifold(rdim, kdim)

# Generate random initial point and orthogonalize
start_point = torch.nn.init.orthogonal_(torch.empty(rdim, kdim))

# Test gradient descent
print("\nTesting Gradient Descent:")
gd = GradientDescent(manifold)
result_gd = gd.optimize(start_point, test_function, max_iter)
print(f"Final function value: {test_function(result_gd):.10f}")

# Test BB method
print("\nTesting BB Method:")
bb = BBMethod(manifold)
result_bb = bb.optimize(start_point, test_function, max_iter)
print(f"Final function value: {test_function(result_bb):.10f}")

# Test regularized Newton method
print("\nTesting Regularized Newton Method:")
rn = RegularizedNewton(manifold)
result_rn = rn.optimize(start_point, test_function, max_iter)
print(f"Final function value: {test_function(result_rn):.10f}")

# Plot convergence curves
plt.figure(figsize=(10, 6))

# 获取所有函数值
gd_values = gd.log_data["function_value"]
bb_values = bb.log_data["function_value"]
rn_values = rn.log_data["function_value"]

# 计算所有数据中的最小值
min_value = min(min(gd_values), min(bb_values), min(rn_values))

# 对数据进行处理：减去最小值后取对数
gd_processed = torch.log(torch.tensor(gd_values) - min_value + 1e-10)
bb_processed = torch.log(torch.tensor(bb_values) - min_value + 1e-10)
rn_processed = torch.log(torch.tensor(rn_values) - min_value + 1e-10)

plt.plot(gd.log_data["iteration"], gd_processed, label="Gradient Descent")
plt.plot(bb.log_data["iteration"], bb_processed, label="BB Method")
plt.plot(rn.log_data["iteration"], rn_processed, label="Regularized Newton")
plt.xlabel("Iterations")
plt.ylabel("Log(Function Value - Minimum)")
plt.title("Optimization Methods Convergence Comparison")
plt.legend()
plt.grid(True)
plt.savefig("optimization_comparison.png")
plt.close()
