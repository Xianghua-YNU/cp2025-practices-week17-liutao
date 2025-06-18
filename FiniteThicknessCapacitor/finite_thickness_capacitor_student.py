import numpy as np
from scipy.ndimage import laplace
import matplotlib.pyplot as plt


def solve_laplace_sor(grid_size=100, omega=1.9, max_iter=10000, tol=1e-4):
    """
    使用逐次超松弛迭代方法求解二维拉普拉斯方程。

    参数:
    grid_size (int): 网格分辨率，默认值为100。
    omega (float): 松弛因子，默认值为1.9。
    max_iter (int): 最大迭代次数，默认值为10000。
    tol (float): 收敛判据，相邻迭代间最大差值，默认值为1e-4。

    返回:
    np.ndarray: 收敛后的电势分布。
    """
    # 初始化电势网格
    U = np.zeros((grid_size, grid_size))
    # 设定边界条件
    U[0, :] = 100  # 上导体板表面
    U[-1, :] = -100  # 下导体板表面
    U[:, 0] = 0  # 左边界
    U[:, -1] = 0  # 右边界

    for _ in range(max_iter):
        U_old = U.copy()
        for i in range(1, grid_size - 1):
            for j in range(1, grid_size - 1):
                U[i, j] = (1 - omega) * U[i, j] + omega * 0.25 * \
                    (U[i+1, j] + U[i-1, j] + U[i, j+1] + U[i, j-1])
        # 检查收敛
        if np.max(np.abs(U - U_old)) < tol:
            break
    return U


def calculate_charge_density(U):
    """
    通过泊松方程计算电荷密度。

    参数:
    U (np.ndarray): 电势分布。

    返回:
    np.ndarray: 电荷密度分布。
    """
    laplacian_U = laplace(U)
    rho = -1 / (4 * np.pi) * laplacian_U
    return rho


def plot_results(U, rho):
    """
    生成电势分布等高线图和表面电荷密度分布。

    参数:
    U (np.ndarray): 电势分布。
    rho (np.ndarray): 电荷密度分布。
    """
    grid_size = U.shape[0]
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    X, Y = np.meshgrid(x, y)

    # 电势分布等高线图
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.contourf(X, Y, U, cmap='viridis')
    plt.colorbar(label='Potential (V)')
    plt.contour(X, Y, U, colors='k', linewidths=0.5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Potential Distribution')

    # 表面电荷密度分布
    plt.subplot(1, 2, 2)
    plt.plot(x, rho[0, :], label='Top Surface')
    plt.plot(x, rho[-1, :], label='Bottom Surface')
    plt.xlabel('x')
    plt.ylabel('Charge Density')
    plt.title('Surface Charge Density Distribution')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 任务1: 求解拉普拉斯方程
    U = solve_laplace_sor()
    # 任务2: 计算表面电荷密度分布
    rho = calculate_charge_density(U)
    # 任务3: 结果可视化与分析
    plot_results(U, rho)
