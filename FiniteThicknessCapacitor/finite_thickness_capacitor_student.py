import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time
from scipy.ndimage import laplace
from mpl_toolkits.mplot3d import Axes3D


def solve_laplace_sor(nx=100, ny=100, plate_thickness=5, plate_separation=20, omega=1.9, max_iter=10000, tolerance=1e-4):
    """
    使用逐次超松弛（SOR）方法求解二维拉普拉斯方程。

    参数:
    nx (int): x 方向的网格点数，默认值为 100。
    ny (int): y 方向的网格点数，默认值为 100。
    plate_thickness (int): 极板厚度，默认值为 5。
    plate_separation (int): 极板间距，默认值为 20。
    omega (float): 松弛因子，默认值为 1.9。
    max_iter (int): 最大迭代次数，默认值为 10000。
    tolerance (float): 收敛容差，默认值为 1e-4。

    返回:
    np.ndarray: 收敛后的电势分布。
    """
    # 初始化电势网格
    U = np.zeros((ny, nx))
    # 创建导体掩码
    conductor_mask = np.zeros((ny, nx), dtype=bool)

    # 定义导体区域
    top_plate_y_start = plate_separation
    top_plate_y_end = top_plate_y_start + plate_thickness
    bottom_plate_y_start = 0
    bottom_plate_y_end = bottom_plate_y_start + plate_thickness

    U[top_plate_y_start:top_plate_y_end, :] = 100
    U[bottom_plate_y_start:bottom_plate_y_end, :] = -100

    conductor_mask[top_plate_y_start:top_plate_y_end, :] = True
    conductor_mask[bottom_plate_y_start:bottom_plate_y_end, :] = True

    # 边界条件
    U[0, :] = 0
    U[-1, :] = 0
    U[:, 0] = 0
    U[:, -1] = 0

    for _ in range(max_iter):
        U_old = U.copy()
        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                if not conductor_mask[i, j]:
                    U[i, j] = (1 - omega) * U[i, j] + omega * 0.25 * \
                        (U[i+1, j] + U[i-1, j] + U[i, j+1] + U[i, j-1])
        # 检查收敛
        if np.max(np.abs(U - U_old)) < tolerance:
            break
    return U


def calculate_charge_density(potential_grid, dx, dy):
    """
    根据泊松方程计算电荷密度。

    参数:
    potential_grid (np.ndarray): 电势网格。
    dx (float): x 方向的网格间距。
    dy (float): y 方向的网格间距。

    返回:
    np.ndarray: 电荷密度分布的二维数组。
    """
    laplacian_U = laplace(potential_grid)
    rho = -1 / (4 * np.pi) * laplacian_U / (dx * dy)
    return rho


def plot_results(potential, charge_density, x_coords, y_coords):
    """
    对计算结果进行可视化。

    参数:
    potential (np.ndarray): 电势分布。
    charge_density (np.ndarray): 电荷密度分布。
    x_coords (np.ndarray): x 坐标数组。
    y_coords (np.ndarray): y 坐标数组。
    """
    X, Y = np.meshgrid(x_coords, y_coords)

    fig = plt.figure(figsize=(12, 5))

    # 电势的 3D 可视化
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(X, Y, potential, cmap='viridis')
    fig.colorbar(surf1, shrink=0.5, aspect=5, label='Potential (V)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('Potential')
    ax1.set_title('Potential Distribution')

    # 电荷密度的 3D 分布
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(X, Y, charge_density, cmap='viridis')
    fig.colorbar(surf2, shrink=0.5, aspect=5, label='Charge Density')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('Charge Density')
    ax2.set_title('Charge Density Distribution')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 设置模拟参数
    nx = 100
    ny = 100
    plate_thickness = 5
    plate_separation = 20
    omega = 1.9

    # 计算物理尺寸和网格间距
    Lx = 1.0
    Ly = 1.0
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    x_coords = np.linspace(0, Lx, nx)
    y_coords = np.linspace(0, Ly, ny)

    # 求解拉普拉斯方程
    start_time = time.time()
    U = solve_laplace_sor(nx, ny, plate_thickness, plate_separation, omega)
    end_time = time.time()
    solve_time = end_time - start_time
    print(f"求解时间: {solve_time:.4f} 秒")

    # 计算电荷密度
    rho = calculate_charge_density(U, dx, dy)

    # 可视化结果
    plot_results(U, rho, x_coords, y_coords)

    # 打印统计信息
    print(f"电势最小值: {np.min(U):.4f} V")
    print(f"电势最大值: {np.max(U):.4f} V")
    print(f"电荷密度最小值: {np.min(rho):.4e} C/m^2")
    print(f"电荷密度最大值: {np.max(rho):.4e} C/m^2")
    total_charge = np.sum(rho) * dx * dy
    print(f"总电荷: {total_charge:.4e} C")
