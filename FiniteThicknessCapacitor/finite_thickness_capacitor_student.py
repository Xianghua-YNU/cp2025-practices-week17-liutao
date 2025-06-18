import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time
from scipy.ndimage import laplace

# 设置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体，可根据系统情况修改
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def solve_laplace_sor(nx, ny, plate_thickness, plate_separation, omega=1.9, max_iter=10000, tolerance=1e-6):
    """
    使用逐次超松弛（SOR）方法求解二维拉普拉斯方程，用于有限厚度平行板电容器。

    参数:
        nx (int): x 方向的网格点数
        ny (int): y 方向的网格点数
        plate_thickness (int): 导体板的厚度（网格点数）
        plate_separation (int): 极板间的间距（网格点数）
        omega (float): 松弛因子 (1.0 < omega < 2.0)
        max_iter (int): 最大迭代次数
        tolerance (float): 收敛容差

    返回:
        tuple: (potential_grid, conductor_mask)
            - potential_grid: 二维电势数组
            - conductor_mask: 布尔数组，标记导体区域
    """
    # 初始化电势网格
    U = np.zeros((ny, nx))

    # 创建导体掩码
    conductor_mask = np.zeros((ny, nx), dtype=bool)

    # 定义导体区域
    # 上极板: +100V
    conductor_left = nx // 4
    conductor_right = nx // 4 * 3
    y_upper_start = ny // 2 + plate_separation // 2
    y_upper_end = y_upper_start + plate_thickness
    conductor_mask[y_upper_start:y_upper_end,
                   conductor_left:conductor_right] = True
    U[y_upper_start:y_upper_end, conductor_left:conductor_right] = 100.0

    # 下极板: -100V
    y_lower_end = ny // 2 - plate_separation // 2
    y_lower_start = y_lower_end - plate_thickness
    conductor_mask[y_lower_start:y_lower_end,
                   conductor_left:conductor_right] = True
    U[y_lower_start:y_lower_end, conductor_left:conductor_right] = -100.0

    # 边界条件: 接地边界
    U[:, 0] = 0.0
    U[:, -1] = 0.0
    U[0, :] = 0.0
    U[-1, :] = 0.0

    # SOR 迭代
    for iteration in range(max_iter):
        U_old = U.copy()
        max_error = 0.0

        # 更新内部点（不包括导体和边界）
        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                if not conductor_mask[i, j]:  # 跳过导体点
                    # SOR 更新公式
                    U_new = 0.25 * (U[i + 1, j] + U[i - 1, j] +
                                    U[i, j + 1] + U[i, j - 1])
                    U[i, j] = (1 - omega) * U[i, j] + omega * U_new

                    # 跟踪最大误差
                    error = abs(U[i, j] - U_old[i, j])
                    max_error = max(max_error, error)

        # 检查收敛性
        if max_error < tolerance:
            print(f"经过 {iteration + 1} 次迭代后收敛")
            break
    else:
        print(f"警告: 达到最大迭代次数 ({max_iter})")

    return U


def calculate_charge_density(potential_grid, dx, dy):
    """
    使用泊松方程计算电荷密度: rho = -1/(4*pi) * nabla^2(U)

    参数:
        potential_grid (np.ndarray): 二维电势分布
        dx (float): x 方向的网格间距
        dy (float): y 方向的网格间距

    返回:
        np.ndarray: 二维电荷密度分布
    """
    # 使用 scipy.ndimage.laplace 计算拉普拉斯算子
    laplacian_U = laplace(potential_grid, mode='nearest') / \
        (dx**2)  # 假设 dx = dy

    # 根据泊松方程计算电荷密度: rho = -1/(4*pi) * nabla^2(U)
    rho = -laplacian_U / (4 * np.pi)

    return rho


def plot_results(potential, charge_density, x_coords, y_coords):
    """
    对结果进行综合可视化

    参数:
        potential (np.ndarray): 二维电势分布
        charge_density (np.ndarray): 电荷密度分布
        x_coords (np.ndarray): x 坐标数组
        y_coords (np.ndarray): y 坐标数组
    """
    X, Y = np.meshgrid(x_coords, y_coords)

    fig = plt.figure(figsize=(15, 6))

    # 子图 1: 电势的 3D 可视化
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_wireframe(X, Y, potential, rstride=3, cstride=3, color='r')
    levels = np.linspace(potential.min(), potential.max(), 20)
    ax1.contour(X, Y, potential, zdir='z',
                offset=potential.min(), levels=levels)
    ax1.set_title('电势的 3D 可视化')
    ax1.set_xlabel('X 位置')
    ax1.set_ylabel('Y 位置')
    ax1.set_zlabel('电势')

    # 子图 2: 电荷密度的 3D 分布
    ax2 = fig.add_subplot(122, projection='3d')
    surf = ax2.plot_surface(X, Y, charge_density,
                            cmap='RdBu_r', edgecolor='none')
    fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=5, label='电荷密度')
    ax2.set_xlabel('X 位置')
    ax2.set_ylabel('Y 位置')
    ax2.set_zlabel('电荷密度')
    ax2.set_title('电荷密度的 3D 分布')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 模拟参数
    nx, ny = 120, 100  # 网格尺寸
    plate_thickness = 10  # 导体厚度（网格点数）
    plate_separation = 40  # 极板间距
    omega = 1.9  # SOR 松弛因子

    # 物理尺寸
    Lx, Ly = 1.0, 1.0  # 区域大小
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)

    # 创建坐标数组
    x_coords = np.linspace(0, Lx, nx)
    y_coords = np.linspace(0, Ly, ny)

    print("正在求解有限厚度平行板电容器问题...")
    print(f"网格大小: {nx} x {ny}")
    print(f"极板厚度: {plate_thickness} 个网格点")
    print(f"极板间距: {plate_separation} 个网格点")
    print(f"SOR 松弛因子: {omega}")

    # 求解拉普拉斯方程
    start_time = time.time()
    potential = solve_laplace_sor(
        nx, ny, plate_thickness, plate_separation, omega
    )
    solve_time = time.time() - start_time

    print(f"求解完成，用时 {solve_time:.2f} 秒")

    # 计算电荷密度
    charge_density = calculate_charge_density(potential, dx, dy)

    # 可视化结果
    plot_results(potential, charge_density, x_coords, y_coords)

    # 打印一些统计信息
    print(f"\n电势统计信息:")
    print(f"  最小电势: {np.min(potential):.2f} V")
    print(f"  最大电势: {np.max(potential):.2f} V")
    print(f"  电势范围: {np.max(potential) - np.min(potential):.2f} V")

    print(f"\n电荷密度统计信息:")
    print(f"  最大电荷密度: {np.max(np.abs(charge_density)):.6f}")
    print(
        f"  总正电荷: {np.sum(charge_density[charge_density > 0]) * dx * dy:.6f}")
    print(
        f"  总负电荷: {np.sum(charge_density[charge_density < 0]) * dx * dy:.6f}")
    print(f"  总电荷: {np.sum(charge_density) * dx * dy:.6f}")
