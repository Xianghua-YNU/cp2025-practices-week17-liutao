import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# 设置图片清晰度
plt.rcParams['figure.dpi'] = 300
# 设置中文字体，这里使用黑体，不同系统字体名称可能不同
plt.rcParams['font.sans-serif'] = ['SimHei']
# 解决负号显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False

def solve_laplace_jacobi(xgrid, ygrid, w, d, tol=1e-5):
    """
    使用雅可比迭代法求解拉普拉斯方程。

    参数:
        xgrid (int): x 方向的网格点数
        ygrid (int): y 方向的网格点数
        w (int): 平行板的宽度
        d (int): 平行板之间的距离
        tol (float): 收敛容差

    返回:
        tuple: (电位数组, 迭代次数, 收敛历史记录)
    """
    # 初始化电位网格
    u = np.zeros((ygrid, xgrid))

    # 计算极板位置
    xL = (xgrid - w) // 2
    xR = (xgrid + w) // 2
    yB = (ygrid - d) // 2
    yT = (ygrid + d) // 2

    # 设置极板的边界条件
    u[yT, xL:xR+1] = 100.0  # 上极板: +100V
    u[yB, xL:xR+1] = -100.0  # 下极板: -100V

    迭代次数 = 0
    最大迭代次数 = 10000
    收敛历史记录 = []

    while 迭代次数 < 最大迭代次数:
        u_old = u.copy()

        # 雅可比迭代
        u[1:-1, 1:-1] = 0.25 * (u[2:, 1:-1] + u[:-2, 1:-1] +
                                u[1:-1, 2:] + u[1:-1, :-2])

        # 保持边界条件
        u[yT, xL:xR+1] = 100.0
        u[yB, xL:xR+1] = -100.0

        # 计算收敛指标
        最大变化量 = np.max(np.abs(u - u_old))
        收敛历史记录.append(最大变化量)

        # 检查收敛情况
        迭代次数 += 1
        if 最大变化量 < tol:
            break

    return u, 迭代次数, 收敛历史记录


def solve_laplace_sor(xgrid, ygrid, w, d, omega=1.25, Niter=1000, tol=1e-5):
    """
    使用高斯 - 赛德尔逐次超松弛（SOR）迭代法求解拉普拉斯方程。

    参数:
        xgrid (int): x 方向的网格点数
        ygrid (int): y 方向的网格点数
        w (int): 平行板的宽度
        d (int): 平行板之间的距离
        omega (float): 松弛因子
        Niter (int): 最大迭代次数

    返回:
        tuple: (电位数组, 迭代次数, 收敛历史记录)
    """
    # 初始化电位网格
    u = np.zeros((ygrid, xgrid))

    # 计算极板位置
    xL = (xgrid - w) // 2
    xR = (xgrid + w) // 2
    yB = (ygrid - d) // 2
    yT = (ygrid + d) // 2

    # 设置极板的边界条件
    u[yT, xL:xR+1] = 100.0  # 上极板: +100V
    u[yB, xL:xR+1] = -100.0  # 下极板: -100V

    收敛历史记录 = []

    for 迭代次数 in range(Niter):
        u_old = u.copy()

        # SOR 迭代
        for i in range(1, ygrid - 1):
            for j in range(1, xgrid - 1):
                # 跳过极板区域
                if (i == yT and xL <= j <= xR) or (i == yB and xL <= j <= xR):
                    continue

                # 计算残差
                r_ij = 0.25 * (u[i + 1, j] + u[i - 1, j] + u[i, j + 1] + u[i, j - 1])

                # 应用 SOR 公式
                u[i, j] = (1 - omega) * u[i, j] + omega * r_ij

        # 保持边界条件
        u[yT, xL:xR+1] = 100.0
        u[yB, xL:xR+1] = -100.0

        # 计算收敛指标
        最大变化量 = np.max(np.abs(u - u_old))
        收敛历史记录.append(最大变化量)

        # 检查收敛情况
        if 最大变化量 < tol:
            break

    return u, 迭代次数 + 1, 收敛历史记录


def plot_results(x, y, u, 方法名称):
    """
    绘制 3D 电位分布和等电位线与电场线的组合图。

    参数:
        x (array): X 坐标
        y (array): Y 坐标
        u (array): 电位分布
        方法名称 (str): 使用的方法名称
    """
    fig = plt.figure(figsize=(10, 5))

    # 3D 线框图
    ax1 = fig.add_subplot(121, projection='3d')
    X, Y = np.meshgrid(x, y)
    ax1.plot_wireframe(X, Y, u, alpha=0.7)
    等值线级别 = np.linspace(u.min(), u.max(), 20)
    ax1.contour(x, y, u, zdir='z', offset=u.min(), levels=等值线级别)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('电位 (V)')
    ax1.set_title(f'3D 电位分布\n({方法名称})')

    # 等电位线图和电场流线图组合
    ax2 = fig.add_subplot(122)
    等值线级别 = np.linspace(u.min(), u.max(), 20)
    等高线 = ax2.contour(X, Y, u, levels=等值线级别, colors='red',
                          linestyles='dashed', linewidths=0.8)
    ax2.clabel(等高线, inline=True, fontsize=8, fmt='%1.1f')

    # 电场是电位的负梯度（注意：np.gradient 返回的梯度是先沿行 (y) 再沿列 (x)）
    EY, EX = np.gradient(-u, 1)
    ax2.streamplot(X, Y, EX, EY, density=1.5, color='blue',
                   linewidth=1, arrowsize=1.5, arrowstyle='->')

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title(
        f'等电位线与电场线\n({方法名称})')
    ax2.set_aspect('equal')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 模拟参数
    xgrid, ygrid = 50, 50
    w, d = 20, 20  # 极板宽度和间距
    tol = 1e-3
    # 创建坐标数组
    x = np.linspace(0, xgrid - 1, xgrid)
    y = np.linspace(0, ygrid - 1, ygrid)

    print("正在求解平行板电容器的拉普拉斯方程...")
    print(f"网格大小: {xgrid} x {ygrid}")
    print(f"极板宽度: {w}, 间距: {d}")
    print(f"容差: {tol}")

    # 使用雅可比方法求解
    print("1. 雅可比迭代法:")
    开始时间 = time.time()
    u_jacobi, 迭代次数_jacobi, 收敛历史记录_jacobi = solve_laplace_jacobi(
        xgrid, ygrid, w, d, tol=tol)
    时间_jacobi = time.time() - 开始时间
    print(f"   在 {迭代次数_jacobi} 次迭代后收敛")
    print(f"   耗时: {时间_jacobi:.3f} 秒")

    # 使用 SOR 方法求解
    print("2. 高斯 - 赛德尔 SOR 迭代法:")
    开始时间 = time.time()
    u_sor, 迭代次数_sor, 收敛历史记录_sor = solve_laplace_sor(
        xgrid, ygrid, w, d, tol=tol)
    时间_sor = time.time() - 开始时间
    print(f"   在 {迭代次数_sor} 次迭代后收敛")
    print(f"   耗时: {时间_sor:.3f} 秒")

    # 性能比较
    print("\n3. 性能比较:")
    print(f"   雅可比法: {迭代次数_jacobi} 次迭代, {时间_jacobi:.3f} 秒")
    print(f"   SOR 法:    {迭代次数_sor} 次迭代, {时间_sor:.3f} 秒")
    print(
        f"   加速比: {迭代次数_jacobi / 迭代次数_sor:.1f} 倍迭代次数, {时间_jacobi / 时间_sor:.2f} 倍时间")

    # 绘制结果
    plot_results(x, y, u_jacobi, "雅可比方法")
    plot_results(x, y, u_sor, "SOR 方法")

    # 绘制收敛比较图
    plt.figure(figsize=(10, 6))
    plt.semilogy(range(len(收敛历史记录_jacobi)),
                 收敛历史记录_jacobi, 'r-', label='雅可比方法')
    plt.semilogy(range(len(收敛历史记录_sor)),
                 收敛历史记录_sor, 'b-', label='SOR 方法')
    plt.xlabel('迭代次数')
    plt.ylabel('最大变化量')
    plt.title('收敛比较')
    plt.grid(True)
    plt.legend()
    plt.show()
