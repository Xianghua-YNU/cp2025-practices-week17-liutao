"""
学生模板：波动方程FTCS解
文件：wave_equation_ftcs_student.py
重要：函数名称必须与参考答案一致！
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def u_t(x, C=1, d=0.1, sigma=0.3, L=1):
    """
    计算初始速度剖面 psi(x)。

    参数:
        x (np.ndarray): 位置数组。
        C (float): 振幅常数。
        d (float): 指数项的偏移量。
        sigma (float): 指数项的宽度。
        L (float): 弦的长度。
    返回:
        np.ndarray: 初始速度剖面。
    """
    # 实现初始速度剖面函数
    return C * x * (L - x) / (L * L) * np.exp(-(x - d)**2 / (2 * sigma**2))

def solve_wave_equation_ftcs(parameters):
    """
    使用FTCS有限差分法求解一维波动方程。
    
    参数:
        parameters (dict): 包含以下参数的字典：
            - 'a': 波速 (m/s)。
            - 'L': 弦的长度 (m)。
            - 'd': 初始速度剖面的偏移量 (m)。
            - 'C': 初始速度剖面的振幅常数 (m/s)。
            - 'sigma': 初始速度剖面的宽度 (m)。
            - 'dx': 空间步长 (m)。
            - 'dt': 时间步长 (s)。
            - 'total_time': 总模拟时间 (s)。
    返回:
        tuple: 包含以下内容的元组：
            - np.ndarray: 解数组 u(x, t)。
            - np.ndarray: 空间数组 x。
            - np.ndarray: 时间数组 t。
    """
    # 从字典中获取参数
    a = parameters['a']
    L = parameters['L']
    d = parameters['d']
    C = parameters['C']
    sigma = parameters['sigma']
    dx = parameters['dx']
    dt = parameters['dt']
    total_time = parameters['total_time']
    
    # 初始化空间网格和时间网格
    x = np.arange(0, L + dx, dx)
    t = np.arange(0, total_time + dt, dt)
    
    # 创建解数组
    u = np.zeros((len(x), len(t)))
    
    # 计算稳定性条件 c = (a * dt / dx)^2
    c_val = (a * dt / dx) ** 2
    
    # 检查稳定性条件
    if c_val >= 1:
        print(f"警告: 稳定性条件 c = {c_val:.4f} >= 1，解可能不稳定!")
    
    # 计算初始速度剖面
    velocity_profile = u_t(x, C, d, sigma, L)
    
    # 应用初始条件：
    # u(x, 0) = 0 (由np.zeros初始化满足)
    # 计算第一个时间步 u(x, 1)
    u[1:-1, 1] = velocity_profile[1:-1] * dt
    
    # FTCS主迭代循环
    for j in range(1, len(t) - 1):
        # 内部点更新
        u[1:-1, j+1] = (c_val * (u[2:, j] + u[:-2, j]) + 
                         2 * (1 - c_val) * u[1:-1, j] - 
                         u[1:-1, j-1])
    
    return u, x, t

if __name__ == "__main__":
    # 演示和测试
    params = {
        'a': 100,
        'L': 1,
        'd': 0.1,
        'C': 1,
        'sigma': 0.3,
        'dx': 0.01,
        'dt': 5e-5,
        'total_time': 0.1
    }
    u_sol, x_sol, t_sol = solve_wave_equation_ftcs(params)

    # 创建动画
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, xlim=(0, params['L']), 
                         ylim=(np.min(u_sol) * 1.1, np.max(u_sol) * 1.1))
    line, = ax.plot([], [], 'g-', lw=2)
    ax.set_title("一维波动方程 (FTCS方法)")
    ax.set_xlabel("位置 (m)")
    ax.set_ylabel("位移")
    ax.grid(True)
    
    # 帧数文本
    frame_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    
    def init():
        line.set_data([], [])
        frame_text.set_text('')
        return line, frame_text
    
    def update(frame):
        line.set_data(x_sol, u_sol[:, frame])
        frame_text.set_text(f'时间 = {t_sol[frame]:.4f} s')
        return line, frame_text
    
    ani = FuncAnimation(fig, update, frames=len(t_sol),
                        init_func=init, interval=20, blit=True)
    
    plt.tight_layout()
    plt.show()

