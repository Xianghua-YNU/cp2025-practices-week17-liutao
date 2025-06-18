#!/usr/bin/env python3
"""
学生模板：求解正负电荷构成的泊松方程
文件：poisson_equation_student.py
重要：函数名称必须与参考答案一致！
"""

#!/usr/bin/env python3
"""
求解正负电荷构成的泊松方程
文件：poisson_equation_solution.py

import numpy as np  # 导入NumPy库用于数值计算
import matplotlib.pyplot as plt  # 导入Matplotlib库用于数据可视化
from typing import Tuple  # 导入Tuple用于类型注解

def solve_poisson_equation(M: int = 100, target: float = 1e-6, max_iterations: int = 10000) -> Tuple[np.ndarray, int, bool]:
    """
    Solve 2D Poisson equation using relaxation method.
    
    Args:
        M (int): Number of grid points per side  # 网格每边的点数，默认100
        target (float): Convergence tolerance  # 收敛精度，默认1e-6
        max_iterations (int): Maximum number of iterations  # 最大迭代次数，默认10000
    
    Returns:
        tuple: (phi, iterations, converged)
            phi (np.ndarray): Electric potential distribution  # 电势分布数组
            iterations (int): Number of iterations performed  # 实际迭代次数
            converged (bool): Whether solution converged  # 是否收敛标志
    """
    # Grid spacing
    h = 1.0  # 网格间距设为1.0（无量纲）
    
    # Initialize potential array with boundary conditions
    phi = np.zeros((M+1, M+1), dtype=float)  # 创建(M+1)x(M+1)的全零电势数组
    phi_prev = np.copy(phi)  # 创建phi的副本用于存储前一次迭代的结果
    
    # Set up charge density distribution
    rho = np.zeros((M+1, M+1), dtype=float)  # 创建与phi相同大小的电荷密度数组并初始化为0
    
    # Scale charge positions based on grid size
    # For M=100: pos (60:80, 20:40), neg (20:40, 60:80)
    pos_y1, pos_y2 = int(0.6*M), int(0.8*M)  # 计算正电荷在y方向的起始和结束位置
    pos_x1, pos_x2 = int(0.2*M), int(0.4*M)  # 计算正电荷在x方向的起始和结束位置
    neg_y1, neg_y2 = int(0.2*M), int(0.4*M)  # 计算负电荷在y方向的起始和结束位置
    neg_x1, neg_x2 = int(0.6*M), int(0.8*M)  # 计算负电荷在x方向的起始和结束位置
    
    rho[pos_y1:pos_y2, pos_x1:pos_x2] = 1.0   # Positive charge: 设置正电荷区域密度为+1
    rho[neg_y1:neg_y2, neg_x1:neg_x2] = -1.0  # Negative charge: 设置负电荷区域密度为-1
    
    # Relaxation iteration
    delta = 1.0  # 初始化最大变化量
    iterations = 0  # 初始化迭代计数器
    converged = False  # 初始化收敛标志为False
    
    while delta > target and iterations < max_iterations:  # 迭代循环：当变化量大于阈值且未达到最大迭代次数时继续
        # Update interior points using finite difference formula
        phi[1:-1, 1:-1] = 0.25 * (phi[0:-2, 1:-1] + phi[2:, 1:-1] +  # 使用五点差分格式更新内部点
                                   phi[1:-1, :-2] + phi[1:-1, 2:] + 
                                   h*h * rho[1:-1, 1:-1])
        
        # Calculate maximum change for convergence check
        delta = np.max(np.abs(phi - phi_prev))  # 计算当前解与前一次解的最大绝对差异
        
        # Update previous solution
        phi_prev = np.copy(phi)  # 将当前解复制给phi_prev，用于下一次迭代比较
        iterations += 1  # 迭代计数器加1
    
    converged = bool(delta <= target)  # 根据最终变化量设置收敛标志
    
    return phi, iterations, converged  # 返回电势分布、迭代次数和收敛状态

def visualize_solution(phi: np.ndarray, M: int = 100) -> None:
    """
    Visualize the electric potential distribution.
    
    Args:
        phi (np.ndarray): Electric potential array  # 电势分布数组
        M (int): Grid size  # 网格大小
    """
    plt.figure(figsize=(10, 8))  # 创建10x8英寸的图形
    
    # Create potential plot
    im = plt.imshow(phi, extent=[0, M, 0, M], origin='lower',  # 绘制电势分布图，设置坐标范围和原点
                    cmap='RdBu_r', interpolation='bilinear')  # 使用红蓝渐变色和双线性插值
    
    # Add colorbar
    cbar = plt.colorbar(im)  # 添加颜色条
    cbar.set_label('Electric Potential (V)', fontsize=12)  # 设置颜色条标签
    
    # Mark charge locations
    plt.fill_between([20, 40], [60, 60], [80, 80], alpha=0.3, color='red', label='Positive Charge')  # 标记负电荷区域（红）
    plt.fill_between([60, 80], [20, 20], [40, 40], alpha=0.3, color='blue', label='Negative Charge')  # 标记正电荷区域（蓝）
    
    # Add labels and title
    plt.xlabel('x (grid points)', fontsize=12)  # 设置x轴标签
    plt.ylabel('y (grid points)', fontsize=12)  # 设置y轴标签
    plt.title('Electric Potential Distribution\nPoisson Equation with Positive and Negative Charges', fontsize=14)  # 设置标题
    plt.legend()  # 显示图例
    
    # Add grid
    plt.grid(True, alpha=0.3)  # 添加半透明网格线
    
    plt.tight_layout()  # 自动调整子图参数
    plt.show()  # 显示图形

def analyze_solution(phi: np.ndarray, iterations: int, converged: bool) -> None:
    """
    Analyze and print solution statistics.
    
    Args:
        phi (np.ndarray): Electric potential array  # 电势分布数组
        iterations (int): Number of iterations  # 迭代次数
        converged (bool): Convergence status  # 收敛状态
    """
    print(f"Solution Analysis:")  # 打印分析标题
    print(f"  Iterations: {iterations}")  # 打印迭代次数
    print(f"  Converged: {converged}")  # 打印是否收敛
    print(f"  Max potential: {np.max(phi):.6f} V")  # 打印最大电势值
    print(f"  Min potential: {np.min(phi):.6f} V")  # 打印最小电势值
    print(f"  Potential range: {np.max(phi) - np.min(phi):.6f} V")  # 打印电势范围
    
    # Find locations of extrema
    max_idx = np.unravel_index(np.argmax(phi), phi.shape)  # 找到最大电势的索引位置
    min_idx = np.unravel_index(np.argmin(phi), phi.shape)  # 找到最小电势的索引位置
    print(f"  Max potential location: ({max_idx[0]}, {max_idx[1]})")  # 打印最大电势位置
    print(f"  Min potential location: ({min_idx[0]}, {min_idx[1]})")  # 打印最小电势位置

if __name__ == "__main__":
    # Solve the Poisson equation
    print("Solving 2D Poisson equation with relaxation method...")  # 开始求解提示
    
    # Parameters
    M = 100  # 网格大小设为100x100
    target = 1e-6  # 收敛精度设为1e-6
    max_iter = 10000  # 最大迭代次数设为10000
    
    # Solve
    phi, iterations, converged = solve_poisson_equation(M, target, max_iter)  # 调用求解函数
    
    # Analyze results
    analyze_solution(phi, iterations, converged)  # 调用分析函数
    
    # Visualize
    visualize_solution(phi, M)  # 调用可视化函数
    
    # Additional analysis: potential along center lines
    plt.figure(figsize=(12, 5))  # 创建12x5英寸的新图形
    
    # Horizontal cross-section
    plt.subplot(1, 2, 1)  # 创建左侧子图
    center_y = M // 2  # 计算y方向中心线位置
    plt.plot(phi[center_y, :], 'b-', linewidth=2)  # 绘制y=50水平线上的电势分布
    plt.xlabel('x (grid points)')  # 设置x轴标签
    plt.ylabel('Potential (V)')  # 设置y轴标签
    plt.title(f'Potential along y = {center_y}')  # 设置子图标题
    plt.grid(True, alpha=0.3)  # 添加半透明网格
    
    # Vertical cross-section
    plt.subplot(1, 2, 2)  # 创建右侧子图
    center_x = M // 2  # 计算x方向中心线位置
    plt.plot(phi[:, center_x], 'r-', linewidth=2)  # 绘制x=50垂直线上的电势分布
    plt.xlabel('y (grid points)')  # 设置x轴标签
    plt.ylabel('Potential (V)')  # 设置y轴标签
    plt.title(f'Potential along x = {center_x}')  # 设置子图标题
    plt.grid(True, alpha=0.3)  # 添加半透明网格
    
    plt.tight_layout()  # 自动调整子图参数
    plt.show()  # 显示图形
