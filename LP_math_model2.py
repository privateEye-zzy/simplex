'''
线性规划实战—连续投资问题
考虑下列投资项目：
项目A：在第1~4年每年年初可以投资，并于次年年末收回本利115%；
项目B：第3年年初可以投资，一直到第5年年末能收回本利125%，且规定最大投资额不超过4万元；
项目C：第2年年初可以投资，一直到第5年年末能收回本利140%，且规定最大投资额不超过3万元；
项目D：5年内每一年年初均可以购买公债，并于当年年末归还本金，并加获得利息6%。
如果你有10万元本金，求确定一个有效的投资方案，使得第5年年末你拥有的资金的本利总额最大？
'''
import numpy as np
from copy import deepcopy
np.set_printoptions(precision=4, suppress=True, threshold=np.inf)
# 线性规划转化为松弛形式
def get_loose_matrix(matrix):
    row, col = matrix.shape
    loose_matrix = np.zeros((row, row + col))
    for i, _ in enumerate(loose_matrix):
        loose_matrix[i, 0: col] = matrix[i]
        loose_matrix[i, col + i] = 1.0  # 对角线
    return loose_matrix
# 松弛形式的系数矩阵A、约束矩阵B和目标函数矩阵C组合为一个矩阵
def join_matrix(a, b, c):
    row, col = a.shape
    s = np.zeros((row + 1, col + 1))
    s[1:, 1:] = a  # 右下角是松弛系数矩阵A
    s[1:, 0] = b  # 左下角是约束条件值矩阵B
    s[0, 1: len(c) + 1] = c  # 右上角是目标函数矩阵C
    return s
# 旋转矩阵—替换替出替入变量的角色位置
def pivot_matrix(matrix, k, j):
    # 单独处理替出变量所在行，需要除以替入变量的系数matrix[k][j]
    matrix[k] = matrix[k] / matrix[k][j]
    # 循环除了替出变量所在行之外的所有行
    for i, _ in enumerate(matrix):
        if i != k:
            matrix[i] = matrix[i] - matrix[k] * matrix[i][j]
# 根据旋转后的矩阵，从基本变量数组中得到一组基解
def get_base_solution(matrix, base_ids):
    X = [0.0] * (matrix.shape[1])  # 解空间
    for i, _ in enumerate(base_ids):
        X[base_ids[i]] = matrix[i + 1][0]
    return X
# 构造辅助线性规划
def Laux(matrix, base_ids):
    l_matrix = np.copy(matrix)
    # 辅助矩阵的最后一列存放x0的系数，初始化为-1
    l_matrix = np.column_stack((l_matrix, [-1] * l_matrix.shape[0]))
    # 辅助线性函数的目标函数为z = x0
    l_matrix[0, :-1] = 0.0
    l_matrix[0, -1] = 1
    k = l_matrix[1:, 0].argmin() + 1  # 选择一个b最小的那一行的基本变量作为替出变量
    j = l_matrix.shape[1] - 1   # 选择x0作为替入变量
    # 第一次旋转矩阵，使得所有b为正数
    pivot_matrix(l_matrix, k=k, j=j)
    base_ids[k - 1] = j  # 维护基本变量索引数组
    # 用单纯形算法求解该辅助线性规划
    l_matrix = simplex(l_matrix, base_ids)
    # 如果求解后的辅助线性规划中x0仍然是基本变量，需要再次旋转消去x0
    if l_matrix.shape[1] - 1 in base_ids:
        j = np.where(l_matrix[0, 1:] != 0)[0][0] + 1   # 找到矩阵第一行(目标函数)系数不为0的变量作为替入变量
        k = base_ids.index(l_matrix.shape[1] - 1) + 1  # 找到x0作为基本变量所在的那一行，将x0作为替出变量
        pivot_matrix(l_matrix, k=k, j=j)   # 旋转矩阵消去基本变量x0
        base_ids[k - 1] = j  # 维护基本变量索引数组
    return l_matrix, base_ids
# 从辅助函数中恢复原问题的目标函数
def resotr_from_Laux(l_matrix, z, base_ids):
    z_ids = np.where(z != 0)[0] - 1  # 得到目标函数系数不为0的索引数组(即基本变量索引数组)
    restore_matrix = np.copy(l_matrix[:, 0:-1])  # 去掉x0那一列
    restore_matrix[0] = z  # 初始化矩阵的第一行为原问题的目标函数向量
    for i, base_v in enumerate(base_ids):
        # 如果原问题的基本变量存在新基本变量数组中，说明需要替换消去
        if base_v in z_ids:
            restore_matrix[0] -= restore_matrix[0, base_v + 1] * restore_matrix[i + 1]  # 消去原目标函数中的基本变量
    return restore_matrix
# 单纯形算法求解线性规划
def simplex(matrix, base_ids):
    matrix = matrix.copy()
    # 如果目标系数向量里有负数，则旋转矩阵
    while matrix[0, 1:].min() < 0:
        # 在目标函数向量里，选取系数为负数的第一个变量索引，作为替入变量
        j = np.where(matrix[0, 1:] < 0)[0][0] + 1
        # 在约束集合里，选取对替入变量约束最紧的约束行，那一行的基本变量作为替出变量
        k = np.array([matrix[i][0] / matrix[i][j] if matrix[i][j] > 0 else 0x7fff for i in
                      range(1, matrix.shape[0])]).argmin() + 1
        # print('替出变量为：{}，替入变量为：{}， 值为：{}'.format(k, j, matrix[k][j]))
        # 说明原问题无界
        if matrix[k][j] <= 0:
            print('原问题无界')
            return None, None
        pivot_matrix(matrix, k, j)  # 旋转替换替入变量和替出变量
        base_ids[k - 1] = j - 1  # 维护当前基本变量索引数组
    return matrix
# 单纯形算法求解步骤入口
def solve(a, b, c, equal=None):
    loose_matrix = get_loose_matrix(a)  # 转化得到松弛矩阵
    if equal is not None:
        for i, e in enumerate(equal):
            loose_matrix = np.insert(loose_matrix, i, e, axis=0)
    matrix = join_matrix(loose_matrix, b, c)  # 得到ABC的组合矩阵
    base_ids = list(range(len(c), len(b) + len(c)))  # 初始化基本变量的索引数组
    # 约束系数矩阵有负数约束，证明没有可行解，需要辅助线性函数
    if matrix[:, 0].min() < 0:
        print('构造求解辅助线性规划函数...')
        l_matrix, base_ids = Laux(matrix, base_ids)  # 构造辅助线性规划函数并旋转求解之
        if l_matrix is not None:
            matrix = resotr_from_Laux(l_matrix, matrix[0], base_ids)  # 恢复原问题的目标函数
        else:
            print('辅助线性函数的原问题没有可行解')
            return None, None, None
    ret_matrix = simplex(matrix, base_ids)  # 单纯形算法求解拥有基本可行解的线性规划
    X = get_base_solution(ret_matrix, base_ids)  # 得到当前最优基本可行解
    if ret_matrix is not None:
        return matrix, ret_matrix, X
    else:
        print('原线性规划问题无界')
        return None, None, None
if __name__ == '__main__':
    x = [0 for _ in range(20)]  # 定义决策变量
    e1, e2, e3, e4, e5 = deepcopy(x), deepcopy(x), deepcopy(x), deepcopy(x), deepcopy(x)  # 等式约束
    e1[0], e1[15] = 1, 1
    e2[1], e2[11], e2[16], e2[15] = 1, 1, 1, -1.06
    e3[2], e3[7], e3[17], e3[0], e3[16] = 1, 1, 1, -1.15, -1.06
    e4[3], e4[18], e4[1], e4[17] = 1, 1, -1.15, -1.06
    e5[19], e5[2], e5[18] = 1, -1.15, -1.06
    a1, a2 = deepcopy(x), deepcopy(x)  # 不等式约束
    a1[7] = 1
    a2[11] = 1
    a = np.array([a1, a2])
    b = [10, 0, 0, 0, 0] + [4, 3]
    equal = []
    equal.append(e1 + [0] * a.shape[0])
    equal.append(e2 + [0] * a.shape[0])
    equal.append(e3 + [0] * a.shape[0])
    equal.append(e4 + [0] * a.shape[0])
    equal.append(e5 + [0] * a.shape[0])
    c = deepcopy(x)  # 目标函数
    c[3], c[7], c[11], c[19] = -1.15, -1.25, -1.4, -1.06
    # 单纯形法求解数学模型
    matrix, ret_matrix, X = solve(a, b, c, equal=equal)
    Y = np.round(np.array(X[0: len(c)]).reshape(4, -1).T, 4)  # 得到最优投资方案
    print(Y)
    for i, y in enumerate(Y):
        print('第{}年年初投资组合：项目A：{}万元，项目B：{}万元，项目C：{}万元，项目D：{}万元，该年总投入：{}万元'.format(
            i + 1, y[-4], y[-3], y[-2], y[-1], np.sum(y)))
    total_money = np.abs(np.round(-ret_matrix[0][0], 4))  # 总收入本息
    profit = np.round((total_money - b[0]) / b[0] * 100, 4)  # 总赢利
    print('第5年年末总收入本金+利息为：{}万元，总赢利：{}%'.format(total_money, profit))
