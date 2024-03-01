import math
import matplotlib.pyplot as plt
from read_stl import TriangleMesh


def trans_x(s1x, s2x, s1y, s2y, sin_val, cos_val, len_change):
    """
    給定點s1=(s1x, s1y)、點s2=(s2x, s2y)，透過旋轉矩陣計算第三點s3的x座標

    參數：
    - s1x (float): s1的x座標
    - s2x (float): s2的x座標
    - s1y (float): s1的y座標
    - s2y (float): s2的y座標
    - sin_val (float): 旋轉角的正弦值sin(s2s1s3)
    - cos_val (float): 旋轉角的餘弦值cos(s2s1s3)
    - len_change (float): 長度變化因子=(length of s1s3)/(length of s1s2)

    返回：
    - s3x (float): s3的x座標
    """

    # 計算旋轉矩陣轉換
    Re = (cos_val * s2x) - (cos_val * s1x) - (sin_val * s2y) + (sin_val * s1y)
    s3x = s1x + len_change * Re
    return s3x


def trans_y(s1x, s2x, s1y, s2y, sin_val, cos_val, len_change):
    """
    給定點s1=(s1x, s1y)、點s2=(s2x, s2y)，透過旋轉矩陣計算第三點s3的y座標

    參數：
    - s1x (float): s1的x座標
    - s2x (float): s2的x座標
    - s1y (float): s1的y座標
    - s2y (float): s2的y座標
    - sin_val (float): 旋轉角的正弦值sin(s2s1s3)
    - cos_val (float): 旋轉角的餘弦值cos(s2s1s3)
    - len_change (float): 長度變化因子=(length of s1s3)/(length of s1s2)

    返回：
    - s3x (float): s3的y座標
    """
    # 計算旋轉矩陣轉換
    Re = (sin_val * s2x) - (sin_val * s1x) + (cos_val * s2y) - (cos_val * s1y)
    s3y = s1y + len_change * Re
    return s3y


def calculate_third_point(s1, s2, length_12, length_13, length_23):
    """
    根據給定的兩點s1, s2和三邊長length_12, length_13, length_23，計算第三個點s3的位置。
    s1->s2->s3為順時針方向

    參數：
    - s1 (list): 第一點的座標 [s1x, s1y]
    - s2 (list): 第二點的座標 [s2x, s2y]
    - length_12 (float): s1 到 s2 的距離
    - length_13 (float): s1 到第三點s3的距離
    - length_23 (float): s2 到第三點s3的距離

    返回：
    - list: 第三點的座標 [s3x, s3y]
    """
    cos312 = (length_12**2 + length_13**2 - length_23**2) / \
        (2.0 * length_12 * length_13)  # 餘弦定理
    sin312_1 = math.sqrt(1 - cos312**2)
    sin312_2 = -math.sqrt(1 - cos312**2)
    length_change = length_13 / length_12

    s3x1 = trans_x(s1[0], s2[0], s1[1], s2[1], sin312_1, cos312, length_change)
    s3y1 = trans_y(s1[0], s2[0], s1[1], s2[1], sin312_1, cos312, length_change)
    s3x2 = trans_x(s1[0], s2[0], s1[1], s2[1], sin312_2, cos312, length_change)
    s3y2 = trans_y(s1[0], s2[0], s1[1], s2[1], sin312_2, cos312, length_change)
    return [s3x1, s3y1]


def flatten(mesh, s1, s2, s3):
    """
    將三角網格的指定三點平坦化，並遞迴處理相鄰的點。

    Parameters:
    - mesh: TriangleMesh 物件，包含三角網格的相關資訊。
    - s1, s2, s3: 三個點的索引，用於指定要平坦化的三角形。
    """
    s3 = mesh.connect[s1, s2,0]
    mesh.s[s3, :] = calculate_third_point(
        mesh.s[s1], mesh.s[s2], mesh.length[s1, s2], mesh.length[s1, s3], mesh.length[s2, s3])
    s4 = mesh.connect[s1, s3,0]
    s5 = mesh.connect[s3, s2,0]
    if s4 != -1 and mesh.s[s4, 0] == 99999999:
        flatten(mesh, s1, s3, s4)
    if s5 != -1 and mesh.s[s5, 0] == 99999999:
        flatten(mesh, s3, s2, s5)


mesh = TriangleMesh('cylinder.stl')
print(mesh.start_edges)
#print("mesh.length",mesh.start_edges[0][0],mesh.start_edges[0][1])
mesh.start_edges = [mesh.start_edges[0][:]]
for i, start_edge in enumerate(mesh.start_edges):
    start_point1 = start_edge[0]
    start_point2 = start_edge[1]
    mesh.s[start_point1, :] = [i*10, 0]
    mesh.s[start_point2, :] = [i*10+mesh.length[start_point1, start_point2], 0]

    next_point1 = mesh.connect[start_point1, start_point2,0]
    next_point2 = mesh.connect[start_point2, start_point1,0]
    if next_point1 != -1:
        flatten(mesh, start_point1, start_point2, next_point1)
    if next_point2 != -1:
        flatten(mesh, start_point2, start_point1, next_point2)


# 提取 x 和 y 值
x_value = mesh.s[:, 0]
y_value = mesh.s[:, 1]

# 繪製單一點的散點圖
plt.scatter(x_value, y_value, color='red', marker='o')
plt.axis('equal')
# 添加標籤和標題
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scatter Plot of a Single Point')

# 顯示圖例
plt.legend()

# 顯示圖形
plt.show()
