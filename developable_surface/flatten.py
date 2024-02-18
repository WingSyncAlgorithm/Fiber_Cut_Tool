import math
import matplotlib.pyplot as plt
from read_stl import TriangleMesh


def trans_x(s1x, s2x, s1y, s2y, sin_val, cos_val, len_change):
    Re = (cos_val * s2x) - (cos_val * s1x) - \
        (sin_val * s2y) + (sin_val * s1y)  # 旋轉矩陣轉換
    s3x = s1x + len_change * Re
    return s3x


def trans_y(s1x, s2x, s1y, s2y, sin_val, cos_val, len_change):
    Re = (sin_val * s2x) - (sin_val * s1x) + \
        (cos_val * s2y) - (cos_val * s1y)  # 旋轉矩陣轉換
    s3y = s1y + len_change * Re
    return s3y


def calculate_third_point(s1, s2, length_12, length_13, length_23):
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
    s3 = mesh.connect[s1, s2]
    mesh.s[s3, :] = calculate_third_point(
        mesh.s[s1], mesh.s[s2], mesh.length[s1, s2], mesh.length[s1, s3], mesh.length[s2, s3])
    s4 = mesh.connect[s1, s3]
    s5 = mesh.connect[s3, s2]
    if s4 != -1 and mesh.s[s4, 0] == 99999999:
        flatten(mesh, s1, s3, s4)
    if s5 != -1 and mesh.s[s5, 0] == 99999999:
        flatten(mesh, s3, s2, s5)


mesh = TriangleMesh('cylinder.stl')
print(mesh.vertices)
for i, start_edge in enumerate(mesh.start_edges):
    start_point1 = start_edge[0]
    start_point2 = start_edge[1]
    mesh.s[start_point1, :] = [i*10, 0]
    mesh.s[start_point2, :] = [i*10+mesh.length[start_point1, start_point2], 0]

    next_point1 = mesh.connect[start_point1, start_point2]
    next_point2 = mesh.connect[start_point2, start_point1]
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
print(mesh.triangles)
