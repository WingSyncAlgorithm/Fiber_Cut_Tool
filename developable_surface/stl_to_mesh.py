from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import tripy
from stl import mesh


def stl_to_point_cloud(stl_filename, num_points=10000):
    # 加載STL文件
    mesh_data = mesh.Mesh.from_file(stl_filename)

    # 提取STL文件的所有三角面片的頂點
    all_points = mesh_data.vectors.reshape(-1, 3)

    # 隨機選取一部分頂點，以減少點的數量
    selected_points_indices = np.random.choice(
        len(all_points), num_points, replace=False)
    selected_points = all_points[selected_points_indices]

    return selected_points


# 指定STL文件的路徑
stl_filename = 'cylinder.stl'

# 將STL轉換為點雲
point_cloud = stl_to_point_cloud(stl_filename)


def plot_point_cloud(point_cloud):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 提取點雲的x、y、z坐標
    x = point_cloud[:, 0]
    y = point_cloud[:, 1]
    z = point_cloud[:, 2]

    # 繪製散點圖
    ax.scatter(x, y, z, c='blue', marker='o', s=10)

    ax.set_xlabel('X 軸')
    ax.set_ylabel('Y 軸')
    ax.set_zlabel('Z 軸')

    plt.show()


# 使用前面示例中的 stl_to_point_cloud 函數獲得點雲
point_cloud = stl_to_point_cloud(stl_filename)

# 繪製點雲
plot_point_cloud(point_cloud)
