import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from stl import mesh
import numpy as np

# 載入STL文件


def load_stl(file_path):
    mesh_data = mesh.Mesh.from_file(file_path)
    return mesh_data.vectors

# 顯示3D圖


def plot_3d(vectors):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 繪製三角網格
    poly3d = Poly3DCollection(vectors, edgecolor='k')
    ax.add_collection3d(poly3d)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


if __name__ == "__main__":
    stl_file_path = "arc.stl"
    vectors = load_stl(stl_file_path)
    plot_3d(vectors)
