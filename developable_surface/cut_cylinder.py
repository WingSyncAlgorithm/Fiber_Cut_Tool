import numpy as np
from stl import mesh
import tripy


def cut_stl_along_edge(stl_file_path, edge_vertices):
    # 讀取STL檔案
    mesh_data = mesh.Mesh.from_file(stl_file_path)

    # 定義切割邊緣的兩個端點
    point_a = np.array(edge_vertices[0])
    point_b = np.array(edge_vertices[1])

    # 找到切割平面上的所有點
    cut_points = []
    for triangle in mesh_data:
        for vertex in triangle:
            vertex = np.array(vertex)
            if np.all((vertex - point_a) * (point_b - point_a) >= 0):
                cut_points.append(vertex)

    # 使用triangulate函數將切割平面上的點三角化
    triangles = tripy.earclip(cut_points)

    # 創建新的STL模型
    new_mesh = mesh.Mesh(np.zeros(len(triangles), dtype=mesh.Mesh.dtype))
    for i, triangle in enumerate(triangles):
        for j in range(3):
            new_mesh.vectors[i][j] = triangle[j]

    # 保存新的STL檔案
    new_mesh.save('cut_result.stl')


# 指定STL檔案路徑和切割邊緣的端點座標
stl_file_path = 'cylinder.stl'
edge_vertices = [(10.16, 0, 0),
                 (-10.16, 0, 0)]

# 執行切割
cut_stl_along_edge(stl_file_path, edge_vertices)
