import numpy as np
from stl import mesh

# 讀取STL檔案


def read_stl(file_path):
    mesh_data = mesh.Mesh.from_file(file_path)
    return mesh_data

# 估計圓柱的半徑和高度


def estimate_cylinder_params(stl_mesh):
    # 假設STL檔案的底部點是圓柱的中心
    bottom_center = np.mean(stl_mesh.points, axis=0)

    # 假設STL檔案的最高點是圓柱的頂部
    top_center = np.max(stl_mesh.points, axis=0)

    # 估計半徑
    radius = np.linalg.norm(bottom_center[:2] - top_center[:2]) / 2.0

    # 估計高度
    height = top_center[2] - bottom_center[2]

    return radius, height

# 切割圓柱


# 切割圓柱
def cut_cylinder(stl_mesh, radius, height):
    # 定義切割平面
    z_plane = height

    # 過濾點，只保留在切割平面上方的點
    points_above_plane = stl_mesh.points[stl_mesh.points[:, 2] > z_plane]

    # 計算上方點的索引
    indices_above_plane = np.where(stl_mesh.points[:, 2] > z_plane)[0]

    # 切割三角形
    new_triangles = []
    for triangle in stl_mesh.vectors:
        # 檢查三角形的所有頂點是否在切割平面的同一側
        if all(index in indices_above_plane for vertex in triangle for index in vertex):
            new_triangles.append(triangle)

    # 確保有符合條件的三角形才進行合併
    if new_triangles:
        # 創建新的STL物件
        new_mesh = mesh.Mesh(np.concatenate(new_triangles, axis=0))
        return new_mesh
    else:
        print("No triangles above the cutting plane.")
        return None


# 保存STL檔案


# 保存STL檔案
def save_stl(file_path, stl_mesh):
    if stl_mesh is not None:
        stl_mesh.save(file_path)
        print(f"STL檔案已成功保存到 {file_path}")
    else:
        print("無法保存STL檔案，因為沒有符合條件的三角形。")


# 使用範例
input_file = "cylinder.stl"
output_file = "output_cut.stl"

# 讀取STL檔案
cylinder_mesh = read_stl(input_file)

# 估計圓柱的半徑和高度
estimated_radius, estimated_height = estimate_cylinder_params(cylinder_mesh)

# 切割圓柱
cut_mesh = cut_cylinder(
    cylinder_mesh, radius=estimated_radius, height=estimated_height)

# 保存切割後的STL檔案
save_stl(output_file, cut_mesh)
