import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def generate_cylinder_points(radius, height, num_points, num_slices):
    theta = np.linspace(0, 2 * np.pi, num_points)
    z = np.linspace(0, height, num_slices)
    theta, z = np.meshgrid(theta, z)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return x, y, z


def plot_cylinder_surface(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, cmap='viridis', edgecolor='k', linewidth=0.1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


# 設定圓柱參數
cylinder_radius = 1.0
cylinder_height = 3.0
num_points = 100
num_slices = 100

# 生成圓柱點雲
x, y, z = generate_cylinder_points(
    cylinder_radius, cylinder_height, num_points, num_slices)

# 繪製圓柱側面表面
plot_cylinder_surface(x, y, z)
points = []
for s in range(len(x.flatten())):
    points.append([x.flatten()[s], y.flatten()[s], z.flatten()[s]])


with open("large_cylinder.stl", "w") as stl_file:
    # Write STL header
    stl_file.write("solid cylinder\n")

    # Write each triangular face
    for i in range(num_slices-1):
        for j in range(num_points):
            idx1 = num_points*i+j
            idx2 = num_points*i+num_points+(j+1) % num_points
            idx3 = num_points*i+(j+1) % num_points
            print([idx1, idx2, idx3])
            # Write normal vector
            stl_file.write(f"  facet normal 0.0 0.0 0.0\n")
            stl_file.write("    outer loop\n")

            # Write vertices
            for idx in [idx1, idx2, idx3]:
                stl_file.write(
                    f"      vertex {x.flatten()[idx]} {y.flatten()[idx]} {z.flatten()[idx]}\n")
            # Close loop
            stl_file.write("    endloop\n")
            stl_file.write("  endfacet\n")
            # Write normal vector
            stl_file.write(f"  facet normal 0.0 0.0 0.0\n")
            stl_file.write("    outer loop\n")
            idx1 = num_points*i+j
            idx2 = idx1+num_points
            idx3 = num_points*i+num_points+(j+1) % num_points
            print([idx1, idx2, idx3])
            # Write vertices
            for idx in [idx1, idx2, idx3]:
                stl_file.write(
                    f"      vertex {x.flatten()[idx]} {y.flatten()[idx]} {z.flatten()[idx]}\n")
            # Close loop
            stl_file.write("    endloop\n")
            stl_file.write("  endfacet\n")

    # Write STL footer
    stl_file.write("endsolid cylinder\n")
