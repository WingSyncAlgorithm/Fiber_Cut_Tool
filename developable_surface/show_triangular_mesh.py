import numpy as np
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Mesh:
    def __init__(self, stl_file):
        self.stl_file = stl_file
        self.vertices = np.array([])  # Initialize as an empty array
        self.triangles = None
        self.length = None

    def parse_vertex(self, line):
        match = re.match(
            r'vertex\s+([\d.e+-]+)\s+([\d.e+-]+)\s+([\d.e+-]+)', line)
        if match:
            vertex = np.array([float(match.group(1)), float(
                match.group(2)), float(match.group(3))])
            return vertex if not np.all(np.isnan(vertex)) else None
        else:
            return None

    def read_stl(self):
        vertices = []

        with open(self.stl_file, 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith('vertex'):
                    vertex = self.parse_vertex(line)
                    if vertex is not None:
                        vertices.append(vertex)

        self.vertices = np.array(vertices)

        # Assuming each triangle is defined by three consecutive vertices
        if self.vertices.shape[0] % 3 == 0:
            num_triangles = self.vertices.shape[0] // 3
            self.triangles = np.zeros((num_triangles, 3, 3), dtype=int)

            for i in range(num_triangles):
                self.triangles[i] = self.vertices[i * 3: (i + 1) * 3]

    def plot_mesh(self):
        if self.triangles is not None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            for triangle in self.triangles:
                for i in range(3):
                    ax.plot([triangle[i][0], triangle[(i + 1) % 3][0]],
                            [triangle[i][1], triangle[(i + 1) % 3][1]],
                            [triangle[i][2], triangle[(i + 1) % 3][2]], c='r')

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.show()


if __name__ == "__main__":
    mesh = Mesh('arc.stl')
    mesh.read_stl()
    mesh.plot_mesh()
