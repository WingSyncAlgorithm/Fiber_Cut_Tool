import numpy as np
from stl import Mesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
#from memory_profiler import profile


class Point:
    def __init__(self, x, y, z, triangle_idx, p_idx):
        self.x = x
        self.y = y
        self.z = z
        self.triangle_idx = triangle_idx
        self.p_idx = p_idx

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __gt__(self, other):
        if self.x != other.x:
            return self.x > other.x
        elif self.y != other.y:
            return self.y > other.y
        else:
            return self.z > other.z

    def __lt__(self, other):
        if self.x != other.x:
            return self.x < other.x
        elif self.y != other.y:
            return self.y < other.y
        else:
            return self.z < other.z


class TriangleMesh:
    def __init__(self, stl_file):
        self.stl = stl_file
        stl = Mesh.from_file(self.stl)
        x = stl.x.reshape(np.size(stl.x), -1)
        y = stl.y.reshape(np.size(stl.y), -1)
        z = stl.z.reshape(np.size(stl.z), -1)
        self.vertices = np.unique(np.concatenate((x, y, z), axis=1), axis=0)
        self.num_original_vertices = np.size(self.vertices, axis=0)
        self.num_vertices = np.size(self.vertices, axis=0)
        self.triangles = np.zeros((np.size(stl.vectors, axis=0), 3), dtype=int)
        self.high_curvature_points = np.full(0, -1, dtype=int)
        # 2d list --> key: point index, value: triangle index list
        self.point_triangle_adj = [[] for i in range(self.num_vertices)]

        # 轉換三角形儲存格式
        print("p1")
        st = time.time()
        temp = [Point for i in range(np.size(stl.vectors, axis=0)*3)]
        cnt = 0
        for triangle_idx in range(np.size(stl.vectors, axis=0)):
            for vertex in range(3):
                temp[cnt] = Point(stl.vectors[triangle_idx, vertex, 0], stl.vectors[triangle_idx,
                                  vertex, 1], stl.vectors[triangle_idx, vertex, 2], triangle_idx, vertex)
                cnt += 1
        temp.sort()

        point_idx = 0
        self.triangles[temp[0].triangle_idx, temp[0].p_idx] = point_idx
        for i in range(1, np.size(temp)):
            if temp[i] != temp[i-1]:
                point_idx += 1
            self.triangles[temp[i].triangle_idx, temp[i].p_idx] = point_idx
            self.point_triangle_adj[point_idx].append(temp[i].triangle_idx)
            # self.point_triangle_adj[point_idx] = np.append(self.point_triangle_adj[point_idx],temp[i].triangle_idx)
        del temp

        print(time.time()-st)
        print("p2")
        st = time.time()

        print()

        # 初始化self.length

        self.length = np.full(
            (self.num_vertices, self.num_vertices), -1, dtype=float)
        self.calculate_length()

        # 計算三角形面積
        self.area = np.zeros(np.size(self.triangles), dtype=float)
        self.calculate_area()

        # 計算角度
        self.angle = np.zeros(
            (self.num_vertices, self.num_vertices, self.num_vertices), dtype=float)
        self.calculate_angle()

        # 計算高斯曲率
        #######################
        self.gaussian_curvature = np.zeros(self.num_vertices, dtype=float)
        self.calculate_gaussian_curvature()
        #################
        print(self.gaussian_curvature)
        # 尋找高斯曲率過大的點
        for vertex_idx in range(self.num_vertices):
            if self.gaussian_curvature[vertex_idx] > 0.001:
                self.high_curvature_points = np.append(
                    self.high_curvature_points, vertex_idx)
        #print("self.high_curvature_points",np.size(self.high_curvature_points))
        self.high_curvature_graph = np.full(
            [np.size(self.high_curvature_points), np.size(self.high_curvature_points)], -1, dtype=float)
        for i in range(np.size(self.high_curvature_points)):
            for j in range(i):
                point1_idx = self.high_curvature_points[i]
                point2_idx = self.high_curvature_points[j]
                if self.length[point1_idx, point2_idx] != -1:
                    self.high_curvature_graph[i, j] = (
                        self.gaussian_curvature[point1_idx] + self.gaussian_curvature[point2_idx]) / 2.0
                    self.high_curvature_graph[j, i] = self.high_curvature_graph[i, j]
        high_curvature_subgraph = self.separate_disconnected_components(
            self.high_curvature_graph)
        #print("high_curvature_subgraph",high_curvature_subgraph)
        '''
        x_data, y_data, z_data = [], [], []
        for i in range(np.size(self.high_curvature_graph, axis=0)):
            for j in range(np.size(self.high_curvature_graph, axis=1)):
                if self.high_curvature_graph[i][j] != -1:
                    point1_idx = self.high_curvature_points[i]
                    x_data.append(
                        self.vertices[point1_idx, 0])
                    y_data.append(
                        self.vertices[point1_idx, 1])
                    z_data.append(
                        self.vertices[point1_idx, 2])
        print("x_data",len(x_data))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_data, y_data, z_data, c='blue',
                   marker='o', s=1)  # c是顏色，marker是標記，s是大小

        # 設定座標軸標籤
        ax.set_xlabel('X 軸')
        ax.set_ylabel('Y 軸')
        ax.set_zlabel('Z 軸')
        ax.axis('equal')
        plt.show()
        '''
        # 執行切割
        self.start_edges = []
        for subgraph in range(np.size(high_curvature_subgraph, axis=0)):
            print(subgraph)
            max_cycle_cost, max_cycle_path = self.find_max_cycle_cost(
                high_curvature_subgraph[subgraph][:][:])
            print("self.length",self.length[max_cycle_path[0],max_cycle_path[1]])
            for point in range(np.size(max_cycle_path)):
                # print(max_cycle_path[point % np.size(
                #    max_cycle_path)], max_cycle_path[(point+1) % np.size(max_cycle_path)])
                self.cut_edge(max_cycle_path[point % np.size(
                    max_cycle_path)], max_cycle_path[(point+1) % np.size(max_cycle_path)], point == 0)
            # print(high_curvature_subgraph[subgraph][:][:])
        print("self.num_vertices",self.num_vertices-self.num_original_vertices)
        self.length = np.full(
            (self.num_vertices, self.num_vertices), -1, dtype=float)
        self.calculate_length()
        # 初始化儲存展開到平面的點的矩陣
        self.s = np.full((self.num_vertices, 2), 99999999, dtype=float)
        # 儲存由兩點所連接的第三點
        self.connect = np.full(
            (self.num_vertices, self.num_vertices), -1, dtype=int)

        for triangle_idx in range(np.size(self.triangles, axis=0)):
            self.connect[self.triangles[triangle_idx, 0],
                         self.triangles[triangle_idx, 1]] = self.triangles[triangle_idx, 2]
            self.connect[self.triangles[triangle_idx, 1],
                         self.triangles[triangle_idx, 2]] = self.triangles[triangle_idx, 0]
            self.connect[self.triangles[triangle_idx, 2],
                         self.triangles[triangle_idx, 0]] = self.triangles[triangle_idx, 1]

    def calculate_length(self):
        for triangle_idx in range(np.size(self.triangles, axis=0)):
            point1_idx = self.triangles[triangle_idx, 0]
            point2_idx = self.triangles[triangle_idx, 1]
            point3_idx = self.triangles[triangle_idx, 2]
            point1 = self.vertices[point1_idx, :]
            point2 = self.vertices[point2_idx, :]
            point3 = self.vertices[point3_idx, :]
            self.length[point1_idx, point2_idx] = np.sqrt(
                np.sum((point2 - point1)**2))
            self.length[point2_idx, point1_idx] = np.sqrt(
                np.sum((point2 - point1)**2))
            self.length[point1_idx, point3_idx] = np.sqrt(
                np.sum((point3 - point1)**2))
            self.length[point3_idx, point1_idx] = np.sqrt(
                np.sum((point3 - point1)**2))
            self.length[point2_idx, point3_idx] = np.sqrt(
                np.sum((point3 - point2)**2))
            self.length[point3_idx, point2_idx] = np.sqrt(
                np.sum((point3 - point2)**2))

    def cut_edge(self, vertex_idx1, vertex_idx2, start_flatten):
        '''
        vertex_idx1往vertex_idx2切割
        '''
        vertex_add_idx1 = 0
        vertex_add_idx2 = 0
        is_added1 = False
        is_added2 = False
        for added_idx in range(self.num_original_vertices, self.num_vertices):
            if (self.vertices[vertex_idx1, :] == self.vertices[added_idx, :]).all():
                vertex_add_idx1 = added_idx
                is_added1 = True

            if (self.vertices[vertex_idx2, :] == self.vertices[added_idx, :]).all():
                vertex_add_idx2 = added_idx
                is_added2 = True

        if is_added1 == False:
            vertex_add_idx1 = self.num_vertices
            self.vertices = np.append(
                self.vertices, [self.vertices[vertex_idx1, :]], axis=0)
            self.num_vertices = np.size(self.vertices, axis=0)
        if is_added2 == False:
            vertex_add_idx2 = self.num_vertices
            self.vertices = np.append(
                self.vertices, [self.vertices[vertex_idx2, :]], axis=0)
            self.num_vertices = np.size(self.vertices, axis=0)
        for triangle_idx in range(np.size(self.triangles, axis=0)):
            if self.triangles[triangle_idx, 0] == vertex_idx1:
                if self.triangles[triangle_idx, 1] == vertex_idx2:
                    print("9")
                    self.triangles[triangle_idx, 0] = vertex_add_idx1
                    self.triangles[triangle_idx, 1] = vertex_add_idx2
                    if start_flatten == True:
                        print("Yes")
                        self.start_edges.append(
                            [vertex_add_idx1, vertex_add_idx2])
                        self.start_edges.append([vertex_idx2, vertex_idx1])
            elif self.triangles[triangle_idx, 1] == vertex_idx1:
                if self.triangles[triangle_idx, 2] == vertex_idx2:
                    self.triangles[triangle_idx, 1] = vertex_add_idx1
                    self.triangles[triangle_idx, 2] = vertex_add_idx2
                    print("9")
                    if start_flatten == True:
                        print("Yes")
                        self.start_edges.append(
                            [vertex_add_idx1, vertex_add_idx2])
                        self.start_edges.append([vertex_idx2, vertex_idx1])
            elif self.triangles[triangle_idx, 2] == vertex_idx1:
                if self.triangles[triangle_idx, 0] == vertex_idx2:
                    self.triangles[triangle_idx, 2] = vertex_add_idx1
                    self.triangles[triangle_idx, 0] = vertex_add_idx2
                    print("9")
                    if start_flatten == True:
                        print("Yes")
                        self.start_edges.append(
                            [vertex_add_idx1, vertex_add_idx2])
                        self.start_edges.append([vertex_idx2, vertex_idx1])
        self.num_vertices = np.size(self.vertices, axis=0)

    def calculate_area(self):
        for triangle_idx in range(np.size(self.triangles, axis=0)):
            v1 = self.vertices[self.triangles[triangle_idx, 0], :]
            v2 = self.vertices[self.triangles[triangle_idx, 1], :]
            v3 = self.vertices[self.triangles[triangle_idx, 2], :]
            cross_product = np.cross(v2 - v1, v3 - v1)
            self.area[triangle_idx] = 0.5 * np.linalg.norm(cross_product)

    def calculate_angle(self):
        for triangle_idx in range(np.size(self.triangles, axis=0)):
            for point in range(3):
                point1_idx = self.triangles[triangle_idx, (point+1) % 3]
                point2_idx = self.triangles[triangle_idx, point]
                point3_idx = self.triangles[triangle_idx, (point+2) % 3]
                point1 = self.vertices[point1_idx, :]
                point2 = self.vertices[point2_idx, :]
                point3 = self.vertices[point3_idx, :]
                v1 = point1 - point2
                v2 = point3 - point2
                dot_product = np.dot(v1, v2)
                magnitude_v1 = np.linalg.norm(v1)
                magnitude_v2 = np.linalg.norm(v2)
                cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
                self.angle[point1_idx, point2_idx,
                           point3_idx] = np.arccos(cos_theta)
                self.angle[point3_idx, point2_idx,
                           point1_idx] = np.arccos(cos_theta)

    # @profile
    def calculate_gaussian_curvature(self):

        for vertex_idx in range(self.num_vertices):
            a_vertex = 0
            for triangle_idx in self.point_triangle_adj[vertex_idx]:
                a_vertex += self.area[triangle_idx]

            angles = self.angle[:, vertex_idx, :]
            sum_theta = 0
            for triangle_idx in self.point_triangle_adj[vertex_idx]:
                if self.triangles[triangle_idx][0] == vertex_idx:
                    sum_theta += angles[self.triangles[triangle_idx]
                                        [1], self.triangles[triangle_idx][2]]
                elif self.triangles[triangle_idx][1] == vertex_idx:
                    sum_theta += angles[self.triangles[triangle_idx]
                                        [2], self.triangles[triangle_idx][0]]
                elif self.triangles[triangle_idx][2] == vertex_idx:
                    sum_theta += angles[self.triangles[triangle_idx]
                                        [0], self.triangles[triangle_idx][1]]

            self.gaussian_curvature[vertex_idx] = (
                2*np.pi-sum_theta)/(a_vertex/3)
            # if self.gaussian_curvature[vertex_idx] > 0.01:
            # print(
            #   self.gaussian_curvature[vertex_idx], self.vertices[vertex_idx, :])

    def find_max_cycle_cost_helper(self, graph, start_node, current_node, visited, current_cost, max_cost, path, max_path):
        visited[current_node] = True
        path.append(current_node)
      #  print("find_max_cycle_cost_helper")

        for neighbor in range(len(graph)):
            if graph[current_node, neighbor] > 0:  # Check if there is an edge
                if not visited[neighbor]:
                    max_cost, max_path = self.find_max_cycle_cost_helper(
                        graph, start_node, neighbor, visited, current_cost + graph[current_node, neighbor], max_cost, path, max_path)
                elif neighbor == start_node:  # Found a cycle
                    if current_cost + graph[current_node, neighbor] > max_cost:
                        max_cost = current_cost + graph[current_node, neighbor]
                        max_path = path.copy()

        visited[current_node] = False
        path.pop()

        return max_cost, max_path

    def find_max_cycle_cost(self, graph):
        num_nodes = graph.shape[0]
        max_cost = float('-inf')
        max_path = []

        for start_node in range(num_nodes):
            visited = np.zeros(num_nodes, dtype=bool)
            cycle_cost, cycle_path = self.find_max_cycle_cost_helper(
                graph, start_node, start_node, visited, 0, max_cost, [], [])
            # print("find_max_cycle_cost", cycle_cost, cycle_path)
            if cycle_cost > max_cost:
                max_cost = cycle_cost
                max_path = cycle_path

        return max_cost, max_path

    def dfs(self, graph, start, visited, component):
        visited[start] = True
        component.append(start)

        for neighbor, weight in enumerate(graph[start]):
            if weight > 0 and not visited[neighbor]:
                self.dfs(graph, neighbor, visited, component)

    def connected_components(self, graph):
        n = len(graph)
        visited = np.zeros(n, dtype=bool)
        components = []

        for node in range(n):
            if not visited[node]:
                component = []
                self.dfs(graph, node, visited, component)
                components.append(component)

        return components

    def separate_disconnected_components(self, graph):
        components = self.connected_components(graph)

        # Create separate graphs for each connected component
        separated_graphs = []
        for component in components:
            separated_graph = np.full_like(graph,-1,dtype=float)
            for node in component:
                for neighbor, weight in enumerate(graph[node]):
                    if neighbor in component:
                        separated_graph[node][neighbor] = weight
                        separated_graph[neighbor][node] = weight
            separated_graphs.append(separated_graph)

        return separated_graphs


if __name__ == "__main__":
    mesh = TriangleMesh('cylinder_closed.stl')
