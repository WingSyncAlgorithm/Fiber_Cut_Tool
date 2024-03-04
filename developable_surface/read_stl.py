import numpy as np
from stl import Mesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import heapq
import time
from memory_profiler import profile


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
    @profile
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
        self.vertices[point_idx,:] = [temp[0].x, temp[0].y,temp[0].z]
        for i in range(1, np.size(temp)):
            if temp[i].x != temp[i-1].x or temp[i].y != temp[i-1].y or temp[i].z != temp[i-1].z:
                point_idx += 1
                #print(point_idx,temp[i].x, temp[i-1].x,temp[i].y, temp[i-1].y,temp[i].z, temp[i-1].z)
            self.triangles[temp[i].triangle_idx, temp[i].p_idx] = point_idx
            self.vertices[point_idx,:] = [temp[i].x, temp[i].y,temp[i].z]
            self.point_triangle_adj[point_idx].append(temp[i].triangle_idx)
            # self.point_triangle_adj[point_idx] = np.append(self.point_triangle_adj[point_idx],temp[i].triangle_idx)
        del temp
        #print(self.vertices)
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
        # 尋找高斯曲率過大的點
        for vertex_idx in range(self.num_vertices):
            if self.gaussian_curvature[vertex_idx] > 0.0001:
                self.high_curvature_points = np.append(
                    self.high_curvature_points, vertex_idx)
        # print("self.high_curvature_points",np.size(self.high_curvature_points))
        self.high_curvature_graph = np.full(
            [np.size(self.high_curvature_points), np.size(self.high_curvature_points)], -1, dtype=float)
        for i in range(np.size(self.high_curvature_points)):
            for j in range(i):
                point1_idx = self.high_curvature_points[i]
                point2_idx = self.high_curvature_points[j]
                if self.length[point1_idx, point2_idx] != -1:
                    self.high_curvature_graph[i, j] = (
                        self.gaussian_curvature[point1_idx] + self.gaussian_curvature[point2_idx]) / 2.0
                    self.high_curvature_graph[j,
                                              i] = self.high_curvature_graph[i, j]
        high_curvature_subgraph = self.separate_disconnected_components(
            self.high_curvature_graph)
        # print("high_curvature_subgraph",high_curvature_subgraph)
        
        self.plot_graph(high_curvature_subgraph[0])
        self.plot_graph(high_curvature_subgraph[1])
        
        # 儲存由兩點所連接的第三點
        self.connect = np.full(
            (self.num_vertices, self.num_vertices, 2), -1, dtype=int)
        self.find_connect()
        # 執行切割
        self.boundaries = []
        #print(np.size(high_curvature_subgraph, axis=0))
        for subgraph in range(np.size(high_curvature_subgraph, axis=0)):
            #print("find_max_cycle_cost")
            max_cycle_cost, max_cycle_path = self.find_max_cycle_cost(
                high_curvature_subgraph[subgraph][:][:])
            self.boundaries.append(max_cycle_path)
            #self.plot_points(max_cycle_path)
            add_cycle = []
            for point in range(np.size(max_cycle_path)):
                # print(max_cycle_path[point % np.size(
                #    max_cycle_path)], max_cycle_path[(point+1) % np.size(max_cycle_path)])
                if len(max_cycle_path)>2:
                    #print(point==0)
                    add_point = self.cut_edge(self.high_curvature_points[max_cycle_path[point % np.size(
                        max_cycle_path)]], self.high_curvature_points[max_cycle_path[(point+1) % np.size(max_cycle_path)]], self.high_curvature_points[max_cycle_path[(point+2) % np.size(max_cycle_path)]])
                    add_cycle.append(add_point)
            self.boundaries.append(add_cycle)
            # print(high_curvature_subgraph[subgraph][:][:])
        #print("self.num_vertices", self.num_vertices-self.num_original_vertices)
        self.length = np.full(
            (self.num_vertices, self.num_vertices), -1, dtype=float)
        self.calculate_length()
        surface_groups = self.separate_disconnected_components(self.length)
        self.boundary_idx = []
        for i in range(len(self.boundaries)):
            c=0
            for j in range(len(surface_groups)):
                for k in range(self.num_vertices):
                    if surface_groups[j][self.boundaries[i][0]][k] != -1:
                        self.boundary_idx.append(j)
                        c=1
                        break
                if c==1: break
        cut_path = []
        for i in range(len(set(self.boundary_idx))):
            if self.boundary_idx.count(i) >= 2:
                indices = [j for j, value in enumerate(self.boundary_idx) if value == i]
                for m in range(len(my_array)):
                    for n in range(m+1, len(my_array)):
                        boundary1_idx = indices[m]
                        boundary2_idx = indices[n]
                        boundary1 = self.boundaries[boundary1_idx]
                        boundary2 = self.boundaries[boundary2_idx]
                        cut_path.append(self.find_cut_path(surface_groups[i]))
        

                        

        self.angle = np.zeros(
            (self.num_vertices, self.num_vertices, self.num_vertices), dtype=float)
        self.calculate_angle()
        # 初始化儲存展開到平面的點的矩陣
        self.s = np.full((self.num_vertices, 2), 99999999, dtype=float)
        # 儲存由兩點所連接的第三點
        self.connect = np.full(
            (self.num_vertices, self.num_vertices, 2), -1, dtype=int)
        self.find_connect()
        self.start_edges = []
        surface_groups = self.separate_disconnected_components(self.length)
        for i in range(np.size(surface_groups, axis=0)):
            print(i)
            self.find_start_edges(surface_groups[i])

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

    def find_connect(self):
        for triangle_idx in range(np.size(self.triangles, axis=0)):
            if self.connect[self.triangles[triangle_idx, 0],
                         self.triangles[triangle_idx, 1], 0] != -1:
                print("yyyy")
            self.connect[self.triangles[triangle_idx, 0],
                         self.triangles[triangle_idx, 1], 0] = self.triangles[triangle_idx, 2]
            self.connect[self.triangles[triangle_idx, 0],
                         self.triangles[triangle_idx, 1], 1] = triangle_idx
            if self.connect[self.triangles[triangle_idx, 1],
                         self.triangles[triangle_idx, 2], 0] != -1:
                print("yyyy")
            self.connect[self.triangles[triangle_idx, 1],
                         self.triangles[triangle_idx, 2], 0] = self.triangles[triangle_idx, 0]
            self.connect[self.triangles[triangle_idx, 1],
                         self.triangles[triangle_idx, 2], 1] = triangle_idx
            if self.connect[self.triangles[triangle_idx, 2],
                         self.triangles[triangle_idx, 0], 0] != -1:
                print("yyyy")
            self.connect[self.triangles[triangle_idx, 2],
                         self.triangles[triangle_idx, 0], 0] = self.triangles[triangle_idx, 1]
            self.connect[self.triangles[triangle_idx, 2],
                         self.triangles[triangle_idx, 0], 1] = triangle_idx

    def cut_edge(self, vertex_idx1, vertex_idx2, vertex_idx3):
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
        triangle_idx = self.connect[vertex_idx1, vertex_idx2, 1]
        if self.triangles[triangle_idx, 0] == vertex_idx1:
            if self.triangles[triangle_idx, 1] == vertex_idx2:
                self.triangles[triangle_idx, 0] = vertex_add_idx1
                self.triangles[triangle_idx, 1] = vertex_add_idx2
                
                point1_idx = self.connect[vertex_idx1, vertex_idx2, 0]
                point2_idx = vertex_idx2
                point3_idx = self.connect[point1_idx, point2_idx, 0]
                if point1_idx != vertex_idx3:
                    while point3_idx != vertex_idx3:
                        triangle_middle_idx = self.connect[point1_idx,
                                                           point2_idx, 1]
                        #print(point3_idx,vertex_idx3)
                        for i in range(3):
                            if self.triangles[triangle_middle_idx, i] == vertex_idx2:
                                self.triangles[triangle_middle_idx,
                                               i] = vertex_add_idx2
                                break
                        point1_idx = self.connect[point1_idx, point2_idx, 0]
                        point3_idx = self.connect[point1_idx, point2_idx, 0]
                
        elif self.triangles[triangle_idx, 1] == vertex_idx1:
            if self.triangles[triangle_idx, 2] == vertex_idx2:
                self.triangles[triangle_idx, 1] = vertex_add_idx1
                self.triangles[triangle_idx, 2] = vertex_add_idx2
                
                point1_idx = self.connect[vertex_idx1, vertex_idx2, 0]
                point2_idx = vertex_idx2
                point3_idx = self.connect[point1_idx, point2_idx, 0]
                #print(self.vertices[vertex_idx1],self.vertices[point2_idx],self.vertices[point1_idx])
                #print(vertex_idx1,vertex_idx2,point1_idx,self.connect[point2_idx, point1_idx, 0], point3_idx,vertex_idx3)
                #print(self.connect[499,477,0])
                if point1_idx != vertex_idx3:
                    while point3_idx != vertex_idx3:
                        triangle_middle_idx = self.connect[point1_idx,
                                                           point2_idx, 1]
                        #print(point3_idx,vertex_idx3)
                        for i in range(3):
                            if self.triangles[triangle_middle_idx, i] == vertex_idx2:
                                self.triangles[triangle_middle_idx,
                                               i] = vertex_add_idx2
                                break
                        point1_idx = self.connect[point1_idx, point2_idx, 0]
                        point3_idx = self.connect[point1_idx, point2_idx, 0]
                
        elif self.triangles[triangle_idx, 2] == vertex_idx1:
            if self.triangles[triangle_idx, 0] == vertex_idx2:
                self.triangles[triangle_idx, 2] = vertex_add_idx1
                self.triangles[triangle_idx, 0] = vertex_add_idx2
                
                point1_idx = self.connect[vertex_idx1, vertex_idx2, 0]
                point2_idx = vertex_idx2
                point3_idx = self.connect[point1_idx, point2_idx, 0]
                if point1_idx != vertex_idx3:
                    while point3_idx != vertex_idx3:
                        triangle_middle_idx = self.connect[point1_idx,
                                                           point2_idx, 1]
                        #print(point3_idx,vertex_idx3)
                        for i in range(3):
                            if self.triangles[triangle_middle_idx, i] == vertex_idx2:
                                self.triangles[triangle_middle_idx,
                                               i] = vertex_add_idx2
                                break
                        point1_idx = self.connect[point1_idx, point2_idx, 0]
                        point3_idx = self.connect[point1_idx, point2_idx, 0]
                
        self.num_vertices = np.size(self.vertices, axis=0)
        return vertex_add_idx1
    
    def find_start_edges(self,surface_group):
        c=0
        for point1_idx in range(self.num_vertices):
            for point2_idx in range(self.num_vertices):
                if surface_group[point1_idx, point2_idx] != -1 and self.connect[point1_idx,point2_idx,0]!=-1:
                    self.start_edges.append([point1_idx,point2_idx])
                    c =1
                    break
            if c==1: break

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
        #print("find_max_cycle_cost_helper")

        for neighbor in range(len(graph)):
            if graph[current_node, neighbor] != -1:  # Check if there is an edge
                if not visited[neighbor]:
                    #print(neighbor)
                    max_cost, max_path = self.find_max_cycle_cost_helper(
                        graph, start_node, neighbor, visited, current_cost + graph[current_node, neighbor], max_cost, path, max_path)
                elif neighbor == start_node:  # Found a cycle
                    #print(neighbor)
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
            #print("find_max_cycle_cost_helper")
            cycle_cost, cycle_path = self.find_max_cycle_cost_helper(
                graph, start_node, start_node, visited, 0, max_cost, [], [])
            #print("find_max_cycle_cost", cycle_cost, cycle_path)
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
            separated_graph = np.full_like(graph, -1, dtype=float)
            for node in component:
                for neighbor, weight in enumerate(graph[node]):
                    if neighbor in component:
                        separated_graph[node][neighbor] = weight
                        separated_graph[neighbor][node] = weight
            separated_graphs.append(separated_graph)

        return separated_graphs

    def dijkstra(self, graph, start, end):
        num_nodes = len(graph)
        
        # 初始化距離列表，表示從起始節點到各節點的最短距離
        distances = [float('infinity')] * num_nodes
        distances[start] = 0

        # 初始化前驅節點列表
        predecessors = [None] * num_nodes

        # 初始化優先佇列
        priority_queue = [(0, start)]

        while priority_queue:
            current_distance, current_node = heapq.heappop(priority_queue)

            # 如果已經找到更短的路徑，則忽略當前節點
            if current_distance > distances[current_node]:
                continue

            # 遍歷所有相鄰節點
            for neighbor in range(num_nodes):
                weight = graph[current_node][neighbor]

                # 如果有邊且找到更短的路徑，則更新距離列表、前驅節點列表和優先佇列
                if weight > 0:
                    distance = current_distance + weight
                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        predecessors[neighbor] = current_node
                        heapq.heappush(priority_queue, (distance, neighbor))

        # 構建最短路徑節點序列
        path = []
        current_node = end
        while current_node is not None:
            path.insert(0, current_node)
            current_node = predecessors[current_node]

        return path, distances[end]  # 返回最短路徑的節點序列和路徑長度
    def find_cut_path(self, graph,start_nodes, end_nodes):
        shortest_distance = float('inf')
        shortest_path = []
        for start_node in start_nodes:
            for end_node in end_nodes:
                path, distance = self.dijkstra(graph, start_node, end_node)
                if distance < shortest_distance:
                    shortest_distance = distance
                    shortest_path = path
        return shortest_path


    def plot_graph(self,high_curvature_graph):
        x_data, y_data, z_data = [], [], []
        for i in range(np.size(high_curvature_graph, axis=0)):
            for j in range(np.size(high_curvature_graph, axis=1)):
                if high_curvature_graph[i][j] != -1:
                    point1_idx = self.high_curvature_points[i]
                    x_data.append(
                        self.vertices[point1_idx, 0])
                    y_data.append(
                        self.vertices[point1_idx, 1])
                    z_data.append(
                        self.vertices[point1_idx, 2])
                    #print(point1_idx)
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
    def plot_points_graph(self,high_curvature_graph):
        x_data, y_data, z_data = [], [], []
        for i in range(np.size(high_curvature_graph, axis=0)):
            for j in range(np.size(high_curvature_graph, axis=1)):
                if high_curvature_graph[i][j] != -1:
                    point1_idx = i
                    x_data.append(
                        self.vertices[point1_idx, 0])
                    y_data.append(
                        self.vertices[point1_idx, 1])
                    z_data.append(
                        self.vertices[point1_idx, 2])
                    #print(point1_idx)
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
    def plot_points(self,points):
        x_data, y_data, z_data = [], [], []
        for point_idx in points:
            x_data.append(
                self.vertices[self.high_curvature_points[point_idx], 0])
            y_data.append(
                self.vertices[self.high_curvature_points[point_idx], 1])
            z_data.append(
                self.vertices[self.high_curvature_points[point_idx], 2])
            #print(point_idx)
        #print("x_data",len(x_data))
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

if __name__ == "__main__":
    mesh = TriangleMesh('inclined_cylinder.stl')
