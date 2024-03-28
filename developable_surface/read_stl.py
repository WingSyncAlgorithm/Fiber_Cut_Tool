import numpy as np
from stl import Mesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import time
from memory_profiler import profile
import heapq
from itertools import combinations
#from tools import *

count_time = True
#wrapper for counting time
def timer(func):
    def wrapper(*args, **kwargs):
        if(count_time):
            start = time.time()
            result = func(*args, **kwargs)
            print("/--",func.__name__, ":", time.time()-start)
            return result
        else:
            return func(*args, **kwargs)
    return wrapper

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

class SpaceCoor:
    def __init__(self, r, theta, phi,index):
        self.r = r
        self.theta = theta
        self.phi = phi
        self.index = index
    def __eq__(self, other):
        return self.r == other.r and self.theta == other.theta and self.phi == other.phi
    def __gt__(self, other):
        if self.theta != other.theta:
            return self.theta > other.theta
        elif self.phi != other.phi:
            return self.phi < other.phi
        elif self.r != other.r:
            return self.r > other.r
        else:
            return self.index > other.index
        
    def __lt__(self, other):
        if self.theta != other.theta:
            return self.theta < other.theta
        elif self.phi != other.phi:
            return self.phi > other.phi
        elif self.r != other.r:
            return self.r < other.r
        else:
            return self.index < other.index

class AStarNode:
    def __init__(self, index, neighbor,endPoint,coors):
        self.D = 1
        self.idx = index
        self.inheap = False
        self.neighbor = neighbor
        self.score = None
        self.gscore = None # distance from startPoint to current
        self.hscore = self.D*self._distance_(coors[index],coors[endPoint]) # predict distance

    def _distance_(self,coor1,coor2):
        return math.sqrt(
                    (abs(coor1[0]-coor2[0]))**2+
                    (abs(coor1[1]-coor2[1]))**2+
                    (abs(coor1[2]-coor2[2]))**2
            )
    def __gt__(self,other):
        if(self.score == None or other.score == None):
            if(self.score==None and other.score==None):
                return False
            elif(other.score==None):
                return False
            else:
                return True
            
        if(self.score!= other.score):
            return self.score > other.score
        elif(self.hscore != other.hscore):
            return self.hscore > other.hscore
        else:
            return False
        
    def __eq__(self,other):
        if(self.score == None or other.score == None):
            return False
        return (
            self.score  == other.score and
            self.gscore == other.gscore and
            self.hscore == other.hscore
            )

    def __lt__(self,other):
        if(self.score == None or other.score == None):
            if(self.score==None and other.score==None):
                return False
            elif(other.score==None):
                return True
            else:
                return False
            
        if(self.score!= other.score):
            return self.score < other.score
        elif(self.hscore != other.hscore):
            return self.hscore < other.hscore
        else:
            return False #equal
        
class TriangleMesh:
    @profile
    @timer
    def __init__(self, stl_file,gaussian_curvature_limit=0.0001,nograph=False):
        self.gaussian_curvature_limit = gaussian_curvature_limit
        self.nograph = nograph

        self._current_node_ = -1 #for sorting
        self._current_node_info_ = {'current':self._current_node_} #for sorting, dp

        self.stl = stl_file
        stl = Mesh.from_file(self.stl)
        self.triangles = np.zeros((np.size(stl.vectors, axis=0), 3), dtype=int)
        ### self.triangles: 2d list --> key: triangle index, value: 3 point index
        self.num_triangles = np.size(stl.vectors, axis=0)
        self.vertices = list() # 2d list : index=vertex index, value[3]=coordinate
        ### self.vertices: 2d list --> key: point index, value: 3 coordinate
        
        print("p1")
        st = time.time()

        # 分配編號for點和三角形
        # 2d list --> key: point index, value: triangle index list
        self.point_triangle_adj = dict()
        self._convert_triangle_format_(stl)
        self.num_original_vertices = len(self.vertices) # int  : stl檔案中的點數(contain duplicated points)
        self.num_vertices = len(self.vertices)
        
        print(time.time()-st)
        print("p2")
        st = time.time()

        # 初始化self.edge_length
        self.edge_length = {i:dict() for i in range(self.num_vertices)} #儲存每個點到其他點的距離
        ### self.edge_length: 2d list --> index: point index, value: dict --> key: point index, value: distance(index,key)
        self.calculate_length()

        print(time.time()-st)
        print("p3")
        st = time.time()

        # 計算三角形面積
        self.area = np.zeros(np.size(self.triangles), dtype=float)
        self.calculate_area()

        print(time.time()-st)
        print("p4")
        st = time.time()

        # 計算角度
        self.angle = [dict() for i in range(self.num_vertices)]
        self.calculate_angle()

        print(time.time()-st)
        print("p5")
        st = time.time()

        # 計算高斯曲率
        self.gaussian_curvature = [0 for _ in range(self.num_vertices)]
        self.high_curvature_graph = dict() # 2d list --> key: point index, value: dict --> key: point index, value: curvature(index,key)
        self.high_curvature_points = []
        self.calculate_gaussian_curvature()
        # 尋找高斯曲率過大的點
        self.filter_gaussian_curvature()

        print(time.time()-st)
        print("p6")
        st = time.time()

        # 分離高曲率點
        high_curvature_subgraph = self.separate_disconnected_components(
            self.high_curvature_graph) # 2d list, each element is a subgraph(dict)        
        print(time.time()-st)

        self.plot_graph(high_curvature_subgraph[0])
        self.plot_graph(high_curvature_subgraph[1])

        print('p7')
        st = time.time()

        # 儲存由兩點所連接的第三點
        self.connect = {i:dict() for i in range(self.num_vertices)}
        self.find_connect()
        
        print(time.time()-st)
        
        print('p8')
        st = time.time()
        # 執行切割
        self.boundaries = []
        #print(np.size(high_curvature_subgraph, axis=0))
        self.process_cut(high_curvature_subgraph)
        #print("self.num_vertices", self.num_vertices-self.num_original_vertices)
        self.calculate_length()
        surface_groups = self.separate_disconnected_components(self.edge_length)
        for g in surface_groups: self.plot_graph(g)
        #self.plot(dict((key, value) for key, value in self.edge_length.items() if key > self.num_original_vertices),label=0)
        self.boundary_idx = []
        
        print(time.time()-st)
        print('p9')
        st = time.time()

## here~
        # determine which boundary edge belongs to which surface group
        ### one boundary edge can only belong to one surface group
        ### one surface group can have multiple boundary edges
        self.boundary_idx = [-1 for i in range(len(self.boundaries))]
        for b_idx,b in enumerate(self.boundaries):
            boundary_points = np.array(b)
            for sg_idx, sg in enumerate(surface_groups):
                    surface_group_points = np.array(list(sg.keys()))
                    if np.isin(boundary_points, surface_group_points).all():
                        self.boundary_idx[b_idx] = sg_idx
                        break

        cut_path = []

        print(time.time()-st)

        print('p10')
        st = time.time()
        ## find short cut path between two boundaries which belong to the same surface group
        for i in range(len(set(self.boundary_idx))):
            if self.boundary_idx.count(i) >= 2:
                indices = [j for j, value in enumerate(self.boundary_idx) if value == i]
                for m in range(len(indices)):
                    for n in range(m+1, len(indices)):
                        boundary1_idx = indices[m]
                        boundary2_idx = indices[n]
                        boundary1 = self.boundaries[boundary1_idx]
                        boundary2 = self.boundaries[boundary2_idx]
                        cut_path.append(self.find_cut_path4(surface_groups[i],boundary1,boundary2))
        
        print(time.time()-st)
        print('p11')
        st = time.time()
                        

        self.angle = [dict() for i in range(self.num_vertices)]
        self.calculate_angle()
        # 初始化儲存展開到平面的點的矩陣
        self.s = np.full((self.num_vertices, 2), 99999999, dtype=float)
        # 儲存由兩點所連接的第三點
        #self.connect[point1_idx,point2_idx] = [point3_idx,triangle_idx]   逆時針
        self.connect = {i:dict() for i in range(self.num_vertices)}
        self.find_connect()
        self.start_edges = []
        print(time.time()-st)
        print('p12')
        st = time.time()
        surface_groups = self.separate_disconnected_components(self.edge_length)
        for i in range(np.size(surface_groups, axis=0)):
            print(i)
            self.find_start_edges(surface_groups[i])
        print(time.time()-st)

    @timer
    def _convert_triangle_format_(self, stl):
        '''
        time complexity: O(nlogn)
        space complexity: O(n)
        分配編號for點和三角形
        self.triangles: 2d list --> key: triangle index, value: 3 point index
        self.vertices: 2d list --> key: point index, value: 3 coordinate
        '''
        temp = [
                Point(stl.vectors[triangle_idx, vertex, 0], 
                      stl.vectors[triangle_idx, vertex, 1],
                      stl.vectors[triangle_idx, vertex, 2], 
                      triangle_idx, 
                      vertex)
                for triangle_idx in range(np.size(stl.vectors, axis=0)) 
                    for vertex in range(3)
                ]

        temp.sort()

        # point_idx starts from 0
        point_idx = 0
        self.triangles[temp[0].triangle_idx, temp[0].p_idx] = point_idx
        self.vertices.append([temp[0].x, temp[0].y,temp[0].z])

        for i in range(1, len(temp)):
            if (temp[i].x != temp[i-1].x) or (temp[i].y != temp[i-1].y) or (temp[i].z != temp[i-1].z) :
                point_idx += 1
                self.vertices.append([temp[i].x, temp[i].y,temp[i].z])
            self.triangles[temp[i].triangle_idx, temp[i].p_idx] = point_idx
            if(point_idx not in self.point_triangle_adj):
                self.point_triangle_adj[point_idx] = [temp[i].triangle_idx]
            else:
                self.point_triangle_adj[point_idx].append(temp[i].triangle_idx)
        del temp

    def p2p_length(self,point1_idx,point2_idx):
        point1 = self.vertices[point1_idx]
        point2 = self.vertices[point2_idx]
        return math.sqrt(
                (point1[0]-point2[0])**2+
                (point1[1]-point2[1])**2+
                (point1[2]-point2[2])**2
            )

    @timer
    def calculate_length(self):
        self.edge_length = {i:dict() for i in range(self.num_vertices)}
        for triangle_idx in range(np.size(self.triangles, axis=0)):
            point1_idx = self.triangles[triangle_idx, 0]
            point2_idx = self.triangles[triangle_idx, 1]
            point3_idx = self.triangles[triangle_idx, 2]
            self.edge_length[point1_idx][point2_idx] = self.p2p_length(point1_idx,point2_idx)
            self.edge_length[point2_idx][point1_idx] = self.edge_length[point1_idx][point2_idx]
            self.edge_length[point1_idx][point3_idx] = self.p2p_length(point1_idx,point3_idx)
            self.edge_length[point3_idx][point1_idx] = self.edge_length[point1_idx][point3_idx]
            self.edge_length[point2_idx][point3_idx] = self.p2p_length(point2_idx,point3_idx)
            self.edge_length[point3_idx][point2_idx] = self.edge_length[point2_idx][point3_idx]
        return

    @timer
    def calculate_area(self):
        for triangle_idx in range(np.size(self.triangles, axis=0)):
            v1 = np.array(self.vertices[self.triangles[triangle_idx, 0]],dtype=float)
            v2 = np.array(self.vertices[self.triangles[triangle_idx, 1]],dtype=float)
            v3 = np.array(self.vertices[self.triangles[triangle_idx, 2]],dtype=float)
            cross_product = np.cross(v2 - v1, v3 - v1)
            self.area[triangle_idx] = 0.5 * np.linalg.norm(cross_product)
        return

    @timer
    def calculate_angle(self):
        for triangle_idx in range(np.size(self.triangles, axis=0)):
            for point in range(3):
                point1_idx = self.triangles[triangle_idx, (point+1) % 3]
                point2_idx = self.triangles[triangle_idx, point]
                point3_idx = self.triangles[triangle_idx, (point+2) % 3]
                point1 = np.array(self.vertices[point1_idx],dtype=float)
                point2 = np.array(self.vertices[point2_idx],dtype=float)
                point3 = np.array(self.vertices[point3_idx],dtype=float)
                v1 = point1 - point2
                v2 = point3 - point2
                dot_product = np.dot(v1, v2)
                magnitude_v1 = np.linalg.norm(v1)
                magnitude_v2 = np.linalg.norm(v2)
                cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
                self.angle[point2_idx][point1_idx,
                           point3_idx] = np.arccos(cos_theta)
                self.angle[point2_idx][point3_idx,
                           point1_idx] = np.arccos(cos_theta)
        return

    # @profile
    @timer
    def calculate_gaussian_curvature(self):

        for vertex_idx in range(self.num_vertices):
            a_vertex = 0
            for triangle_idx in self.point_triangle_adj[vertex_idx]:
                a_vertex += self.area[triangle_idx]

            angles = self.angle[vertex_idx]
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
        return
 
    @timer
    def filter_gaussian_curvature(self):
        # 尋找高斯曲率過大的點
        self.high_curvature_points = [idx for idx,v in enumerate(self.gaussian_curvature) if v>self.gaussian_curvature_limit]
        self.high_curvature_graph = {p:dict() for p in self.high_curvature_points}

        for p1,p2 in combinations(self.high_curvature_points,2):
            if(p2 in self.edge_length[p1]):
                self.high_curvature_graph[p1][p2] = (self.gaussian_curvature[p1]+self.gaussian_curvature[p2])/2.0
                self.high_curvature_graph[p2][p1] = self.high_curvature_graph[p1][p2]        
        for k in self.high_curvature_graph:
            if(len(self.high_curvature_graph[k])==0):
                print("delete",k)
                del self.high_curvature_graph[k] 
        return

    def separate_disconnected_components(self, graph):
        components = self.connected_components(graph)
        #2d list, each list is a component(point_idx)

        # Create separate graphs for each connected component
        separated_graphs = []
        for component in components:
            separated_graph = {idx:dict() for idx in component}
            #key: idx1, value: dict()-->key:idx2, value:weight
            for p1_idx,p2_idx in combinations(component,2):
                if p2_idx in graph[p1_idx]:
                    separated_graph[p1_idx][p2_idx] = graph[p1_idx][p2_idx]
                    separated_graph[p2_idx][p1_idx] = graph[p1_idx][p2_idx]
            separated_graphs.append(separated_graph)
        return separated_graphs

    def connected_components(self, graph):
        visited = {i: False for i in graph.keys()}
        components = []

        for node in graph.keys():
            if visited[node] == False:
                component = []
                self.dfs(graph, node, visited, component)
                components.append(component)

        return components
    
    def dfs(self, graph, start, visited, component):
        visited[start] = True
        component.append(start)

        for neighbor, weight in graph[start].items():
            if visited[neighbor] == False:
                self.dfs(graph, neighbor, visited, component)
        return

    @timer
    def find_connect(self):
        for triangle_idx in range(np.size(self.triangles, axis=0)):
            self.connect[self.triangles[triangle_idx, 0]][self.triangles[triangle_idx, 1]] = (self.triangles[triangle_idx, 2],triangle_idx)
            self.connect[self.triangles[triangle_idx, 1]][self.triangles[triangle_idx, 2]] = (self.triangles[triangle_idx, 0],triangle_idx)
            self.connect[self.triangles[triangle_idx, 2]][self.triangles[triangle_idx, 0]] = (self.triangles[triangle_idx, 1],triangle_idx)
        return

    @timer
    def process_cut(self, high_curvature_subgraph):
        for subgraph in high_curvature_subgraph:
            #print("find_max_cycle_cost")
            max_cycle_path = self.find_max_cycle_cost(
                subgraph)
            self.boundaries.append(max_cycle_path)
            #self.plot_points(max_cycle_path)
            add_cycle = []
            for point in range(len(max_cycle_path)):
                add_point = self.cut_edge(max_cycle_path[(point)%len(max_cycle_path)], max_cycle_path[(point+1)%len(max_cycle_path)], max_cycle_path[(point+2)%len(max_cycle_path)])
                add_cycle.append(add_point)
            self.boundaries.append(add_cycle)
            # print(high_curvature_subgraph[subgraph][:][:])
        return

    def find_max_cycle_cost(self, graph):
        num_nodes = max(graph.keys())+1  #base 0
        max_cost = float('-inf')
        max_path = []
        for start_node in graph: #
            visited = [False for _ in range(num_nodes)]
            #visited = np.zeros(num_nodes, dtype=bool)
            #print("find_max_cycle_cost_helper")
            cycle_cost, cycle_path = self.find_max_cycle_cost_helper(
                graph, start_node, start_node, visited, 0, max_cost, [], [])
            #print("find_max_cycle_cost", cycle_cost, cycle_path)
            if cycle_cost > max_cost:
                max_cost = cycle_cost
                max_path = cycle_path

        return  max_path
    
## recursion --> must change
    def find_max_cycle_cost_helper(self, graph, start_node, current_node, visited, current_cost, max_cost, path, max_path):
        visited[current_node] = True
        path.append(current_node)
        #print("find_max_cycle_cost_helper")

        for neighbor in graph[current_node]:
            if not visited[neighbor]:
                #print(neighbor)
                max_cost, max_path = self.find_max_cycle_cost_helper(
                    graph, start_node, neighbor, visited, current_cost + graph[current_node][neighbor], max_cost, path, max_path)
            elif neighbor == start_node:  # Found a cycle
                #print(neighbor)
                if current_cost + graph[current_node][neighbor] > max_cost:
                    max_cost = current_cost + graph[current_node][neighbor]
                    max_path = path.copy()

        visited[current_node] = False
        path.pop()

        return max_cost, max_path
    
    def cut_edge(self, vertex_idx1, vertex_idx2, vertex_idx3):
        '''
        vertex_idx1往vertex_idx2切割
        '''
        vertex_add_idx2 = 0
        is_added2 = False
        for added_idx in range(self.num_original_vertices, self.num_vertices):
            if (self.vertices[vertex_idx2] == self.vertices[added_idx]):
                vertex_add_idx2 = added_idx
                is_added2 = True

        if is_added2 == False:
            vertex_add_idx2 = self.num_vertices
            self.vertices.append(self.vertices[vertex_idx2])
            self.num_vertices = len(self.vertices)
                
        point1_idx = vertex_idx1
        point2_idx = vertex_idx2
        point3_idx = self.connect[point1_idx][point2_idx][0]
        while True:
            triangle_middle_idx = self.connect[point1_idx][point2_idx][1]
            for i in range(3):
                if self.triangles[triangle_middle_idx, i] == vertex_idx2:
                    self.triangles[triangle_middle_idx,
                                   i] = vertex_add_idx2
                    break
            if point3_idx == vertex_idx3:
                break
            point1_idx = point3_idx
            point3_idx = self.connect[point1_idx][point2_idx][0]
        return vertex_add_idx2

    def find_cut_path3(self, graph1,start_nodes, end_nodes):
        """
        A* algorithm
        shortest path, but time complexity is high
        """
        """
        for idx, node in enumerate(graph1):
            tmp = dict()
            for nidx, dis in enumerate(node):
                if(dis!=-1):
                    tmp[nidx]=dis
            if(len(tmp)!=0):
                graph[idx] = tmp
        """
        shortest_distance = float('inf')
        shortest_path = []
        for start_node in start_nodes:
            for end_node in end_nodes:
                path, distance = self.AStar(graph1, start_node, end_node)
                if distance < shortest_distance:
                    shortest_distance = distance
                    shortest_path = path
        return shortest_path

    def find_cut_path4(self, graph1,start_nodes, end_nodes):
        """
        A* algorithm
        not shortest path but somewhat a short path, with acceptable time complexity
        """
        shortest_path = []
        start_node = start_nodes[0]
        end_node = end_nodes[0]
        min_distance = float('inf')
        for s_node in start_nodes:
            for e_node in end_nodes:
                distance = self.p2p_length(s_node,e_node)
                if distance < min_distance:
                    min_distance = distance
                    start_node = s_node
                    end_node = e_node
        shortest_path, distance = self.AStar(graph1, start_node, end_node)
        return shortest_path

    def AStar(self,graph1, start_node, end_node):
        graph=dict()
        for nidx, n in graph1.items():
            graph[nidx] = AStarNode(nidx,n,end_node,self.vertices)
        heap = [graph[start_node]]
        graph[start_node].gscore = 0
        graph[start_node].score = graph[start_node].gscore + graph[start_node].hscore
        graph[start_node].inheap = True
        heapq.heapify(heap)
        current_node_idx = start_node
        while(len(heap)!=0 and current_node_idx!=end_node):
            flag = False
            current_node = heapq.heappop(heap)
            current_node.inheap = False
            current_node_idx = current_node.idx
            for nidx, dis in current_node.neighbor.items():
                if(graph[nidx].score == None or graph[nidx].gscore > current_node.gscore+dis):
                    graph[nidx].gscore = current_node.gscore+dis
                    graph[nidx].score = graph[nidx].gscore + graph[nidx].hscore
                    if(graph[nidx].inheap==False):
                        graph[nidx].inheap = True
                        heapq.heappush(heap,graph[nidx])
                    else:
                        flag = True
            if(flag==True):
                heapq.heapify(heap)
        assert(current_node_idx == end_node) #unknown error
        
        # a path is find
        path = [end_node]
        while(current_node_idx != start_node):
            neighbor = list(graph[current_node_idx].neighbor.keys())
            min_distance = math.inf
            min_idx = -1
            #find idx which is current_node's neighbor and have min gscore
            for idx in neighbor:
                if(graph[idx].gscore != None and graph[idx].gscore < min_distance):
                    min_distance = graph[idx].gscore
                    min_idx = idx

            assert(min_idx!=-1)
            path.append(min_idx)
            current_node_idx = min_idx
        
        path.reverse()
        return path, graph[end_node].gscore

    def find_start_edges(self,surface_group):
        """
        find any edge in the surface group
        """
        self.start_edges.append([list(surface_group.keys())[0],list(surface_group[list(surface_group.keys())[0]].keys())[0]]) 
        return
        '''
        c=0
        for point1_idx in range(self.num_vertices):
            for point2_idx in range(self.num_vertices):
                if(point1_idx==point2_idx or point1_idx not in surface_group.keys() or point2_idx not in surface_group[point1_idx].keys()):
                    continue
## here
                if self.connect[point1_idx,point2_idx,0]!=-1:
                    self.start_edges.append([point1_idx,point2_idx])
                    c =1
                    break
            if c==1: break
        return
        '''

    def plot_graph(self,high_curvature_graph):
        if(self.nograph):
            return
        x_data, y_data, z_data = [], [], []
        for idx in high_curvature_graph:
            x_data.append(
                self.vertices[idx][0])
            y_data.append(
                self.vertices[idx][1])
            z_data.append(
                self.vertices[idx][2])
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
        return

    def plot_points_graph(self,high_curvature_graph):
        x_data, y_data, z_data = [], [], []
        for i in high_curvature_graph:
            point1_idx = i
            x_data.append(
                self.vertices[point1_idx][0])
            y_data.append(
                self.vertices[point1_idx][1])
            z_data.append(
                self.vertices[point1_idx][2])
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
        return
    
    def plot_points(self,points):
        x_data, y_data, z_data = [], [], []
        for point_idx in points:
            x_data.append(
                self.vertices[point_idx][0])
            y_data.append(
                self.vertices[point_idx][1])
            z_data.append(
                self.vertices[point_idx][2])
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
        return
    
    def plot(self,graph,label=1,edge=1,enable=1,highlight=[]):
        if(enable==0):
            return
        if(highlight!=[]):
            highlight.append(highlight[0])
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        

        for p1, p2s in graph.items():
            # draw point
            ax.scatter(self.vertices[p1][0], self.vertices[p1][1], self.vertices[p1][2], color='red', marker='o',s = 1)
            # draw label
            if(label==1):
                ax.text(self.vertices[p1][0], self.vertices[p1][1], self.vertices[p1][2], str(p1), color='black', fontsize=12)
            for p2, draw_edge in p2s.items():
                #ax.scatter(self.vertices[p2][0], self.vertices[p2][1], self.vertices[p2][2], color='red', marker='o',s = 1)
                #ax.plot([self.vertices[p1][0], self.vertices[p2][0]], [self.vertices[p1][1], self.vertices[p2][1]],[self.vertices[p1][2], self.vertices[p2][2]], color='green', linestyle='-', linewidth=2)
                if edge==1 and draw_edge == True:
                    #draw edge from p1 to p2
                    if(abs(highlight.index(p1)-highlight.index(p2))==1):
                        ax.plot([self.vertices[p1][0], self.vertices[p2][0]], [self.vertices[p1][1], self.vertices[p2][1]],[self.vertices[p1][2], self.vertices[p2][2]], color='green', linestyle='-', linewidth=2)
                    else:
                        ax.plot([self.vertices[p1][0], self.vertices[p2][0]], [self.vertices[p1][1], self.vertices[p2][1]],[self.vertices[p1][2], self.vertices[p2][2]], color='blue', linestyle='-', linewidth=2)
        plt.axis('equal')
        # 添加標籤和標題
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')

        # 顯示圖形
        plt.show()

    def dijkstra(self, graph, start, end):
        num_nodes = len(graph)
        
        # 初始化距離列表，表示從起始節點到各節點的最短距離
        distances = {nidx:float('infinity') for nidx in graph.keys()}
        distances[start] = 0

        # 初始化前驅節點列表
        predecessors = {nidx:None for nidx in graph.keys()}

        # 初始化優先佇列
        priority_queue = [(0, start)]

        while len(priority_queue)>0:
            current_distance, current_node = heapq.heappop(priority_queue)

            # 如果已經找到更短的路徑，則忽略當前節點
            if current_distance > distances[current_node]:
                continue

            # 遍歷所有相鄰節點
            for neighbor in graph[current_node].keys():

                weight = graph[current_node][neighbor]

                # 如果有邊且找到更短的路徑，則更新距離列表、前驅節點列表和優先佇列
                distance = current_distance + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    predecessors[neighbor] = current_node
                    heapq.heappush(priority_queue, (distance, neighbor))

        # 構建最短路徑節點序列
        path = []
        current_node = end
        while current_node != None:
            path.append(current_node)
            current_node = predecessors[current_node]
        path.reverse()
        return path, distances[end]  # 返回最短路徑的節點序列和路徑長度

    def find_cut_path1(self, graph,start_nodes, end_nodes):
        """
        dijkstra
        shortest path but time complexity is high
        """
        shortest_distance = float('inf')
        shortest_path = []
        for start_node in start_nodes:
            for end_node in end_nodes:
                path, distance = self.dijkstra(graph, start_node, end_node)
                if distance < shortest_distance:
                    shortest_distance = distance
                    shortest_path = path
        return shortest_path

    def find_cut_path2(self, graph,start_nodes, end_nodes):
        """
        dijkstra
        not shorstest path but is somewhat short with acceptable time complexity
        """
        shortest_path = []
        start_node = start_nodes[0]
        end_node = end_nodes[0]
        min_distance = float('inf')
        for s_node in start_nodes:
            for e_node in end_nodes:
                distance = self.p2p_length(s_node,e_node)
                if distance < min_distance:
                    min_distance = distance
                    start_node = s_node
                    end_node = e_node

        shortest_path, distance = self.dijkstra(graph, start_node, end_node)
        return shortest_path

    '''   
        def cartesian_to_spherical(self,start_idx,end_idx):
        assert(start_idx == self._current_node_info_['current'])
        if (end_idx in self._current_node_info_):
            return self.vertices[end_idx]

        x1 = self.vertices[start_idx][0]
        y1 = self.vertices[start_idx][1]
        z1 = self.vertices[start_idx][2]
        x2 = self.vertices[end_idx][0]
        y2 = self.vertices[end_idx][1]
        z2 = self.vertices[end_idx][2]
        x = x2-x1
        y = y2-y1
        z = z2-z1
        r = math.sqrt(x**2 + y**2 + z**2)
        xy = math.sqrt(x**2 + y**2)
        theta = math.degrees(math.atan2(z,xy))
        #theta = math.acos(z / r)
        phi = math.degrees(math.atan2(y, x))
        if(phi<0):
            phi+=360
        self._current_node_info_[end_idx] = SpaceCoor(r,z,phi,end_idx)
        return self._current_node_info_[end_idx]

    def find_max_cycle_cost2(self, graph, circles, enabled_edges,start_node=None):
        """
        tmp = dict()
        for i in enabled_edges.keys():
            if(i<15):
                tmp[i] = dict()
                for j in enabled_edges[i]:
                    if(j<15):
                        tmp[i][j] = enabled_edges[i][j]
        """
        high = [474, 485, 499, 505, 510, 506, 500, 486, 476, 459, 437, 417, 392, 370, 339, 306, 275, 241, 213, 182, 146, 111, 75, 52, 37, 16, 11, 7, 5, 3, 1, 0, 2, 4, 6, 9, 14, 35, 50, 71, 107, 142, 178, 209, 237, 271, 302, 335, 366, 390, 415, 435, 457]

        self.plot(enabled_edges,label=0,edge=1,enable=1,highlight=high)
        #circles_in_func = []
        circles = []
        cnt = len([1 for i in enabled_edges.values() for j in i.values() if j]) #remaining edges(*2)
        while cnt > 0:
            #find first edge is enabled
            flag=0
            if (start_node == None or list(enabled_edges[start_node].values()) == [False for _ in range(len(enabled_edges[start_node]))]):
                for i in enabled_edges:
                    if(sum(enabled_edges[i].values())>0):
                        start_node = i
                        break
            assert(start_node != -1)

            #right hand rule
            ### sort by (phi_1!=phi_2 ? phi_1<phi_2 : (theta_1!=theta_2 ? theta_1 < theta_2 : (r_1 != r_2 ? r_1 < r_2 : idx1 < idx2)))
            current_node = start_node
            self._current_node_ = current_node
            self._current_node_info_['current'] = current_node
            edge_stack = []
            point_stack = [start_node] #possible circle
            circle=[]
            while(1):
                neighbor = [n for n in enabled_edges[current_node] if enabled_edges[current_node][n]]
                neighbor_info = [self.cartesian_to_spherical(current_node,n) for n in neighbor]
                neighbor_info.sort(reverse=1)
                neighbor = [n.index for n in neighbor_info]

                if(len(neighbor)==0):
                    break
                
                if(len(edge_stack)>0 and (neighbor[0],current_node) == edge_stack[-1]):
                    #dirty branch edge
                    enabled_edges[current_node][neighbor[0]] = False
                    enabled_edges[neighbor[0]][current_node] = False
                    edge_stack.pop()
                    point_stack.pop()
                    current_node = point_stack[-1]
                    self._current_node_ = current_node
                    self._current_node_info_ = dict()
                    self._current_node_info_['current'] = current_node
                    continue
                
                if(neighbor[0] == point_stack[0]):
                    #find a circle
                    circle = point_stack[:]
                    #disable edges
                    for i in range(len(circle)-1):
                        enabled_edges[circle[i]][circle[(i+1)%len(circle)]] = False
                        enabled_edges[circle[(i+1)%len(circle)]][circle[i]] = False
                    enabled_edges[circle[-1]][circle[0]] = False
                    enabled_edges[circle[0]][circle[-1]] = False
                    break
                
                if(neighbor[0] in point_stack):
                    #possible circle
                    current_node = neighbor[0]
                    self._current_node_ = current_node
                    self._current_node_info_ = dict()
                    self._current_node_info_['current'] = current_node
                    point_stack = [current_node]
                    edge_stack = []
                    continue
                    

                #normal edge
                edge_stack.append((current_node,neighbor[0]))
                point_stack.append(neighbor[0])
                current_node = neighbor[0]
                self._current_node_ = current_node
                self._current_node_info_ = dict()
                self._current_node_info_['current'] = current_node

            cnt = len([1 for i in enabled_edges.values() for j in i.values() if j]) #remaining edges(*2)
        assert(cnt==0)
        return

    def find_max_cycle_cost_wrapper(self, graph):
        #self.plot_points([0,1,2,3,4,5])
        #self.plot(graph)
        circles=[]
        enabled_edges = dict()
        for i in graph:
            tmp = dict()
            for j in graph[i]:
                tmp[j] = True
            enabled_edges[i] = tmp
        self.find_max_cycle_cost2(graph, circles, enabled_edges)
        #assert(len(circles)==1)
        for c in circles:
            self.plot_points(c)
        return
    
## test p8 at here
    def find_max_cycle_cost(self, graph): #help testing
        #return self.find_max_cycle_cost1(graph)
        return self.find_max_cycle_cost_wrapper(graph)

    '''

if __name__ == "__main__":
    mesh = TriangleMesh('inclined_cylinder.stl',gaussian_curvature_limit=0.0001,nograph=True)
    #mesh = TriangleMesh('cylinder.stl')