from getopt import *
from time import *
from numpy import *
import sys
from random import *
import multiprocessing

# Set some initial widely used argument
start = 0
ter_time = 600
ran_seed = 0
vertices = 0
depot = -1
capacity = -1
graph_edge = []
threads = []


# The main solver method
def solver(file_name, return_dict):
    pc_list = {}
    set_globals(file_name)
    path_cost = cal_dijk()
    cost, path, remain, car_cap, car_position, car_index = reset_all()
    result_path, result_cost = priorkey_near_first(remain, path, car_cap, car_position, car_index, cost, path_cost)
    pc_list[result_cost] = result_path
    cost, path, remain, car_cap, car_position, car_index = reset_all()
    result_path, result_cost = priorkey_far_first(remain, path, car_cap, car_position, car_index, cost, path_cost)
    pc_list[result_cost] = result_path
    cost, path, remain, car_cap, car_position, car_index = reset_all()
    result_path, result_cost = priorkey_half_half(remain, path, car_cap, car_position, car_index, cost, path_cost)
    pc_list[result_cost] = result_path
    cost, path, remain, car_cap, car_position, car_index = reset_all()
    result_path, result_cost = priorkey_half_half2(remain, path, car_cap, car_position, car_index, cost, path_cost)
    pc_list[result_cost] = result_path
    k = sorted(pc_list.items())[0][0]
    for i in range(0, 3000):
        if ter_time < time() - start + 3 * (ter_time / 60):
            break
        cost, path, remain, car_cap, car_position, car_index = reset_all()
        result_path, result_cost = priorkey_random(remain, path, car_cap, car_position, car_index, cost, path_cost)
        pc_list[result_cost] = result_path
        cost, path, remain, car_cap, car_position, car_index = reset_all()
        result_path, result_cost = priorkey_random2(remain, path, car_cap, car_position, car_index, cost, path_cost)
        pc_list[result_cost] = result_path
        k = sorted(pc_list.items())[0][0]
    return_dict[k] = pc_list[k]


def reset_all():
    path = []
    remain = graph_edge.copy()
    for i in range(0, len(graph_edge))[::-1]:
        if graph_edge[i][3] == 0:
            remain = delete(remain, i, 0)
    remain = remain[argsort(remain[:, 2])]
    path.append(0)
    car_cap = capacity
    car_position = depot
    car_index = 0
    cost = 0
    return cost, path, remain, car_cap, car_position, car_index


# Sort method to sort the result dictionary
def sort_dic(adict):
    items = adict.items()
    items.sort()
    return items[0], [value for key, value in items]


# Method to calculate the near first path
def priorkey_near_first(remain, path, car_cap, car_position, car_index, cost, path_cost):
    while len(remain) != 0:
        at_list, at_list_dic = add_des_list(remain, car_position, car_cap)
        if len(at_list) != 0:
            car_cap, car_position, cost, path, remain = \
                choose_path(car_position, cost, car_cap, at_list, path, at_list_dic, 0, remain)
        if len(at_list) == 0:
            if len(at_list) == 0:
                cost, car_position, path, car_cap, car_index = \
                    go_situation(path_cost, car_position, remain, car_cap, cost, path, car_index)
    path.append(0)
    cost += path_cost[int(car_position) - 1][depot - 1]
    return path, cost


# Method to calculate the far first path
def priorkey_far_first(remain, path, car_cap, car_position, car_index, cost, path_cost):
    while len(remain) != 0:
        at_list, at_list_dic = add_des_list(remain, car_position, car_cap)
        if len(at_list) != 0:
            car_cap, car_position, cost, path, remain = \
                choose_path(car_position, cost, car_cap, at_list, path, at_list_dic, len(at_list) - 1, remain)
        if len(at_list) == 0:
            cost, car_position, path, car_cap, car_index = \
                go_situation(path_cost, car_position, remain, car_cap, cost, path, car_index)
    path.append(0)
    cost += path_cost[int(car_position) - 1][depot - 1]
    return path, cost


# Method to calculate the half near half far path
def priorkey_half_half(remain, path, car_cap, car_position, car_index, cost, path_cost):
    while len(remain) != 0:
        at_list, at_list_dic = add_des_list(remain, car_position, car_cap)
        if len(at_list) != 0:
            if car_cap > capacity / 2:
                car_cap, car_position, cost, path, remain = \
                    choose_path(car_position, cost, car_cap, at_list, path, at_list_dic, len(at_list) - 1, remain)
            else:
                car_cap, car_position, cost, path, remain = \
                    choose_path(car_position, cost, car_cap, at_list, path, at_list_dic, 0, remain)
        if len(at_list) == 0:
            cost, car_position, path, car_cap, car_index = \
                go_situation(path_cost, car_position, remain, car_cap, cost, path, car_index)
    path.append(0)
    cost += path_cost[int(car_position) - 1][depot - 1]
    return path, cost


# Method to calculate the half near half far path in the other testing way
def priorkey_half_half2(remain, path, car_cap, car_position, car_index, cost, path_cost):
    while len(remain) != 0:
        at_list, at_list_dic = add_des_list(remain, car_position, car_cap)
        if len(at_list) != 0:
            if car_cap < capacity / 2:
                car_cap, car_position, cost, path, remain = \
                    choose_path(car_position, cost, car_cap, at_list, path, at_list_dic, len(at_list) - 1, remain)
            else:
                car_cap, car_position, cost, path, remain = \
                    choose_path(car_position, cost, car_cap, at_list, path, at_list_dic, 0, remain)
        if len(at_list) == 0:
            cost, car_position, path, car_cap, car_index = \
                go_situation(path_cost, car_position, remain, car_cap, cost, path, car_index)
    path.append(0)
    cost += path_cost[int(car_position) - 1][depot - 1]
    return path, cost


# Method to calculate the random created first path
def priorkey_random(remain, path, car_cap, car_position, car_index, cost, path_cost):
    while len(remain) != 0:
        at_list, at_list_dic = add_des_list(remain, car_position, car_cap)
        if len(at_list) != 0:
            car_cap, car_position, cost, path, remain = \
                choose_path(car_position, cost, car_cap, at_list, path, at_list_dic, randint(0, len(at_list) - 1),
                            remain)
        if len(at_list) == 0:
            cost, car_position, path, car_cap, car_index = \
                go_situation(path_cost, car_position, remain, car_cap, cost, path, car_index)
    path.append(0)
    cost += path_cost[int(car_position) - 1][depot - 1]
    return path, cost


# Method to calculate the random created first path devided in two path
def priorkey_random2(remain, path, car_cap, car_position, car_index, cost, path_cost):
    while len(remain) != 0:
        at_list, at_list_dic = add_des_list(remain, car_position, car_cap)
        if len(at_list) != 0:
            if car_cap < capacity / 2:
                car_cap, car_position, cost, path, remain = \
                    choose_path(car_position, cost, car_cap, at_list,
                                path, at_list_dic, randint(0, len(at_list) - 1), remain)
            else:
                car_cap, car_position, cost, path, remain = \
                    choose_path(car_position, cost, car_cap, at_list, path,
                                at_list_dic, len(at_list) - 1 - randint(0, len(at_list) - 1), remain)
        if len(at_list) == 0:
            cost, car_position, path, car_cap, car_index = \
                go_situation(path_cost, car_position, remain, car_cap, cost, path, car_index)
    path.append(0)
    cost += path_cost[int(car_position) - 1][depot - 1]
    return path, cost


# Find the destination list
def add_des_list(remain, car_position, car_cap):
    at_list = []
    at_list_dic = {}
    for i in range(0, len(remain)):
        if (remain[i][0] == car_position or remain[i][1] == car_position) and remain[i][3] <= car_cap:
            at_list.append(remain[i])
            at_list_dic[i] = remain[i]
    return at_list, at_list_dic


# Select the right path to go
def choose_path(car_position, cost, car_cap, at_list, path, at_list_dic, way, remain):
    if at_list[way][0] == car_position:
        path.append((int(car_position), int(at_list[way][1])))
        car_position = at_list[way][1]
    elif at_list[way][1] == car_position:
        path.append((int(car_position), int(at_list[way][0])))
        car_position = at_list[way][0]
    for k, v in at_list_dic.iteritems():
        if not (v - at_list[way]).any():
            remain = delete(remain, k, 0)
    cost += at_list[way][2]
    car_cap = car_cap - at_list[way][3]
    return car_cap, car_position, cost, path, remain


# Find the next maybe version destination
def go_situation(path_cost, car_position, remain, car_cap, cost, path, car_index):
    possible_path = creat_possible_path(path_cost[int(car_position) - 1])
    possible_path = possible_path[argsort(possible_path[:, 1])]
    status = False
    for i in range(0, len(possible_path)):
        status, des = check_des(remain, possible_path[i][0], car_cap)
        if status:
            cost += int(path_cost[int(car_position) - 1][int(des) - 1])
            car_position = des
            break
    if not status:
        cost += path_cost[int(car_position) - 1][depot-1]
        car_position = depot
        path.append(0)
        path.append(0)
        car_cap = capacity
        car_index += 1
    return cost, car_position, path, car_cap, car_index


# Set some global variables
def set_globals(file_name):
    global vertices
    vertices_line = str(open(file_name).readlines()[1:2])
    vertices = int(filter(str.isdigit, vertices_line))
    global depot
    depot_line = str(open(file_name).readlines()[2:3])
    depot = int(filter(str.isdigit, depot_line))
    global capacity
    capacity_line = str(open(file_name).readlines()[6:7])
    capacity = int(filter(str.isdigit, capacity_line))
    global graph_edge
    file_new = open(file_name).readlines()
    file_new = file_new[:-1]
    graph_edge = loadtxt(file_new, skiprows=9)


# Calculate the dijkstra martix
def cal_dijk():
    temp_graph = Graph(graph_edge)
    re_path = []
    for j in range(0, vertices):
        temp = []
        visited, path = dijsktra(temp_graph, j)
        for i in range(0, len(visited)):
            temp.append(visited.get(i))
        re_path.append(temp)
    return array(re_path)


# The main dijkstra method
def dijsktra(graph, initial):
    visited = {initial: 0}
    path = {}
    nodes = set(graph.nodes)
    while nodes:
        min_node = None
        for node in nodes:
            if node in visited:
                if min_node is None:
                    min_node = node
                elif visited[node] < visited[min_node]:
                    min_node = node
        if min_node is None:
            break
        nodes.remove(min_node)
        current_weight = visited[min_node]
        for edge in graph.edges[min_node]:
            weight = current_weight + graph.distances[(min(min_node, edge), max(min_node, edge))]
            if edge not in visited or weight < visited[edge]:
                visited[edge] = weight
                path[edge] = min_node
    return visited, path


# Find all the possible path to go next
def creat_possible_path(path_cost_one):
    possible_path = array([0, 1000])
    for i in range(0, len(path_cost_one)):
        row = array([i + 1, path_cost_one[i]])
        possible_path = row_stack((possible_path, row))
    possible_path = delete(possible_path, 0, axis=0)
    return possible_path


# Check if there is a destination via demand edge on current vertex
def check_des(remain, des, car_cap):
    for i in range(0, len(remain)):
        if (remain[i][0] == des or remain[i][1] == des) and remain[i][3] <= car_cap:
            if remain[i][0] == des:
                return True, remain[i][1]
            else:
                return True, remain[i][0]
    return False, 0


# Graph class for dijkstra algorithm to use
class Graph:
    nodes = set()
    edges = []
    distances = {}

    def __init__(self, numpy_array):
        for i in range(0, vertices):
            self.add_node(i)
        for i in range(0, len(numpy_array)):
            self.add_edge(numpy_array[i][0], numpy_array[i][1], numpy_array[i][2])

    def add_node(self, value):
        self.nodes.add(value)
        self.edges.append([])

    def add_edge(self, from_node, to_node, distance):
        self.edges[int(from_node) - 1].append(int(to_node) - 1)
        self.edges[int(to_node) - 1].append(int(from_node) - 1)
        self.distances[(min(int(from_node) - 1, int(to_node) - 1), max(int(from_node) - 1, int(to_node) - 1))] = int(
            distance)

    # Main method to receive the outer parameter using multiprocessing


def main():
    global start
    start = time()
    file_name = sys.argv[1]
    options, args = getopt(sys.argv[2:], "t:s:", [])
    global ter_time
    for name, value in options:
        if name in "-t":
            ter_time = int(value)
        if name in "-s":
            ran_seed = int(value)
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    result = []
    for i in range(8):
        temp_proc = multiprocessing.Process(target=solver, args=(file_name, return_dict))
        result.append(temp_proc)
        temp_proc.start()
    for temp_proc in result:
        temp_proc.join()
    k = sorted(return_dict.items())[0][0]
    res_s = ','.join(str(x) for x in return_dict[k])
    print 's', res_s.replace(' ', '')
    print 'q', int(k)


if __name__ == '__main__':
    main()
