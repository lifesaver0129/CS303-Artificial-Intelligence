from getopt import *
from time import *
from numpy import *
import sys
from random import *

start_time = 0
found = 0
got_value = 0.0
ter_time = 0
R = 10000


# Read the network file
def network_reader(file_path):
    line_one = open(file_path).readline()
    ovr_data = str.split(line_one)
    vertex_num = int(ovr_data[0])
    edge_num = int(ovr_data[1])
    graph_edge = loadtxt(file_path, skiprows=1)
    return vertex_num, edge_num, graph_edge


# Read the seed file
def seed_reader(file_path):
    seeds = set()
    lines = open(file_path).readlines()
    for line in lines:
        seeds.add(int(line.split()[0]))
    return seeds


# The achieve of basic IC model
def ic_model(target_graph, seed_set):
    act_set = seed_set.copy()
    all_set = seed_set.copy()
    count = len(act_set)
    while len(act_set) != 0:
        new_act_set = set()
        for item in act_set:
            for neighbor in target_graph.edges[int(item) - 1]:
                if random() < target_graph.weight[(item - 1, neighbor)]:
                    if neighbor + 1 not in all_set:
                        new_act_set.add(neighbor + 1)
                        all_set.add(neighbor + 1)
        count = count + len(new_act_set)
        act_set = new_act_set.copy()
    return count


# The achieve basic LT model
def lt_model(target_graph, seed_set):
    act_set = seed_set.copy()
    all_set = seed_set.copy()
    count = len(act_set)
    thres = {}
    for i in range(len(target_graph.nodes)):
        thres[i] = random()
        if random() == 0:
            act_set.add(i)
            all_set.add(i)
    while len(act_set) != 0:
        new_act_set = set()
        for item in act_set:
            for neighbor in target_graph.edges[item - 1]:
                tol_weight = 0
                for nei_nei in target_graph.in_edges[neighbor]:
                    if nei_nei + 1 in all_set:
                        tol_weight = tol_weight + target_graph.weight[(nei_nei, neighbor)]
                if tol_weight > thres[neighbor]:
                    if neighbor + 1 not in all_set:
                        new_act_set.add(neighbor + 1)
                        all_set.add(neighbor + 1)
        count = count + len(new_act_set)
        act_set = new_act_set.copy()
    return count


class Graph:
    nodes = set()
    edges = []
    in_edges = []
    weight = {}

    def __init__(self, numpy_array, num_vertex):
        for i in range(0, num_vertex):
            self.add_node(i)
        for i in range(0, len(numpy_array)):
            self.add_edge(numpy_array[i][0], numpy_array[i][1], numpy_array[i][2])

    def add_node(self, value):
        self.nodes.add(value)
        self.edges.append([])
        self.in_edges.append([])

    def add_edge(self, from_node, to_node, weight):
        self.edges[int(from_node) - 1].append(int(to_node) - 1)
        self.in_edges[int(to_node) - 1].append(int(from_node) - 1)
        self.weight[(int(from_node) - 1, int(to_node) - 1)] = weight


def hill_greedy(graph_class, seed_size, diff_model):
    curr_num = 0
    ans_set = set()
    while curr_num != seed_size:
        curr_high = 0.0
        curr_point = 0
        for i in range(len(graph_class.nodes)):
            if i + 1 not in ans_set:
                curr_set = ans_set.copy()
                curr_set.add(i + 1)
                sum = 0.0
                for j in range(0, 10000):
                    if diff_model == "IC":
                        count = ic_model(graph_class, curr_set)
                    else:
                        count = lt_model(graph_class, curr_set)
                    sum = count + sum
                if sum / 10000.0 > curr_high:
                    curr_point = i + 1
                    curr_high = sum / 10000.0
        ans_set.add(curr_point)
        curr_num += 1
    return ans_set


def spread_check_dig(graph_class, seeds, diff_model):
    global R
    sum = 0.0
    if time() - start_time > ter_time * 0.7:
        R = 1000
    if time() - start_time > time() * 0.9:
        R = 100
    for j in range(0, R):
        if diff_model == "IC":
            count = ic_model(graph_class, seeds)
        else:
            count = lt_model(graph_class, seeds)
        sum = count + sum
    return sum / float(R)


#
# def spread_check_int(graph_class, seeds, diff_model):
#     sum = 0
#     for j in range(0, 100):
#         if diff_model == "IC":
#             count = ic_model(graph_class, seeds)
#         else:
#             count = lt_model(graph_class, seeds)
#         sum = count + sum
#     return sum / 100


def celf_improved(target_seed, graph_class, seed_size, diff_model):
    global got_value
    got_value = 0.0
    curr_num = 0
    ans_set = set()
    res = array([-1, -1, -1])
    while curr_num != seed_size:
        if len(ans_set) != 0:
            while True:
                tem_set = ans_set.copy()
                if res[0][2] == curr_num:
                    ans_set.add(int(res[0][0]) + 1)
                    curr_num += 1
                    res = res[res[:, 1].argsort()]
                    res = res[::-1]
                    res = delete(res, 0, 0)
                    break
                tem_set.add(int(res[0][0]) + 1)
                new_key = spread_check_dig(graph_class, tem_set, diff_model)
                if res[1][1] <= new_key - got_value:
                    ans_set.add(int(res[0][0]) + 1)
                    curr_num += 1
                    res = delete(res, 0, 0)
                    if curr_num == seed_size:
                        break
                    got_value = new_key
                else:
                    res[0][1] = new_key - got_value
                    res[0][2] = curr_num
                    res = res[res[:, 1].argsort()]
                    res = res[::-1]
        else:
            for i in target_seed:
                curr_set = set([i + 1])
                sum = spread_check_dig(graph_class, curr_set, diff_model)
                res = row_stack((res, [i, sum, 0]))
                if res[0][1] == -1:
                    res = delete(res, 0, 0)
            res = res[res[:, 1].argsort()]
            res = res[::-1]
            ans_set.add(int(res[0][0]) + 1)
            got_value = res[0][1]
            curr_num += 1
            res = delete(res, 0, 0)
    return ans_set


def degree_dis(graph_class, seed_size):
    ans_set = set()
    all_set = graph_class.nodes.copy()
    degree = {}
    discount_degree = {}
    t_selected = {}
    for vertex in all_set:
        degree[vertex] = len(graph_class.edges[vertex]) + len(graph_class.in_edges[vertex])
        discount_degree[vertex] = len(graph_class.edges[vertex]) + len(graph_class.in_edges[vertex])
        t_selected[vertex] = 0
    for i in range(seed_size):
        max_vertex = sorted(discount_degree.iteritems(),
                            key=lambda (k, v): v, reverse=True)[0][0]
        del discount_degree[max_vertex]
        ans_set.add(max_vertex)
        all_set.remove(max_vertex)
        for vertex in graph_class.edges[max_vertex]:
            if vertex in all_set:
                t_selected[vertex] += 1
                discount_degree[vertex] = degree[vertex] - 2 * t_selected[vertex] - \
                                          (degree[vertex] - t_selected[vertex]) * \
                                          t_selected[vertex] * 0.8
    return ans_set


def main():
    global start_time, ter_time
    start_time = time()
    options, args = getopt(sys.argv[1:], "i:k:m:b:t:r:", [])
    for syntax, value in options:
        if syntax in "-i":
            network_path = value
        if syntax in "-k":
            seed_size = int(value)
        if syntax in "-m":
            diff_model = value
        if syntax in "-b":
            ter_type = int(value)
        if syntax in "-t":
            ter_time = int(value)
        if syntax in "-r":
            ran_seed = int(value)
    if ter_type == 0:
        ter_time = 10000
    seed(ran_seed)
    # network_path = "network.txt"
    # seed_size = 4
    # diff_model = "IC"
    vertex_num, edge_num, graph_numpy = network_reader(network_path)
    graph_class = Graph(graph_numpy, vertex_num)
    tar = degree_dis(graph_class, min(8 * seed_size, vertex_num))
    seeds = celf_improved(tar, graph_class, seed_size, diff_model)
    for i in seeds:
        print i
    # print "Time cost: ", time() - start_time
    # summ = 0
    # for i in range(0, 10000):
    #     count = ic_model(graph_class, seeds)
    #     summ = count + summ
    # print "In IC model spread: ", summ / 10000.0
    # sumn = 0
    # for i in range(0, 10000):
    #     count = lt_model(graph_class, seeds)
    #     sumn = count + sumn
    # print "In LF model spread: ", sumn / 10000.0


if __name__ == '__main__':
    main()
