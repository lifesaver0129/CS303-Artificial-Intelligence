from getopt import *
from time import *
from numpy import *
import sys
from random import *


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


def ic_model(target_graph, seed_set):
    act_set = seed_set.copy()
    all_set = seed_set.copy()
    count = len(act_set)
    while len(act_set) != 0:
        new_act_set = set()
        for item in act_set:
            for neighbor in target_graph.edges[item - 1]:
                if random() < target_graph.weight[(item - 1, neighbor)]:
                    if neighbor + 1 not in all_set:
                        new_act_set.add(neighbor + 1)
                        all_set.add(neighbor + 1)
        count = count + len(new_act_set)
        act_set = new_act_set.copy()
    return count


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


def main():
    start_time = time()
    options, args = getopt(sys.argv[1:], "i:s:m:b:t:r:", [])
    for syntax, value in options:
        if syntax in "-i":
            network_path = value
        if syntax in "-s":
            seed_path = value
        if syntax in "-m":
            diff_model = value
        if syntax in "-b":
            ter_type = int(value)
        if syntax in "-t":
            ter_time = int(value)
        if syntax in "-r":
            ran_seed = int(value)
    seed(ran_seed)
    vertex_num, edge_num, graph_numpy = network_reader(network_path)
    seeds = seed_reader(seed_path)
    graph_class = Graph(graph_numpy, vertex_num)
    if ter_type == 0:
        ter_time = 10000
    sum, iter = 0, 0
    for i in range(0, 10000):
        if diff_model == "IC":
            count = ic_model(graph_class, seeds)
        else:
            count = lt_model(graph_class, seeds)
        sum = count + sum
        iter += 1
        if ter_time - 3 < time() - start_time:
            break
    print sum / iter
    print time()-start_time

if __name__ == '__main__':
    main()
