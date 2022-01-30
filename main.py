from operator import sub
from imports import *


def get_random_graph(number_nodes, probability_edge):

    # Generate random graph, complexity = O(n^2)
    graph = nx.erdos_renyi_graph(number_nodes, probability_edge)
    
    # Remove isolated nodes
    isolated_nodes = list(nx.isolates(graph))
    graph.remove_nodes_from(isolated_nodes)

    # Ubdate number of nodes
    number_nodes = graph.number_of_nodes()
    
    # Set nodes features
    nodes_features = {}
    for node in list(graph.nodes()):
        angle = random.randint(-314, 314)/100
        number_of_connected_edges =  len(graph.edges(node))
        
        nodes_features[node] = {'angle': angle, 'number_of_connected_edges': number_of_connected_edges}

    nx.set_node_attributes(graph, nodes_features)

    # Set edges features
    edges_features = {}
    for edge in graph.edges():
        branch_len = random.randint(1, 9) + random.random()
        branch_radius = random.random()/10
        edges_features[edge] = {'branch_len': branch_len, 'branch_radius': branch_radius}

    nx.set_edge_attributes(graph, edges_features)

    return graph


def get_random_list(list):
    size = random.randint(3, len(list))
    random_list = []
    for i in range(size):
        node = random.choice(list)
        random_list.append(node)
        list.remove(node)

    return random_list


def get_subgraph(graph):

    list_of_nodes = get_random_list(list(graph.nodes()))
    subgraph = graph.subgraph(list_of_nodes)
    subgraph = nx.Graph(subgraph)
    isolated_nodes = list(nx.isolates(subgraph))
    subgraph.remove_nodes_from(isolated_nodes)

    for node in list(subgraph.nodes()):
        number_of_connected_edges =  len(subgraph.edges(node))
        subgraph.nodes[node]['number_of_connected_edges'] = number_of_connected_edges

    return subgraph
   

def get_data(Nb_of_samples, max_number_of_nodes, min_num_of_nodes, edge_probability=0.6):
    training_data = []

    for i in range(Nb_of_samples):
        sample = {}        
        nb_nodes = random.randint(min_num_nodes, max_num_nodes)
        
        graph = get_random_graph(nb_nodes, edge_probability)
        subgraph = get_subgraph(graph)
        
        sample['Graph'] = graph
        sample['Subgraph'] = subgraph

        training_data.append(sample)

    return training_data


""" training_data = get_data(10, 20, 10)

sample = training_data[3]

graph = sample['Graph']
subgraph = sample['Subgraph']


for node in list(graph.nodes()):
    print(graph.nodes[node])
print("\n")
for node in list(subgraph.nodes()):
    print(subgraph.nodes[node]) """

def data_to_json_file():
    return NotImplementedError()

