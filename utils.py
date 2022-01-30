from imports import *

def get_random_list(list):
    size = random.randint(3, len(list))
    random_list = []
    for i in range(size):
        node = random.choice(list)
        random_list.append(node)
        list.remove(node)

    return random_list

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