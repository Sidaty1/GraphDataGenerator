
from imports import *

def get_random_list(list):
    size = random.randint(min_num_nodes, len(list))
    random_list = []
    for i in range(size):
        node = random.choice(list)
        random_list.append(node)
        list.remove(node)

    return random_list

def get_random_graph(tree_size, probability_edge):
    tree_size = 40
    graph = nx.random_tree(n=tree_size, create_using=nx.DiGraph)

    # Ubdate number of nodes
    
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


def get_subgraph(tree):
    tree_size = 40 
    found = False
    while(not found):
        subtree_source = random.randint(1, tree_size - 1)
        subtree = dfs_tree(tree, subtree_source)
        if len(list(nx.DiGraph(subtree).nodes())) >= min_num_nodes and len(list(nx.DiGraph(subtree).nodes())) <= max_num_nodes:
            found = True

    subtree_nodes = subtree.nodes()

    subtree_node_features = {}
    for node in subtree_nodes:
        angle_noise = random.random()/5
        connected_branches = len(subtree.edges(node))
        subtree_node_features[node] = {'angle': tree.nodes[node]['angle'] + angle_noise, 'number_of_connected_edges': connected_branches }

    nx.set_node_attributes(subtree, subtree_node_features)

    subtree_edge_features = {}
    for edge in subtree.edges(): 
        subtree_branch_len = tree.edges[edge]['branch_len']  + random.random()/5
        subtree_branch_radius = tree.edges[edge]['branch_radius'] + random.random()/10

        subtree_edge_features[edge] = {'branch_len': subtree_branch_len, 'branch_radius': subtree_branch_radius}

    nx.set_edge_attributes(subtree, subtree_edge_features)


    return subtree
""" def get_random_graph(number_nodes, probability_edge):

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

        return subgraph """

def get_node_samples(graph, subgraph):
    samples = []
    cache = []
    for source_node in list(graph.nodes()):
        for target_node in list(subgraph.nodes()):
            sample = {}
            sample['source_node'] = graph.nodes[source_node]
            sample['target_node'] = subgraph.nodes[target_node]
            if source_node == target_node:
                sample['simularity'] = 1
            else:
                sample['simularity'] = 0
            if str(source_node) + str(target_node) not in cache and str(target_node) + str(source_node) not in cache:
                samples.append(sample)
                cache.append(str(source_node) + str(target_node))

    return samples

def get_graph_features(graph, node_i):
    graph_features = []
    count = 0
    for node in list(graph.nodes()):
        graph_feature = graph.nodes[node]
        #graph_feature = graph_features.items()
        graph_features.append([graph_feature['angle'], graph_feature['number_of_connected_edges']])
        if node == node_i:
            node_i = count
        else: 
            count += 1

    adj = nx.adjacency_matrix(graph)
    adj = adj.todense()
    adj = np.array(adj)

    return graph_features, adj, node_i


            