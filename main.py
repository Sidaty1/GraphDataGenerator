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
    for i in range(number_nodes):
        angle = random.randint(-314, 314)/100
        try: 
            number_of_connected_edges =  len(graph.edges(i))
        except:
            number_of_connected_edges = 1
        
        nodes_features[i] = {'angle': angle, 'number_of_connected_edges': number_of_connected_edges}

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
    nb_nodes = graph.number_of_nodes()
    if nb_nodes > 3:
        # Generate random nodes for the subgraph 
        random_nodes = []
        random_size_of_subgraph = random.randint(3, nb_nodes)
        
        # Get a list of random nodes that represent the subgraph
        while len(random_nodes) < random_size_of_subgraph:
            for j in range(random_size_of_subgraph): 
                random_node = random.randint(0, nb_nodes)
                if random_node not in random_nodes and len(random_nodes) < random_size_of_subgraph:
                    random_nodes.append(random_node)

        #random_nodes = list(dict.fromkeys(random_nodes))
                
        # Get a subgraph based on the generated random nodes 
        subgraph = graph.subgraph(random_nodes)

        return subgraph, random_nodes
    else:
        print("Graph require number of nodes > 3")

    
graph = get_random_graph(10, 0.6)

number_nodes = graph.number_of_nodes()
number_edges = graph.number_of_edges()

for i in range(number_nodes):
    print(graph.nodes[i])

print("\n\n")

for edge in graph.edges():
    print(graph.edges[edge])


subgraph = get_subgraph(graph)




def data_to_json_file():
    return NotImplementedError()

