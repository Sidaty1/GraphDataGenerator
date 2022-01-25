from imports import *
from parameters import *



def get_data_list():
    """
        This function generate randomly N sample of: 
            - Randomly generated graph G
            - Randomly generated subgraph of G
    """
    data = []
    for i in range(Nb_graphs):
    # Generate random number of nodes
        nb_nodes = random.randint(min_num_nodes, max_num_nodes)

        # Generate random graph, complexity = O(n^2)
        graph = nx.erdos_renyi_graph(nb_nodes, edge_probability)

        # Remove isolated nodes
        isolated_nodes = list(nx.isolates(graph))
        graph.remove_nodes_from(isolated_nodes)

        # Get number of nodes
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
                    
            # Get a subgraph based on the generated random nodes 
            subgraph = graph.subgraph(random_nodes)
            
            # ReCheck that the subgraph does not contain isolated nodes
            isolated_nodes = list(nx.isolates(subgraph))
            if len(isolated_nodes) == 0:
                """ # Get correspondant nodes: will serve as output for the model
                nodes_correspondances = []
                for k in range(len(random_nodes)):
                    nodes_correspondances.append((k, random_nodes[k]))
                 """
                # fill in data
                data.append([graph, subgraph])

    return data

def data_to_json_file():
    return NotImplementedError()

