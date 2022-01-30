from imports import *
from utils import *

class DataManager:
    def __init__(self, Nb_of_samples, max_number_of_nodes, min_num_of_nodes, edge_probability=0.6):
        self.Nb_of_samples = Nb_of_samples
        self.max_number_of_nodes = max_number_of_nodes
        self.min_number_of_nodes = min_num_nodes
        self.edge_probability = edge_probability


    def get_data(self):
        training_data = []

        for i in range(self.Nb_of_samples):
            sample = {}        
            nb_nodes = random.randint(self.min_number_of_nodes, self.max_number_of_nodes)
            
            graph = get_random_graph(nb_nodes, self.edge_probability)
            subgraph = get_subgraph(graph)
            
            sample['Graph'] = graph
            sample['Subgraph'] = subgraph

            training_data.append(sample)

        return training_data

    def data_to_json_file(self):
        return NotImplementedError()



if __name__ == '__main__':
    datamanager = DataManager(Nb_graphs, min_num_nodes, max_num_nodes)


    training_data = datamanager.get_data()

    sample = training_data[3]

    graph = sample['Graph']
    subgraph = sample['Subgraph']


    for node in list(graph.nodes()):
        print(graph.nodes[node])
    print("\n")
    for node in list(subgraph.nodes()):
        print(subgraph.nodes[node])



