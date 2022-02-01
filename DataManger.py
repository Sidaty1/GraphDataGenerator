from imports import *
from utils import *

class DataManager:
    def __init__(self, edge_probability=0.6):
        self.max_number_of_nodes = max_num_nodes
        self.min_number_of_nodes = min_num_nodes
        self.edge_probability = edge_probability


    def random_graphs(self):

        sample = {}        
        nb_nodes = random.randint(self.min_number_of_nodes, self.max_number_of_nodes)
        
        graph = get_random_graph(nb_nodes, self.edge_probability)
        subgraph = get_subgraph(graph)
        
        sample['Graph'] = graph
        sample['Subgraph'] = subgraph

        return sample

        
    @staticmethod
    def transform_sample(sample):
        new_sample = {}
        input = dict(Graph=sample['Graph'], Subgraph=sample['Subgraph'], Node_i=sample['Node_i'], Node_j=sample['Node_j'])
        
        new_sample['Input'] = input
        new_sample['Output'] = sample['pred']

        return new_sample

    @staticmethod
    def shuffle(dataset):

        new_dataset = []
        permutation = np.random.permutation(len(dataset))
        for elem in permutation:
            new_dataset.append(dataset[elem])

        return new_dataset

    def get_sample_positive(self, data_i):
        
        graph = data_i['Graph']
        subgraph = data_i['Subgraph']

        subgraph_nodes = list(subgraph.nodes())

        node_j = random.choice(subgraph_nodes)

        sample = dict(Graph=graph, Subgraph=subgraph, Node_i=node_j, Node_j=node_j, pred=1)
        sample = self.transform_sample(sample)
        return sample

    def get_sample_negative(self, data_i):
        
        graph = data_i['Graph']
        subgraph = data_i['Subgraph']

        subgraph_nodes = list(subgraph.nodes())
        graph_nodes = list(graph.nodes)
        found = False
        
        node_i = random.choice(graph_nodes)

        while(not found):
            node_j = random.choice(subgraph_nodes)
            if node_i != node_j:
                found = True

        sample = dict(Graph=graph, Subgraph=subgraph, Node_i=node_i, Node_j=node_j, pred=0)
        sample = self.transform_sample(sample)
        return sample

    def get_batch(self, data_i, this_batch_size):       
        batch = []
        positive = True
        
        while len(batch) < this_batch_size:
            if positive:
                sample = self.get_sample_positive(data_i)
                if sample not in batch:
                    batch.append(sample)
                    positive = False

            else: 
                sample = self.get_sample_negative(data_i)
                if sample not in batch:
                    batch.append(sample)
                    positive = True
        return batch


    def get_dataset_per_batch(self, this_number_of_batchs, this_batch_size): 
        dataset = []
        
        for i in range(this_number_of_batchs):
            sample = self.random_graphs()
            batch = self.get_batch(sample, this_batch_size)
            dataset.append(batch)


        return dataset

    
    def get_dataset(self):
        dataset = []
        dataset_per_batch = self.get_dataset_per_batch()
        for batch in dataset_per_batch:
            for sample in batch:
                dataset.append(sample)


        dataset = self.shuffle(dataset)
        return dataset

    def get_splited_dataset(self, dataset, ratio):  # takes a dataset non splited per batch !! 
        training_ratio = int(ratio[0] *len(dataset)) 
        val_ratio = int(ratio[1]*len(dataset))
        test_ratio = int(ratio[2]*len(dataset))

        permutation = np.random.permutation(len(dataset))


        train_set, val_set, test_set = [], [], []
        for i in range(0, training_ratio): 
            train_set.append(dataset[permutation[i]])
        
        for i in range(training_ratio, training_ratio + val_ratio):
            val_set.append(dataset[permutation[i]])

        for i in range(training_ratio + val_ratio, training_ratio + val_ratio + test_ratio): 
            test_set.append(dataset[permutation[i]])

        return train_set, val_set, test_set

    """ def split_data_per_epoch(self, dataset):
        data = []

        for i in range(batch_number): """


    def split_epoch_per_batch(self, epoch_data, nb_batch, batch_size):

        pass

    def get_epoch_data(self, data_epoch):
        """ dataset = self.get_dataset()

        train, val, test = self.get_splited_dataset(dataset, [0.8, 0.1, 0.1])
        
        train_epoch = self.split_epoch_per_batch(train, train, nb_batch=batch_number, batch_size=batch_size)
        val_epoch = self.split_epoch_per_batch(train, val, nb_batch=batch_number, batch_size=batch_size)
        test_epoch = self.split_epoch_per_batch(train, test, nb_batch=batch_number, batch_size=batch_size) """
        
        new_data_epoch = []

        for data_batch in data_epoch:
            features1 = []
            features2 = []
            adjs1 = []
            adjs2 = []
            nodes1 = []
            nodes2 = []
            preds = []
            for sample in data_batch:
                input_data = sample['Input']
                pred = sample['Output']

                graph1 = input_data['Graph']
                graph2 = input_data["Subgraph"]
                node1 = input_data['Node_i']
                node2 = input_data['Node_j']

                feature1, adj1, node1 = get_graph_features(graph1, node1)
                feature2, adj2, node2 = get_graph_features(graph2, node2)

                features1.append(feature1)
                adjs1.append(adj1)
                nodes1.append(node1)

                features2.append(feature2)
                adjs2.append(adj2)
                nodes2.append(node2)

                preds.append(pred)

            new_data_epoch.append([features1, features2, adjs1, adjs2, nodes1, nodes2, preds])

        return new_data_epoch




    def data_to_json_file(self):
        return NotImplementedError()



if __name__ == '__main__':
    datamanager = DataManager()


    """ training_data = datamanager.random_graphs()

    sample = training_data

    

    batch = datamanager.get_batch(sample) """

    """ dataset = datamanager.get_dataset()

    train, val, test = datamanager.get_splited_dataset(dataset, [0.8, 0.1, 0.1])

    print(len(train))
    print(len(val))
    print(len(test)) """

    data_epoch = datamanager.get_dataset_per_batch(this_batch_size=1, this_number_of_batchs=1)
    a = datamanager.get_epoch_data(data_epoch)
    
    print(a)




