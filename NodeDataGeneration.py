from re import sub
from imports import *

from JsonDataGeneration import JsonDataGenerator


class NodeDataGenerator(JsonDataGenerator):

    def __init__(self, number_of_samples, number_of_trees, max_tree_size, min_tree_size, min_number_subtree_nodes, max_number_subtree_nodes, data_dir, data_exists):
        super().__init__(number_of_trees, max_tree_size, min_tree_size, min_number_subtree_nodes, max_number_subtree_nodes, data_dir, data_exists)


        self.number_of_samples = number_of_samples

        self.data_to_csv_file()



    def nodes_data_generator(self): 
        df = pd.DataFrame()
        samples = []
        for i in range(int(self.number_of_samples/10)): 
            tree_path = self.data_dir + '/trees/tree_' + str(i) + '.json'
            subtree_path = self.data_dir + '/subtrees/subtree_' + str(i) + '.json'

            tree = self.json_to_networkx(tree_path)
            subtree = self.json_to_networkx(subtree_path)

            for j in range(10): 
                if j % 2 == 0: 
                    subtree_node = random.choice(list(subtree.nodes()))
                    sample = dict(trees_index=i, subtree_index=i, node_tree_index=subtree_node, subtree_node_index=subtree_node, simularity=1)
                    samples.append(sample)
                else: 
                    found = False
                    while(not found): 
                        tree_node = random.choice(list(tree.nodes()))
                        subtree_node = random.choice(list(subtree.nodes()))
                        sample = dict(trees_index=i, subtree_index=i, node_tree_index=tree_node, subtree_node_index=subtree_node, simularity=0)
                        if tree_node != subtree_node and sample not in samples: 
                            samples.append(sample)
                            found = True

        for sample in samples:
            df = df.append(sample, ignore_index=True)

        return df

    def data_to_csv_file(self): 
        dataframe = self.nodes_data_generator()

        file_path = self.data_dir + '/csv_nodes_data.csv'
        with open(file_path, 'w+') as file:
            dataframe.to_csv(file_path)

        







    
if __name__ == '__main__': 
    
    NodeDataGenerator(40, 1000, 100, 80, 10, 20, './data/json', True)



    