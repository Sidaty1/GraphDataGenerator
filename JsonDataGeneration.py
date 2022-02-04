
from imports import *



class JsonDataGenerator: 
    def __init__(self, number_of_samples, max_tree_size, min_tree_size, min_number_subtree_nodes, max_number_subtree_nodes, data_dir, data_exists):
        self.number_of_samples = number_of_samples
        self.max_tree_size = max_tree_size
        self.min_tree_size = min_tree_size
        self.min_number_subtree_nodes = min_number_subtree_nodes
        self.max_number_subtree_nodes = max_number_subtree_nodes
        self.data_dir = data_dir
        self.data_exists = data_exists


        # data_generation 
        if not self.data_exists:
            self.data_generator()


    def check_branching(self, graph): 
        queue = []
        for node in graph.nodes(): 
            if len(list(graph.edges(node))) > 3: 
                queue.append(node)
                for val in queue: 
                    sucs = graph.successors(val)
                    for suc in sucs: 
                        queue.append(suc)

        return queue


    def get_random_tree(self, tree_size):
        graph = nx.random_tree(n=tree_size, create_using=nx.DiGraph)
        queue = self.check_branching(graph)    
        for node in queue:
            if node in graph.nodes():
                graph.remove_node(node)

        while len(graph.nodes()) < tree_size :
            graph = nx.random_tree(n=tree_size, create_using=nx.DiGraph)
            queue = self.check_branching(graph)    
            for node in queue:
                if node in graph.nodes():
                    graph.remove_node(node)



        # Ubdate number of nodes
        
        # Set nodes features
        nodes_features = {}
        for node in list(graph.nodes()):
            number_of_connected_edges =  len(graph.edges(node)) 
            angle = [random.randint(-89, 89) for i in range(number_of_connected_edges-1)]
            if len(angle) > 0:
                angle = [ang/90 for ang in angle]
            angle = [0 if x == "nan" else x for x in angle]
            if len(angle) < 4:
                for _ in range(len(angle), 4):
                    angle.append(0)
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

    def get_root(self, tree): 
        for n, d in tree.in_degree():
            if d == 0:
                return n

    def get_random_subtree(self, tree):
        tree_size = len(list(tree.nodes())) 
        found = False
        while(not found):
            subtree_source = random.randint(1, tree_size - 1)
            subtree = dfs_tree(tree, subtree_source)
            if len(list(nx.DiGraph(subtree).nodes())) >= self.min_number_subtree_nodes and len(list(nx.DiGraph(subtree).nodes())) <= self.max_number_subtree_nodes:
                found = True

        subtree_nodes = subtree.nodes()

        subtree_node_features = {}
        for node in subtree_nodes:
            connected_branches = len(subtree.edges(node))
            angle = [ tree.nodes[node]['angle'][i] + random.randint(-15, 15) for i in range(connected_branches - 1)]
            if len(angle) > 0:
                angle = [ang/90 for ang in angle]
            angle = [0 if x == "nan" else x for x in angle]
            if len(angle) < 4:
                for _ in range(len(angle), 4):
                    angle.append(0)
            subtree_node_features[node] = {'angle': angle, 'number_of_connected_edges': connected_branches }

        nx.set_node_attributes(subtree, subtree_node_features)

        subtree_edge_features = {}
        for edge in subtree.edges(): 
            subtree_branch_len = tree.edges[edge]['branch_len']  + random.random()/5
            subtree_branch_radius = tree.edges[edge]['branch_radius'] + random.random()/10

            subtree_edge_features[edge] = {'branch_len': subtree_branch_len, 'branch_radius': subtree_branch_radius}

        nx.set_edge_attributes(subtree, subtree_edge_features)


        return subtree

    def json_to_networkx(self, path): 
        with open(path) as file: 
            js_graph = json.load(file)
        
        return json_graph.tree_graph(js_graph)

    def show_tree(self, tree): 
        source = self.get_root(tree)
        print(nx.forest_str(tree, sources=[source]))

    def data_generator(self): 

        trees_path = self.data_dir + '/trees'
        subtrees_path = self.data_dir + '/subtrees'
        if not os.path.exists(trees_path): 
            os.makedirs(trees_path)
            print('Creating directory for trees json format in path: ', trees_path)

        if not os.path.exists(subtrees_path): 
            os.makedirs(subtrees_path)
            print('Creating directory for subrees json format in path: ', subtrees_path)


        for i in range(self.number_of_samples): 
            tree_size = random.randint(self.min_tree_size, self.max_tree_size)
            
            tree = self.get_random_tree(tree_size)
            subtree = self.get_random_subtree(tree)

            with open(trees_path + '/tree_' + str(i) + '.json', 'a') as outfile:
                tree_json = json_graph.tree_data(tree, root=self.get_root(tree))
                json.dump(tree_json, outfile)

            with open(subtrees_path + '/subtree_' + str(i) + '.json', 'a') as outfile: 
                subtree_json = json_graph.tree_data(subtree, root=self.get_root(subtree))
                json.dump(subtree_json, outfile)

        print("Data scussefully generated and stored in ", trees_path, " and ", subtrees_path)





if __name__ == '__main__': 

    pass

    #writed_subtree = json_graph.tree_graph('/home/sidaty/Desktop/GraphDataGenerator/data/subtrees/subtree_0.json')
    

    
    #datageneration = JsonDataGenerator(1000, 100, 80, 10, 20, './data/json')

    """ tree = datageneration.json_to_networkx('/home/sidaty/Desktop/GraphDataGenerator/data/subtrees/subtree_0.json')
    datageneration.show_tree(tree) """
    #datagen.data_writer()

    """ tree = datagen.get_random_tree(100)
    subtree = datagen.get_random_subtree(tree)

    sub_source = datagen.get_root(subtree)

    isolated_nodes = list(nx.isolates(tree))
    print(len(isolated_nodes))

    for node in tree.nodes:
        print(node, ": ", tree.nodes[node])




    print(nx.forest_str(tree, sources=[0]))



    for node in subtree.nodes:
        print(node, ": ", subtree.nodes[node])


    print(nx.forest_str(subtree))


    print(sub_source) """
    """ isolated_nodes = list(nx.isolates(subtree))
    print(len(isolated_nodes))
    """

    """ print(nx.forest_str(subtree))
    writed_subtree = json_graph.tree_data(subtree, root=subtree_source)
    writed_tree = json_graph.tree_data(subtree, root=subtree_source) """
    """ writed_subtree = json_graph.tree_graph(data=writed_subtree)
    print(nx.forest_str(writed_subtree)) """
    """ with open('./data/subtrees/graph1.json', "a") as outfile: 
        json.dump(writed_subtree, outfile)


    with open('./data/trees/graph1.json', "a") as outfile: 
        json.dump(writed_tree, outfile) """
    #print(check_branching(tree))

