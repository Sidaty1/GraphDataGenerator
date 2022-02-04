
from imports import *
from utils import *

from models.NodeEmbedding import NodeMatchingNetwork
from DataManger import DataManager



class Trainer(object):

    def __init__(self, initial_features_dims, device, datamanager, best_model_file):

        self.epochs = epochs
        self.lr = lr
        self.device = device

        self.best_model_file = best_model_file

        self.model = NodeMatchingNetwork(initial_features_dims=initial_features_dims, device=self.device).to(device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.train_data = datamanager.get_dataset_per_batch(this_batch_size=10, this_number_of_batchs=100)
        self.train_data = datamanager.get_epoch_data(self.train_data)

        self.val_data = datamanager.get_dataset_per_batch(this_batch_size=10, this_number_of_batchs=10)
        self.val_data = datamanager.get_epoch_data(self.val_data)

        self.test_data = datamanager.get_dataset_per_batch(this_batch_size=10, this_number_of_batchs=10)
        self.test_data = datamanager.get_epoch_data(self.test_data)


        """ init_val_auc = self.eval_auc_epoch(model=self.model, eval_epoch_data=self.val_data)
        print("Initial auc: ", init_val_auc) """

        
    def fit(self):
        best_val_auc = None
        for i in range(1, self.epochs + 1):
            loss_vag = self.train_one_epoch(model=self.model, optimizer=self.optimizer, epoch_data=self.train_data, device=self.device)
            print(f"Epoch {i}/{self.epochs}:\t MSE loss = {loss_vag} @ {datetime.now()}")

            valid_auc = self.eval_auc_epoch(model=self.model, eval_epoch_data=self.val_data)
            print("Validation AUC = {0} @ {1}".format(valid_auc, datetime.now()))
            if best_val_auc is None or best_val_auc < valid_auc:
                print('Validation AUC increased ({} ---> {}), and saving the model ... '.format(best_val_auc, valid_auc))
                best_val_auc = valid_auc
                torch.save(self.model.state_dict(), self.best_model_file)

            print('Best Validation auc = {} '.format(best_val_auc))

        return best_val_auc

    def train_one_epoch(self, model, optimizer, epoch_data, device):
        model.train()
        
        permutation = np.random.permutation(len(epoch_data))

        loss = 0.0
        sample_index = 0
        for val in permutation:
            sample = epoch_data[val]

            features1, features2, adj1, adj2, nodes1, nodes2, pred = sample
            pred = torch.FloatTensor(pred).to(device)
            print("Ground Truth: ", pred)
            batch_pred = model(features1, features2, adj1, adj2, nodes1, nodes2)


            mse_loss = torch.nn.functional.mse_loss(batch_pred, pred) 
            mse_loss.requires_grad=True
            optimizer.zero_grad()
            mse_loss.backward()
            optimizer.step()

            loss += mse_loss

            #if sample_index % int(len(permutation)/10) == 0:
            #print('\tTraining: {}/{}: index = {} loss = {}'.format(sample_index, len(epoch_data), sample_index, mse_loss))

            sample_index += 1

        return loss / len(permutation)

    
    @staticmethod
    def eval_auc_epoch(model, eval_epoch_data):
        model.eval()
        with torch.no_grad():
            good_pred = 0
            total_pred = 0
            for cur_data in eval_epoch_data:
                x1, x2, adj1, adj2, nodes1, nodes2, y = cur_data
                batch_output = model(x1, x2, adj1, adj2, nodes1, nodes2)
                
                #if len(batch_output) == len(y): 
                for i in range(len(y)):
                    if y[i] == batch_output[i]: 
                        good_pred += 1
                
                total_pred += len(batch_output)          

            print("number of pred:", total_pred)      
            print("number of good pred:", good_pred)
        
        return good_pred/total_pred

    """ @staticmethod
    def eval_auc_epoch(model, eval_epoch_data):
        model.eval()
        with torch.no_grad():
            tot_diff = []
            tot_truth = []
            for cur_data in eval_epoch_data:
                x1, x2, adj1, adj2, nodes1, nodes2, y = cur_data
                batch_output = model(x1, x2, adj1, adj2, nodes1, nodes2)
                
                tot_diff += list(batch_output.data.cpu().numpy())
                tot_truth += [y[i] for i in range(len(y))  if y[i] > 0]
        
        diff = np.array(tot_diff) 
        truth = np.array(tot_truth)
        print(diff)
        print(truth)
        
        fpr, tpr, _ = roc_curve(truth, diff)
        model_auc = auc(fpr, tpr)
        return model_auc """


    
    

if __name__ == '__main__':
    datamanager = DataManager()
    trainer = Trainer(initial_features_dims=2, device='cpu', datamanager=datamanager, best_model_file='files/best_model.file')
    best_val_auc = trainer.fit()

