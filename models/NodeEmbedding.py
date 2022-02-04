import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imports import *
from utils import *


class NodeMatchingNetwork(torch.nn.Module):
    def __init__(self, initial_features_dims, device):
        super(NodeMatchingNetwork, self).__init__()

        self.initial_features_dims = initial_features_dims
        self.device = device

        self.gcn1 = DenseGCNConv(in_channels=initial_features_dims, out_channels=4, bias=True)
        self.gcn2 = DenseGCNConv(in_channels=4, out_channels=8, bias=True)
        self.gcn3 = DenseGCNConv(in_channels=8, out_channels=16, bias=True)
        self.gcn4 = DenseGCNConv(in_channels=16, out_channels=32, bias=True)
        self.gcn5 = DenseGCNConv(in_channels=32, out_channels=16, bias=True)
        self.gcn6 = DenseGCNConv(in_channels=16, out_channels=4, bias=True)
        self.gcn7 = DenseGCNConv(in_channels=4, out_channels=2)


        self.mp_w = nn.Parameter(torch.rand(4, 2))



    @staticmethod
    def div_with_small_value(n, d, eps=1e-8):
        # too small values are replaced by 1e-8 to prevent it from exploding.
        d = d * (d > eps).float() + eps * (d <= eps).float()
        return n / d

    def cosine_attention(self, v1, v2):
        """
        :param v1: (batch, len1, dim)
        :param v2: (batch, len2, dim)
        :return:  (batch, len1, len2)
        """
        # (batch, len1, len2)
        a = torch.bmm(v1, v2.permute(0, 2, 1))
        
        v1_norm = v1.norm(p=2, dim=2, keepdim=True)  # (batch, len1, 1)
        v2_norm = v2.norm(p=2, dim=2, keepdim=True).permute(0, 2, 1)  # (batch, len2, 1)
        d = v1_norm * v2_norm
        return self.div_with_small_value(a, d)

    def multi_perspective_match_func(self, v1, v2, w): #fm function
        """
        :param v1: (batch, len, dim)
        :param v2: (batch, len, dim)
        :param w: (perspectives, dim)
        :return: (batch, len, perspectives)
        """
        v1 = torch.stack([v1] * 4, dim=3) # (batch, len, dim, perspectives)
        v2 = torch.stack([v2] * 4, dim=3) # (batch, len, dim, perspectives)
        w = w.transpose(1, 0).unsqueeze(0).unsqueeze(0)  # (1,  1,  dim, perspectives)

        v1 = w * v1 
        v2 = w * v2  
        return functional.cosine_similarity(v1, v2, dim=2)  # (batch, len, perspectives)

    def forward_dense_gcn_layers(self, feat, adj):
        
        feat_in = feat
        for i in range(1, 8):
            feat_out = functional.relu(getattr(self, 'gcn{}'.format(i))(x=feat_in, adj=adj, mask=None, add_loop=False), inplace=True)
            feat_out = functional.dropout(feat_out, p=0.1, training=self.training)
            feat_in = feat_out
        return feat_out

    
    def forward(self, batch_x_p, batch_x_h, batch_adj_p, batch_adj_h, node_i, node_j):
        feature_p_init = torch.FloatTensor(batch_x_p).to(self.device)
        adj_p = torch.FloatTensor(batch_adj_p).to(self.device)
        feature_h_init = torch.FloatTensor(batch_x_h).to(self.device)
        adj_h = torch.FloatTensor(batch_adj_h).to(self.device)

        feature_p = self.forward_dense_gcn_layers(feat=feature_p_init, adj=adj_p)  # (batch, len_p, dim)
        feature_h = self.forward_dense_gcn_layers(feat=feature_h_init, adj=adj_h)  # (batch, len_h, dim)

        """ print(feature_p.size())
        print(feature_h.size()) """

        attention = self.cosine_attention(feature_p, feature_h)  # (batch, len_p, len_h)
        
        attention_h = feature_h.unsqueeze(1) * attention.unsqueeze(3)  # (batch, 1, len_h, dim) * (batch, len_p, len_h, dim) => (batch, len_p, len_h, dim)
        attention_p = feature_p.unsqueeze(2) * attention.unsqueeze(3)  # (batch, len_p, 1, dim) * (batch, len_p, len_h, dim) => (batch, len_p, len_h, dim)
        
        att_mean_h = self.div_with_small_value(attention_h.sum(dim=2), attention.sum(dim=2, keepdim=True))  # (batch, len_p, dim)
        att_mean_p = self.div_with_small_value(attention_p.sum(dim=1), attention.sum(dim=1, keepdim=True).permute(0, 2, 1))  # (batch, len_h, dim)
        
        
        multi_p = self.multi_perspective_match_func(v1=feature_p, v2=att_mean_h, w=self.mp_w)
        multi_h = self.multi_perspective_match_func(v1=feature_h, v2=att_mean_p, w=self.mp_w)
        
        
        match_p = multi_p
        match_h = multi_h


        

        dis = torch.cdist(match_p, match_h, p=2.0).clamp(min=0, max=1)
        preds_attention = []
        preds_match = []
        for i in range(len(node_i)):
            preds_attention.append(int(attention[i][node_i[i]][node_j[i]]))
            preds_match.append(int(dis[i][node_i[i]][node_j[i]]))

        
            
        preds_attention = torch.Tensor(preds_attention)
        print("Preds attention: ", preds_attention)


        """ preds_match = torch.Tensor(preds_match)
        print("Preds Match: ", preds_match) """



        """ sim = functional.cosine_similarity(match_p[0][node_i], match_h[0][node_j], dim=1).clamp(min=0, max=1) #(node_feature1, node_feature2).clamp(min=0, max=1) # # torch.FloatTensor([random.choice([0, 1]) for val in node_i]) 
        print("Sim size: ", sim)
        print("preds size: ", preds) """
        return preds_attention