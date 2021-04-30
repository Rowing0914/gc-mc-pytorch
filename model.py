import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, BilinearMixture


class GAE(nn.Module):
    def __init__(self,
                 num_users,
                 num_items,
                 num_classes,
                 num_side_features,
                 nb,
                 u_features,
                 v_features,
                 u_features_side,
                 v_features_side,
                 input_dim,
                 emb_dim,
                 hidden,
                 dropout):
        super(GAE, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.dropout = dropout

        self.u_features = u_features
        self.v_features = v_features
        self.u_features_side = u_features_side
        self.v_features_side = v_features_side

        self.gcl1 = GraphConvolution(input_dim=input_dim,
                                     hidden_dim=hidden[0],
                                     num_classes=num_classes,
                                     act=torch.relu,
                                     dropout=self.dropout,
                                     bias=True)
        self.gcl2 = GraphConvolution(input_dim=hidden[0],
                                     hidden_dim=hidden[1],
                                     num_classes=num_classes,
                                     act=torch.relu,
                                     dropout=self.dropout,
                                     bias=True)
        self.denseu1 = nn.Linear(num_side_features, emb_dim, bias=True)
        self.densev1 = nn.Linear(num_side_features, emb_dim, bias=True)
        self.denseu2 = nn.Linear(emb_dim + hidden[1], hidden[2], bias=False)
        self.densev2 = nn.Linear(emb_dim + hidden[1], hidden[2], bias=False)

        self.bilin_dec = BilinearMixture(num_users=num_users,
                                         num_items=num_items,
                                         num_classes=num_classes,
                                         input_dim=hidden[2],
                                         nb=nb,
                                         dropout=0.0)

    def forward(self, u, v, r_matrix):
        # args: u_feat, v_feat, u, v, support
        u_z, v_z = self.gcl1(self.u_features, self.v_features, range(self.num_users), range(self.num_items), r_matrix)
        u_z, v_z = self.gcl2(u_z, v_z, u, v, r_matrix)

        u_f = torch.relu(self.denseu1(self.u_features_side[u]))
        v_f = torch.relu(self.densev1(self.v_features_side[v]))

        u_h = self.denseu2(F.dropout(torch.cat((u_z, u_f), 1), self.dropout))
        v_h = self.densev2(F.dropout(torch.cat((v_z, v_f), 1), self.dropout))

        output, m_hat = self.bilin_dec(u_h, v_h, u, v)
        return output, m_hat, {"user_embedding": u_z.cpu().detach().numpy(),
                               "item_embedding": v_z.cpu().detach().numpy()}


def _test():
    print("=== test ===")
    device = "cpu"
    hidden = [64, 32, 16, 8]
    nb = 2
    dropout = 0.7
    emb_dim = 32
    num_ratings = 5
    num_side_features = 41
    num_users, num_items = 943, 1682

    u_features = torch.randn(num_users, num_users + num_items).to(device=device)
    u_features_side = torch.randn(num_users, num_side_features).to(device=device)
    v_features = torch.randn(num_items, num_users + num_items).to(device=device)
    v_features_side = torch.randn(num_items, num_side_features).to(device=device)
    u, v = torch.tensor(range(num_users), dtype=torch.int64), torch.tensor(range(num_items), dtype=torch.int64)
    rating_train = torch.randint(1, size=(num_ratings, num_users, num_items), dtype=torch.float32, device=device)

    model = GAE(num_users=num_users,
                num_items=num_items,
                num_classes=num_ratings,
                num_side_features=num_side_features,
                nb=nb,
                u_features=u_features,
                v_features=v_features,
                u_features_side=u_features_side,
                v_features_side=v_features_side,
                input_dim=num_users + num_items,
                emb_dim=emb_dim,
                hidden=hidden,
                dropout=dropout)

    output, m_hat, embeddings = model(u, v, rating_train)
    print(output.shape, m_hat.shape, embeddings["user_embedding"].shape, embeddings["item_embedding"].shape)


if __name__ == '__main__':
    _test()
