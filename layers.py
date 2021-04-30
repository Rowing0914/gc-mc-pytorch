import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, input_dim, hidden_dim, num_classes, act=torch.relu, dropout=0.7, bias=True):
        super(GraphConvolution, self).__init__()

        self.act = act
        self.dropout = nn.Dropout(dropout)
        self.u_weight = nn.Parameter(torch.FloatTensor(num_classes, input_dim, hidden_dim))
        self.v_weight = self.u_weight
        if bias:
            self.u_bias = nn.Parameter(torch.FloatTensor(hidden_dim))
            self.v_bias = self.u_bias
        else:
            self.u_bias = None
            self.v_bias = None

        for w in [self.u_weight, self.v_weight]:
            nn.init.xavier_normal_(w)

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = torch.sum(mx, 0)
        r_inv = torch.pow(rowsum, -0.5)
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        colsum = torch.sum(mx, 1)
        c_inv = torch.pow(colsum, -0.5)
        c_inv[torch.isinf(c_inv)] = 0.
        c_mat_inv = torch.diag(c_inv)

        mx = torch.matmul(mx, r_mat_inv)
        mx = torch.matmul(c_mat_inv, mx)
        return mx

    def forward(self, u_feat, v_feat, u, v, support):
        """
        Args:
            u_feat: 2D tensor; num_users x dim_user
            v_feat: 2D tensor; num_items x dim_item
            u: list of all the user ids
            v: list of all the item ids
            support: 3D tensor; num_ratings x num_users x num_items; 5 x 943 x 1682

        Returns:
            u_outputs: 2D tensor; num_users x dim_hidden
            v_outputs: 2D tensor; num_items x dim_hidden
        """

        u_feat = self.dropout(u_feat)
        v_feat = self.dropout(v_feat)

        supports_u = []
        supports_v = []
        u_weight = 0
        v_weight = 0
        for r in range(support.size(0)):  # for each rating(0 ~ 5)
            u_weight = u_weight + self.u_weight[r]
            v_weight = v_weight + self.v_weight[r]

            # multiply feature matrices with weights
            tmp_u = torch.mm(u_feat, u_weight)
            tmp_v = torch.mm(v_feat, v_weight)

            support_norm = self.normalize(support[r])
            support_norm_t = self.normalize(support[r].t())

            # then multiply with rating matrices
            supports_u.append(torch.mm(support_norm[u], tmp_v))
            supports_v.append(torch.mm(support_norm_t[v], tmp_u))

        z_u = torch.sum(torch.stack(supports_u, 0), 0)
        z_v = torch.sum(torch.stack(supports_v, 0), 0)
        if self.u_bias is not None:
            z_u = z_u + self.u_bias
            z_v = z_v + self.v_bias

        u_outputs = self.act(z_u)
        v_outputs = self.act(z_v)
        return u_outputs, v_outputs


class BilinearMixture(nn.Module):
    """
    Decoder model layer for link-prediction with ratings
    To use in combination with bipartite layers.
    """

    def __init__(self,
                 num_users,
                 num_items,
                 num_classes,
                 input_dim,
                 nb=2,
                 dropout=0.7):
        super(BilinearMixture, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.weight = nn.Parameter(torch.randn(nb, input_dim, input_dim))
        self.a = nn.Parameter(torch.randn(nb, num_classes))

        self.u_bias = nn.Parameter(torch.randn(num_users, num_classes))
        self.v_bias = nn.Parameter(torch.randn(num_items, num_classes))

        for w in [self.weight, self.a, self.u_bias, self.v_bias]:
            nn.init.xavier_normal_(w)

    def forward(self, u_hidden, v_hidden, u, v):
        """
        Args:
            u_hidden: latent 2D tensor; num_users x dim_hidden
            v_hidden: latent 2D tensor; num_items x dim_hidden
            u: list of all the user ids
            v: list of all the item ids

        Returns:
            outputs: reconstructed 3D tensor; num_ratings x num_users x num_items
            m_hat: 2D tensor; num_items x num_items
        """

        u_hidden = self.dropout(u_hidden)
        v_hidden = self.dropout(v_hidden)

        basis_outputs = []
        for weight in self.weight:
            u_w = torch.matmul(u_hidden, weight)
            x = torch.matmul(u_w, v_hidden.t())
            basis_outputs.append(x)

        basis_outputs = torch.stack(basis_outputs, 2)
        outputs = torch.matmul(basis_outputs, self.a)
        outputs = outputs + self.u_bias[u].unsqueeze(1).repeat(1, outputs.size(1), 1) \
                  + self.v_bias[v].unsqueeze(0).repeat(outputs.size(0), 1, 1)
        outputs = outputs.permute(2, 0, 1)

        softmax_out = F.softmax(outputs, 0)
        m_hat = torch.stack([(r + 1) * output for r, output in enumerate(softmax_out)], 0)
        m_hat = torch.sum(m_hat, 0)

        return outputs, m_hat


def _test_gcn():
    print("=== _test_gcn ===")
    device = "cpu"
    hidden = 64
    num_ratings = 5
    num_users, num_items = 943, 1682

    u_features = torch.randn(num_users, num_users + num_items).to(device=device)
    v_features = torch.randn(num_items, num_users + num_items).to(device=device)
    u, v = range(num_users), range(num_items)
    r_matrix = torch.randint(1, size=(num_ratings, num_users, num_items), dtype=torch.float32, device=device)

    gcn = GraphConvolution(input_dim=u_features.shape[1], hidden_dim=hidden, num_classes=num_ratings).to(device=device)
    u_out, v_out = gcn(u_features, v_features, u, v, r_matrix)
    print(u_out.shape, v_out.shape)


def _test_bilin_dec():
    print("=== _test_gcn ===")
    from random import sample
    device = "cpu"
    hidden = 64
    nb = 2
    num_ratings = 5
    num_users, num_items = 943, 1682

    u_h = torch.randn(num_users, hidden).to(device=device)
    v_h = torch.randn(num_items, hidden).to(device=device)
    u, v = sample(range(num_users), num_users), sample(range(num_items), num_items)

    model = BilinearMixture(num_users=num_users,
                            num_items=num_items,
                            num_classes=num_ratings,
                            input_dim=hidden,
                            nb=nb).to(device=device)
    out, m_hat = model(u_h, v_h, u, v)
    print(out.shape, m_hat.shape)


if __name__ == '__main__':
    _test_gcn()
    _test_bilin_dec()