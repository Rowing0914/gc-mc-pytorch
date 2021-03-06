import torch
import torch.nn.functional as F


def rmse(logits, labels):
    """
    Computes the mean square error with the predictions
    computed as average predictions. Note that without the average
    this cannot be used as a loss function as it would not be differentiable.
    :param logits: predicted logits
    :param labels: ground truth labels for the ratings, 1-D array containing 0-num_classes-1 ratings
    :param class_values: rating values corresponding to each class.
    :return: mse
    """

    omg = torch.sum(labels, 0).detach()
    len_omg = len(torch.nonzero(omg))

    pred_y = logits
    y = torch.max(labels, 0)[1].float() + 1.

    se = torch.sub(y, pred_y).pow_(2)
    mse = torch.sum(torch.mul(omg, se)) / len_omg
    rmse = torch.sqrt(mse)

    return rmse


def softmax_cross_entropy(input, target):
    """ computes average softmax cross entropy """

    input = input.view(input.size(0), -1).t()
    target = target.view(target.size(0), -1).t()

    omg = torch.sum(target, 1).detach()
    len_omg = len(torch.nonzero(omg))
    target = torch.max(target, 1)[1]

    loss = F.cross_entropy(input=input, target=target, reduction='none')
    loss = torch.sum(torch.mul(omg, loss)) / len_omg

    return loss


def compute_loss(r_matrix, u, v, output, m_hat):
    """ compute the losses """
    r_mx = r_matrix.index_select(1, u).index_select(2, v)
    loss_ce = softmax_cross_entropy(output, r_mx.float())
    loss_rmse = rmse(m_hat, r_mx.float())
    return loss_ce, loss_rmse
