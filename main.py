import os
import argparse
import numpy as np
import torch.nn.parallel
import torch.optim as optim
from torch.utils.data import BatchSampler, SequentialSampler
from random import sample

from model import GAE
from data_loader import get_loader
from metrics import compute_loss


def train():
    global best_loss, best_epoch
    if args.start_epoch:
        model.load_state_dict(torch.load(os.path.join(args.model_path, 'w-%d.pkl' % (args.start_epoch))).state_dict())

    # Training
    for epoch in range(args.start_epoch, args.num_epochs):
        model.train()

        train_loss = 0.
        train_rmse = 0.
        shuffled_users = sample(range(num_users), k=num_users)
        shuffled_items = sample(range(num_items), num_items)
        for s, u in enumerate(BatchSampler(SequentialSampler(shuffled_users), batch_size=num_users, drop_last=False)):
            u = torch.from_numpy(np.array(u)).to(device)
            for t, v in enumerate(
                    BatchSampler(SequentialSampler(shuffled_items), batch_size=num_items, drop_last=False)):
                v = torch.from_numpy(np.array(v)).to(device)
                if len(torch.nonzero(torch.index_select(torch.index_select(rating_train, 1, u), 2, v))) == 0:
                    continue

                output, m_hat, _ = model(u, v, rating_train)

                optimizer.zero_grad()
                loss_ce, loss_rmse = compute_loss(rating_train, u, v, output, m_hat)
                loss_ce.backward()
                optimizer.step()

                train_loss += loss_ce.item()
                train_rmse += loss_rmse.item()

        log = 'epoch: ' + str(epoch + 1) + ' loss_ce: ' + str(train_loss / (s + 1) / (t + 1)) \
              + ' loss_rmse: ' + str(train_rmse / (s + 1) / (t + 1))
        print(log)

        if (epoch + 1) % args.val_step == 0:
            # Validation
            model.eval()
            with torch.no_grad():
                u = torch.from_numpy(np.array(range(num_users))).to(device)
                v = torch.from_numpy(np.array(range(num_items))).to(device)
                output, m_hat, _ = model(u, v, rating_val)
                loss_ce, loss_rmse = compute_loss(rating_val, u, v, output, m_hat)

            print('[val loss] : ' + str(loss_ce.item()) + ' [val rmse] : ' + str(loss_rmse.item()))
            if best_loss > loss_rmse.item():
                best_loss = loss_rmse.item()
                best_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(args.model_path, 'w-%d.pkl' % (best_epoch)))


def test():
    # Test
    model.load_state_dict(torch.load(os.path.join(args.model_path, 'w-%d.pkl' % (best_epoch))))
    model.eval()
    with torch.no_grad():
        u = torch.from_numpy(np.array(range(num_users))).to(device)
        v = torch.from_numpy(np.array(range(num_items))).to(device)
        output, m_hat, _ = model(u, v, rating_test)
        loss_ce, loss_rmse = compute_loss(rating_test, u, v, output, m_hat)

    print('[test loss] : ' + str(loss_ce.item()) + ' [test rmse] : ' + str(loss_rmse.item()))


def get_embedding():
    model.load_state_dict(torch.load(os.path.join(args.model_path, 'w-%d.pkl' % (best_epoch))))
    model.eval()
    with torch.no_grad():
        u = torch.from_numpy(np.array(range(num_users))).to(device)
        v = torch.from_numpy(np.array(range(num_items))).to(device)
        output, m_hat, embeddings = model(u, v, rating_test)
    return embeddings


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--mode', type=str, default="train", help='train / test')
    parser.add_argument('--data_type', type=str, default="ml-100k")
    # parser.add_argument('--data_type', type=str, default="ml-1m")
    parser.add_argument('--model-path', type=str, default="./models")
    parser.add_argument('--data-path', type=str, default="./data")
    parser.add_argument('--data-shuffle', type=bool, default=True)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--val-step', type=int, default=5)
    parser.add_argument('--test-epoch', type=int, default=50)
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--neg-cnt', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--dropout', type=float, default=0.7)
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')

    parser.add_argument('--emb-dim', type=int, default=32)
    parser.add_argument('--hidden', default=[64, 32, 16, 8])
    parser.add_argument('--nb', type=int, default=2)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = "cpu"

    # Load the data
    num_users, num_items, num_classes, num_side_features, num_features, \
    u_features, v_features, u_features_side, v_features_side = get_loader(args.data_type)

    u_features = torch.from_numpy(u_features).to(device).float()  # num_users x (num_users + num_items);
    v_features = torch.from_numpy(v_features).to(device).float()  # num_items x (num_users + num_items);
    u_features_side = torch.from_numpy(u_features_side).to(device)  # num_users x dim_feat;
    v_features_side = torch.from_numpy(v_features_side).to(device)  # num_items x dim_feat;

    # format: rating_cnt x num_users x num_items
    rating_train = torch.load("./data/" + args.data_type + "/rating_train.pkl").to(device)
    rating_val = torch.load("./data/" + args.data_type + "/rating_val.pkl").to(device)
    rating_test = torch.load("./data/" + args.data_type + "/rating_test.pkl").to(device)

    # Creating the architecture of the Neural Network
    model = GAE(num_users=num_users,
                num_items=num_items,
                num_classes=num_classes,
                num_side_features=num_side_features,
                nb=args.nb,
                u_features=u_features,
                v_features=v_features,
                u_features_side=u_features_side,
                v_features_side=v_features_side,
                input_dim=num_users + num_items,
                emb_dim=args.emb_dim,
                hidden=args.hidden,
                dropout=args.dropout).to(device=device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    best_epoch = 0
    best_loss = 9999.

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    train()
    test()
    embeddings = get_embedding()
    print(embeddings["user_embedding"].shape, embeddings["item_embedding"].shape)

    import pickle

    with open("./data/ml-100k/item_embedding.pkl", "wb") as file:
        pickle.dump(embeddings["user_embedding"], file, protocol=pickle.HIGHEST_PROTOCOL)

    with open("./data/ml-100k/user_embedding.pkl", "wb") as file:
        pickle.dump(embeddings["item_embedding"], file, protocol=pickle.HIGHEST_PROTOCOL)
