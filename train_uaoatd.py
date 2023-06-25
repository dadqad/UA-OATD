import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.mixture import GaussianMixture
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader

from config import args
from uaoatd import UA_OATD
from utils import auc_score, make_mask, make_len_mask


def collate_fn(batch):
    max_len = max(len(x) for x in batch)
    seq_lengths = list(map(len, batch))
    batch_trajs = [x + [0] * (max_len - len(x)) for x in batch]
    return torch.LongTensor(batch_trajs), seq_lengths


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_torch(args.seed)


class MyDataset(Dataset):
    def __init__(self, seqs):
        self.seqs = seqs

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):
        data_seqs = self.seqs[index]
        return data_seqs


class train_uaoatd:
    def __init__(self, token_size, train_loader, outliers_loader, labels):
        self.model = UA_OATD(token_size, args.embedding_size, args.hidden_size).to(args.device)

        self.crit = nn.CrossEntropyLoss()
        self.detec = nn.CrossEntropyLoss(reduction='none')

        self.pretrain_optimizer = optim.AdamW(self.model.parameters(), lr=args.pretrain_lr)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)

        self.lr_pretrain = StepLR(self.pretrain_optimizer, step_size=3, gamma=0.9)
        self.lr = StepLR(self.optimizer, step_size=3, gamma=0.8)

        self.train_loader = train_loader
        self.outliers_loader = outliers_loader

        self.pretrain_path = "models/pretrain_{}_{}.pth".format(args.n_cluster, args.dataset)
        self.model_path = "models/model_{}_{}.pth".format(args.n_cluster, args.dataset)
        self.labels = labels

    def pretrain(self, epoch):
        self.model.train()
        epo_loss = 0
        for batch in self.train_loader:
            trajs, seq_lengths = batch
            batch_size = len(trajs)
            self.pretrain_optimizer.zero_grad()
            output, _, _, _, = self.model(trajs, seq_lengths, batch_size, "pretrain", -1)
            loss = self.crit(output.reshape(-1, output.shape[-1]), trajs.to(args.device).reshape(-1))
            loss.backward()
            self.pretrain_optimizer.step()
            epo_loss += loss.item()
        self.lr_pretrain.step()
        print("Epoch : {}, Loss: {}".format(epoch + 1, epo_loss / len(self.train_loader)))
        torch.save(self.model.state_dict(), self.pretrain_path)

    def train_gmm(self):
        z = torch.Tensor([]).to(args.device)
        self.model.load_state_dict(torch.load(self.pretrain_path))
        self.model.eval()

        with torch.no_grad():
            for batch in self.train_loader:
                trajs, seq_lengths = batch
                batch_size = len(trajs)
                _, _, _, hidden = self.model(trajs, seq_lengths, batch_size, "pretrain", -1)
                z = torch.cat((z, hidden.squeeze(0)), dim=0)

        print('...Fiting Gaussian Mixture Model...')
        self.gmm = GaussianMixture(n_components=args.n_cluster, covariance_type="diag", n_init=3)
        self.gmm.fit(z.cpu().numpy())

    def save_weights(self):
        print('...Saving Weights...')
        torch.save(torch.from_numpy(self.gmm.weights_).float(),
                   "models/pi_prior_{}_{}.pth".format(args.n_cluster, args.dataset))
        torch.save(torch.from_numpy(self.gmm.means_).float(),
                   "models/mu_prior_{}_{}.pth".format(args.n_cluster, args.dataset))
        torch.save(torch.log(torch.from_numpy(self.gmm.covariances_)).float(),
                   "models/log_var_prior_{}_{}.pth".format(args.n_cluster, args.dataset))

    def train(self, epoch):
        self.model.train()
        total_loss = 0
        for batch in self.train_loader:
            trajs, seq_lengths = batch
            batch_size = len(trajs)
            self.optimizer.zero_grad()
            output, mu, log_var, z = self.model(trajs, seq_lengths, batch_size, "train", -1)
            loss = self.Loss(output, trajs.to(args.device), mu.squeeze(0), log_var.squeeze(0), z.squeeze(0))
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        self.lr.step()
        print('Epoch : {}, Loss: {}'.format(epoch + 1, total_loss / len(self.train_loader)))
        torch.save(self.model.state_dict(), self.model_path)

    def detection(self):

        self.model.eval()
        all_likelihood = []
        with torch.no_grad():
            for c in range(args.n_cluster):
                c_likelihood = []
                for batch in self.outliers_loader:
                    trajs, seq_lengths = batch
                    batch_size = len(trajs)
                    mask = make_mask(make_len_mask(trajs)).to(args.device)
                    outputs, _, _, _ = self.model(trajs, seq_lengths, batch_size, "test", -1)

                    likelihood = - self.detec(outputs.reshape(-1, outputs.shape[-1]),
                                              trajs.to(args.device).reshape(-1))
                    likelihood = torch.exp(
                        torch.sum(mask * (likelihood.reshape(batch_size, -1)), dim=-1) / torch.sum(mask, 1))

                    c_likelihood.append(likelihood)
                all_likelihood.append(torch.cat(c_likelihood).unsqueeze(0))

        all_likelihood = torch.cat(all_likelihood, dim=0)
        likelihood, _ = torch.max(all_likelihood, dim=0)
        pr_auc = auc_score(self.labels, (1 - likelihood).cpu().detach().numpy())
        print("PR_AUC: {}".format(pr_auc))
        return pr_auc

    def gaussian_pdf_log(self, x, mu, log_var):
        return -0.5 * (torch.sum(np.log(np.pi * 2) + log_var + (x - mu).pow(2) / torch.exp(log_var), 1))

    def gaussian_pdfs_log(self, x, mus, log_vars):
        G = []
        for c in range(args.n_cluster):
            G.append(self.gaussian_pdf_log(x, mus[c:c + 1, :], log_vars[c:c + 1, :]).view(-1, 1))
        return torch.cat(G, 1)

    def Loss(self, x_hat, targets, z_mu, z_sigma2_log, z):
        pi = self.model.pi_prior
        log_sigma2_c = self.model.log_var_prior
        mu_c = self.model.mu_prior
        reconstruction_loss = self.crit(x_hat.reshape(-1, x_hat.shape[-1]), targets.reshape(-1))
        gaussian_loss = torch.mean(torch.mean(
            self.gaussian_pdf_log(z, z_mu, z_sigma2_log).unsqueeze(1) - self.gaussian_pdfs_log(z, mu_c, log_sigma2_c),
            dim=1), dim=-1)
        logits = -torch.mean(torch.square(z.unsqueeze(1) - mu_c.unsqueeze(0)) / torch.exp(log_sigma2_c.unsqueeze(0)),
                             axis=-1)
        log_q = F.log_softmax(logits, dim=-1)
        category_loss = - torch.mean(torch.sum(pi * log_q, dim=-1))
        loss = reconstruction_loss + gaussian_loss.mean() / args.hidden_size - category_loss / args.hidden_size
        return loss


if __name__ == '__main__':
    print("================================================")
    print("Embedding Size: {}".format(args.embedding_size))
    print("Hidden Size: {}".format(args.hidden_size))
    print("Batch Size: {}".format(args.batch_size))
    print("Device:", args.device)
    print("Dataset:", args.dataset)
    token_size = 51 * 158

    train_trajs = np.load('./data/{}/train_data.npy'.format(args.dataset), allow_pickle=True)
    test_trajs = np.load('./data/{}/outliers_data.npy'.format(args.dataset), allow_pickle=True)
    outliers_idx = np.load("./data/{}/outliers_idx.npy".format(args.dataset), allow_pickle=True)

    train_data = MyDataset(train_trajs)
    test_data = MyDataset(test_trajs)

    labels = np.zeros(len(test_trajs))
    for i in outliers_idx:
        labels[i] = 1

    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
                              pin_memory=True)
    outliers_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False,
                                 collate_fn=collate_fn, pin_memory=True)

    Train_uaoatd = train_uaoatd(token_size, train_loader, outliers_loader, labels)

    print("---------------Pretrain---------------")
    for epoch in range(args.pretrain_epochs):
        Train_uaoatd.pretrain(epoch)

    Train_uaoatd.train_gmm()
    Train_uaoatd.save_weights()

    print("---------------Training---------------")
    Train_uaoatd.model.load_state_dict(torch.load(Train_uaoatd.pretrain_path))
    Train_uaoatd.model.pi_prior.data = torch.load("models/pi_prior_{}_{}.pth".format(args.n_cluster, args.dataset)).to(
        args.device)
    Train_uaoatd.model.mu_prior.data = torch.load("models/mu_prior_{}_{}.pth".format(args.n_cluster, args.dataset)).to(
        args.device)
    Train_uaoatd.model.log_var_prior.data = torch.load(
        "models/log_var_prior_{}_{}.pth".format(args.n_cluster, args.dataset)).to(args.device)

    for epoch in range(args.epochs):
        Train_uaoatd.train(epoch)

    print("---------------Testing---------------")
    Train_uaoatd.model.load_state_dict(torch.load(Train_uaoatd.model_path))
    with torch.no_grad():
        Train_uaoatd.detection()
