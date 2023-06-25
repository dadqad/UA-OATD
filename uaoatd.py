import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from config import args


def array2sparse(A):
    values = A.data
    indices = np.vstack((A.row, A.col))
    i = torch.LongTensor(indices).to(args.device)
    v = torch.FloatTensor(values).to(args.device)
    A = torch.sparse.FloatTensor(i, v, torch.Size(A.shape))
    return A


class ATT(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, dropout_rate, device):
        super(ATT, self).__init__()
        self.device = device
        self.gru = nn.GRU(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers,
                          batch_first=True, dropout=dropout_rate)

        self.w_omega = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.u_omega = nn.Parameter(torch.Tensor(hidden_size, 1))

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def getMask(self, seq_lengths):
        max_len = int(seq_lengths.max())
        mask = torch.ones((seq_lengths.size()[0], max_len)).to(self.device)
        for i, l in enumerate(seq_lengths):
            if l < max_len:
                mask[i, l:] = 0
        return mask

    def forward(self, packed_input, hidden):
        packed_output, hidden = self.gru(packed_input, hidden)  # output, h
        outputs, seq_lengths = pad_packed_sequence(packed_output, batch_first=True)

        mask = self.getMask(seq_lengths)
        u = torch.tanh(torch.matmul(outputs, self.w_omega))
        att = torch.matmul(u, self.u_omega).squeeze()
        att = att.masked_fill(mask == 0, -1e10)
        att_score = F.softmax(att, dim=1).unsqueeze(2)
        scored_outputs = outputs * att_score
        out = torch.sum(scored_outputs, dim=1)
        return scored_outputs, out.unsqueeze(0)


class UA_OATD(nn.Module):
    def __init__(self, token_size, embedding_size, hidden_size):
        super(UA_OATD, self).__init__()

        self.pi_prior = nn.Parameter(torch.ones(args.n_cluster) / args.n_cluster)
        self.mu_prior = nn.Parameter(torch.zeros(args.n_cluster, hidden_size))
        self.log_var_prior = nn.Parameter(torch.randn(args.n_cluster, hidden_size))

        self.embedding = nn.Embedding(token_size, embedding_size)
        self.encoder_att = ATT(embedding_size, hidden_size, 1, 0, args.device)
        self.decoder = nn.GRU(embedding_size, hidden_size, 1, batch_first=True)

        self.fc_mu = nn.Linear(hidden_size, hidden_size)
        self.fc_logvar = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

        self.fc_out = nn.Linear(hidden_size, token_size)

        self.nodes = torch.arange(token_size, dtype=torch.long).to(args.device)
        self.adj = np.load("data/{}/adj.npy".format(args.dataset), allow_pickle=True)[()]
        self.d_norm = np.load("data/{}/d_norm.npy".format(args.dataset), allow_pickle=True)[()]

        self.V = nn.Parameter(torch.Tensor(embedding_size, embedding_size))
        nn.init.uniform_(self.V, -0.02, 0.02)

    def Norm_A(self, A, D):
        return D.mm(A).mm(D)

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var * 0.5)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, trajs, lengths, batch_size, mode, c):
        adj = array2sparse(self.adj)
        d_norm = array2sparse(self.d_norm)
        H = self.Norm_A(adj, d_norm)

        e_nodes = self.embedding(self.nodes)
        e_nodes = H.mm(e_nodes).mm(self.V)
        e_input = torch.index_select(e_nodes, 0, trajs.reshape(1, -1).squeeze().to(args.device)). \
            reshape(batch_size, -1, args.embedding_size)
        d_input = torch.cat(
            (torch.zeros(batch_size, 1, args.embedding_size, dtype=torch.long).to(args.device), e_input[:, :-1, :]),
            dim=1)

        decoder_inputs = pack_padded_sequence(d_input, lengths, batch_first=True, enforce_sorted=False)

        if mode == "pretrain" or "train":
            encoder_inputs = pack_padded_sequence(e_input, lengths, batch_first=True, enforce_sorted=False)
            _, encoder_final_state = self.encoder_att(encoder_inputs, None)

            mu = self.fc_mu(encoder_final_state)
            logvar = self.fc_logvar(encoder_final_state)
            z = self.reparameterize(mu, logvar)
            decoder_outputs, _ = self.decoder(decoder_inputs, z)
            decoder_outputs, _ = pad_packed_sequence(decoder_outputs, batch_first=True)

        elif mode == "test":
            mu = torch.stack([self.mu_prior] * batch_size, dim=1)[c: c + 1]
            decoder_outputs, _ = self.decoder(decoder_inputs, mu)
            decoder_outputs, _ = pad_packed_sequence(decoder_outputs, batch_first=True)
            logvar, z = None, None

        output = self.fc_out(self.layer_norm(decoder_outputs))

        return output, mu, logvar, z
