import networkx as nx
import numpy as np
import scipy.sparse as sparse
from scipy.sparse import csgraph, csr_matrix

G = nx.grid_2d_graph(51, 158, periodic=False)
A1 = nx.adjacency_matrix(G)
I = sparse.identity(51 * 158)
D = np.sqrt(csgraph.laplacian(A1) + A1).tocoo()
new_data = np.reciprocal(D.data, out=D.data)
M = csr_matrix((new_data, (D.row, D.col)), shape=D.shape)

np.save("data/porto/adj.npy", (A1 + I).tocoo())
np.save("data/porto/d_norm.npy", M.tocoo())
