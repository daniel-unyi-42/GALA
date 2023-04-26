import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.utils import negative_sampling
from torch_geometric.datasets import Planetoid
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import gcn_deconv


writer = SummaryWriter(log_dir='runs/GALA')

EPS = 1e-15

lr = 0.0001
epochs = 1000


class Encoder(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.conv1 = pyg_nn.GCNConv(num_features, 1600, cached=True)
        self.conv2 = pyg_nn.GCNConv(1600, 400, cached=True)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


class Decoder(nn.Module):
      def __init__(self, num_features):
          super().__init__()
          self.conv1 = gcn_deconv.GCNDeconv(400, 1600, improved=True, cached=True)
          self.conv2 = gcn_deconv.GCNDeconv(1600, num_features, improved=True, cached=True)

      def forward(self, x, edge_index):
          x = F.relu(self.conv1(x, edge_index))
          x = self.conv2(x, edge_index)
          return x


class GAE(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.encoder = Encoder(num_features)
        self.decoder = Decoder(num_features)
        GAE.reset_parameters(self)

    def reset_parameters(self):
        pyg_nn.inits.reset(self.encoder)
        pyg_nn.inits.reset(self.decoder)

    def forward(self, x, edge_index):
        return self.encoder(x, edge_index)

    def encode(self, x, edge_index):
        return self.encoder(x, edge_index)

    def decode(self, x, edge_index):
        return self.decoder(x, edge_index)
    
    def adjrec(self, z, edge_index, sigmoid=True):
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def adj_loss(self, z, pos_edge_index, neg_edge_index = None):
        pos_loss = -torch.log(self.adjrec(z, pos_edge_index) + EPS).mean()
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 - self.adjrec(z, neg_edge_index, sigmoid=True) + EPS).mean()
        return pos_loss + neg_loss
    
    def x_loss(self, x, x_hat):
        return torch.square(torch.norm(x - x_hat)) / 2 / x.shape[0]

    def loss(self, x, edge_index):
        z = self.encode(x, edge_index)
        x_hat = self.decode(z, edge_index)
        return self.x_loss(x, x_hat) + self.adj_loss(z, edge_index)
    
    # test node clustering
    def test_NC(self, z, y):
        kmeans = KMeans(n_clusters=7, n_init=20)
        y_pred = kmeans.fit_predict(z.detach().cpu().numpy())
        y_true = y.detach().cpu().numpy()
        return normalized_mutual_info_score(y_true, y_pred), \
            adjusted_rand_score(y_true, y_pred)


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

dataset = Planetoid(root='', name='Cora')

data = dataset[0].to(device)

model = GAE(dataset.num_features).to(device)

print(model)

print('%d parameters' % sum(p.numel() for p in model.parameters() if p.requires_grad))

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    loss_value = model.loss(data.x, data.edge_index)
    loss_value.backward()
    optimizer.step()
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)
        nmi, ari = model.test_NC(z, data.y)
        writer.add_scalar("loss", loss_value, global_step=epoch)
        writer.add_scalar("nmi", nmi, global_step=epoch)
        writer.add_scalar("ari", ari, global_step=epoch)
        print(f'Epoch: {epoch:03d}, Loss: {loss_value.float():.4f}, NMI: {nmi:.4f}', f'ARI: {ari:.4f}')

writer.flush()
writer.close()
