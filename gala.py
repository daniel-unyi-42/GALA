import tensorflow as tf
from scipy import sparse

def extract_data(filename):
  import numpy as np
  # load data
  data = np.load(filename)
  adj = sparse.csr_matrix((data['adj_data'], data['adj_indices'], data['adj_indptr']), \
                                       shape=data['adj_shape'], dtype='float32')
  attr = sparse.csr_matrix((data['attr_data'], data['attr_indices'], data['attr_indptr']), \
                                     shape=data['attr_shape'], dtype='float32')
  N = len(data['labels'])
  N_classes = len(set(data['labels']))
  labels = sparse.csr_matrix((np.ones(N), data['labels'], np.arange(N + 1)), \
                             shape=(N, N_classes), dtype='float32')
  # symmetrize
  adj = adj.tolil()
  rows, cols = adj.nonzero()
  adj[cols, rows] = adj[rows, cols]
  # shuffle
  permutation = list(range(N))
  #np.random.shuffle(permutation)
  adj = adj.tocoo()
  for i in range(adj.nnz):
    adj.row[i] = permutation[adj.row[i]]
    adj.col[i] = permutation[adj.col[i]]
  attr = attr.tocoo()
  for i in range(attr.nnz):
    attr.row[i] = permutation[attr.row[i]]
  labels = labels.tocoo()
  for i in range(labels.nnz):
    labels.row[i] = permutation[labels.row[i]]
  # result
  return adj.tocsr(), attr.todense(), labels.todense()

filename = 'cora.npz'

# hyperparameters
hidden = 64
latent = 32
learning_rate = 0.01
epochs = 1000

adjacency, features, labels = extract_data(filename)

features = tf.math.l2_normalize(tf.constant(features), axis=1)
labels = tf.math.l2_normalize(tf.constant(labels), axis=1)
N, F = features.shape

# renormalization
smoother = sparse.eye(N, dtype='float32') + adjacency
degrees = sparse.eye(N, dtype='float32') + sparse.diags(adjacency.sum(axis=1).A1)
norm = degrees.power(-0.5)
smoother = norm * smoother * norm  # D**-1/2 * W * D**-1/2

# sparse.csr_matrix -> tf.SparseTensor
indices = list(zip(smoother.tocoo().row, smoother.tocoo().col))
values = smoother.data
dense_shape = smoother.shape
smoother = tf.SparseTensor(indices, values, dense_shape)

# renormalization
sharpener = 2.0 * sparse.eye(N, dtype='float32') - adjacency
degrees = 2.0 * sparse.eye(N, dtype='float32') + sparse.diags(adjacency.sum(axis=1).A1)
norm = degrees.power(-0.5)
sharpener = norm * sharpener * norm  # D**-1/2 * W * D**-1/2

# sparse.csr_matrix -> tf.SparseTensor
indices = list(zip(sharpener.tocoo().row, sharpener.tocoo().col))
values = sharpener.data
dense_shape = sharpener.shape
sharpener = tf.SparseTensor(indices, values, dense_shape)

# layer classes

class ReLU_layer:

  def __call__(self, tensor):
    return tf.nn.relu(tensor)

class GC_layer:

  def __init__(self, indim, outdim):
    initial_value = tf.initializers.he_normal()((indim, outdim,))
    self.weight = tf.Variable(initial_value=initial_value, trainable=True)

  def __call__(self, tensor, support):
    return tf.sparse.sparse_dense_matmul(support, tf.linalg.matmul(tensor, self.weight))

# model class

class Model:

  def __init__(self, size_tuple, optimizer):
    self.build(size_tuple)
    self.optimizer = optimizer

  def build(self, size_tuple):
    F, hidden, latent = size_tuple
    self.encoder0 = GC_layer(F, hidden)
    self.encoder1 = GC_layer(hidden, latent)
    self.decoder1 = GC_layer(latent, hidden)
    self.decoder0 = GC_layer(hidden, F)

  def predict(self, tensor, smoother, sharpener):
    tensor = self.encode(tensor, smoother)
    tensor = ReLU_layer()(tensor)
    tensor = self.decode(tensor, sharpener)
    return tensor

  def encode(self, tensor, support):
    tensor = self.encoder0(tensor, support)
    tensor = ReLU_layer()(tensor)
    tensor = self.encoder1(tensor, support)
    return tensor

  def decode(self, tensor, support):
    tensor = self.decoder1(tensor, support)
    tensor = ReLU_layer()(tensor)
    tensor = self.decoder0(tensor, support)
    return tensor

  def train(self, features, smoother, sharpener, epochs):
    sources = (self.encoder0.weight, self.encoder1.weight, self.decoder1.weight, self.decoder0.weight)
    for epoch in range(epochs):
      with tf.GradientTape() as tape:
        predictions = self.predict(features, smoother, sharpener)
        loss_ = self.loss(predictions, features)
      grads = tape.gradient(loss_, sources)
      self.optimizer.apply_gradients(zip(grads, sources))
      print('%d epoch: loss = %f' % (epoch, loss_.numpy()))

  def loss(self, predictions, features):
    return tf.reduce_sum(tf.square(features - predictions))/2/features.shape[0]

size_tuple = (F, hidden, latent)
optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

model = Model(size_tuple, optimizer)

model.train(features, smoother, sharpener, epochs)

embeddings = model.encode(features, smoother)




import umap
reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(embeddings.numpy())

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(12, 12))
color = tf.argmax(labels, axis=1).numpy()
plt.scatter(embedding[:, 0], embedding[:, 1], c=color, cmap="Spectral", s=10)
plt.setp(ax, xticks=[], yticks=[])
plt.title("UMAP", fontsize=18)
plt.show()
