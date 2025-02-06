import torch
from torch import nn
import torch.nn.functional as F


class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.classifier = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.classifier.weight.data.uniform_(-initrange, initrange)
        self.classifier.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        x = self.fc(embedded)
        x = F.leaky_relu(x)
        x = self.classifier(x)
        return x
    
    def linear_foward(self, x):
        embedding_weight = self.embedding.weight
        embedded = x.matmul(embedding_weight)
        x = self.fc(embedded)
        x = F.leaky_relu(x)
        x = self.classifier(x)
        return x
