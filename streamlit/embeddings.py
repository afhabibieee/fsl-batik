import torch
from torchvision import models
import torchvision.transforms as T
import configs

class Embedding(torch.nn.Module):
    def __init__(self, backbone_name):
        super(Embedding, self).__init__()
        self.backbone = getattr(models, backbone_name)(weights = None)

        try:
            classifier = self.backbone.classifier
        except AttributeError:
            self.backbone.fc = torch.nn.Sequential(torch.nn.Flatten())
        else:
            self.backbone.classifier = torch.nn.Sequential(torch.nn.Flatten())
    
    def forward(self, inputs):
        x = self.backbone(inputs)
        return x

class EmbeddingPostProcess(torch.nn.Module):
    def __init__(self, embedding):
        super(EmbeddingPostProcess, self).__init__()
        self.embedding = embedding
        self.scale = lambda x: x.div(225)
        self.normalize = T.Normalize(configs.IMAGENET_MEAN, configs.IMAGENET_STD)

    def forward(self, inputs):
        scaled = self.scale(inputs)
        normalized = self.normalize(scaled)
        features = self.embedding.forward(normalized)
        return features