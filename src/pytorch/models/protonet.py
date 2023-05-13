import torch
import importlib

class PrototypicalNetwork(torch.nn.Module):
    """
    Jake Snell, Kevin Swersky, and Richard S. Zemel.
    "Prototypical networks for few-shot learning." (2017)
    https://arxiv.org/abs/1703.05175

    A prototypical network for few-shot learning.

    Prototypical network extract feature vectors for both support and query images. Then it computes the mean
    of support features for each class (called prototypes), and predict classification scores for query images
    based on their euclidean distance to the prototypes.

    Args:
        backbone_name (str): The name of the backbone model architecture.
                             Take a look at the candidate pretrained models that match the backbone names in
                             the following link: https://pytorch.org/vision/stable/models.html
        dropout_rate (float): The dropout rate for the backbone model.
        use_softmax (bool, optional): Flag indicating whether to apply softmax to the scores (default: False).
    """
    def __init__(
        self,
        backbone_name,
        dropout_rate,
        use_softmax = False
    ):
        super(PrototypicalNetwork, self).__init__()
        self.use_softmax = use_softmax
        self.backbone = self.get_backbone(backbone_name, dropout_rate)
    
    def forward(
        self,
        support_images,
        support_labels,
        query_images
    ):
        """
        Forward pass of the prototypical network.

        Args:
            support_images (torch.Tensor): The support images.
            support_labels (torch.Tensor): The labels of the support images.
            query_images (torch.Tensor): The query images.

        Returns:
            torch.Tensor: The scores for the query images.

        """
        z_support = self.backbone.forward(support_images)
        z_query = self.backbone.forward(query_images)
        
        z_proto = self.compute_prototype(z_support, support_labels)

        dists = torch.cdist(z_query, z_proto)
        scores = -dists
        return self.softmax_if_needed(scores) if self.use_softmax else scores

    @staticmethod
    def get_backbone(backbone_name, dropout_rate):
        """
        Get the backbone model for the prototypical network.

        Args:
            backbone_name (str): The name of the backbone model architecture.
            dropout_rate (float): The dropout rate for the backbone model.

        Returns:
            torch.nn.Module: The backbone model.

        """
        module_name = 'torchvision.models'
        module = importlib.import_module(module_name)
        
        try:
            pretrained = getattr(module, backbone_name)(weights='DEFAULT')
        except ValueError:
            print('{} as the selected backbone was not found'.format(backbone_name))

        trunk = []
        if dropout_rate: trunk.append(torch.nn.Dropout(dropout_rate))
        trunk.append(torch.nn.Flatten())

        try:
            classifier = pretrained.classifier
        except AttributeError:
            pretrained.fc = torch.nn.Sequential(*trunk)
        else:
            pretrained.classifier = torch.nn.Sequential(*trunk)

        # torch.compile(pretrained, backend='inductor')
        return pretrained

    @staticmethod
    def compute_prototype(support_features, support_labels):
        """
        Compute the prototypes for each class.

        Args:
            support_features (torch.Tensor): The support features.
            support_labels (torch.Tensor): The labels of the support images.

        Returns:
            torch.Tensor: The computed prototypes.

        """
        n_way = torch.unique(support_labels).shape[0]
        return torch.cat([
            support_features[torch.nonzero(support_labels == label)].mean(0)
            for label in range(n_way)
        ])

    @staticmethod
    def softmax_if_needed(output):
        """
        Applies softmax to the output scores if needed.
        
        Args:
            output (torch.Tensor): The output scores.

        Returns:
            torch.Tensor: The scores with softmax applied if needed.
        """
        return output.softmax(-1)