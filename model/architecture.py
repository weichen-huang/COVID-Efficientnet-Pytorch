from torch import nn
from .efficientnet import EfficientNet


class COVIDEfficientnet(nn.Module):
    """COVIDEfficientnet model class."""
    def __init__(self, n_classes):
        """
                Defines Efficientnet model.

                :param n_classes: NUmber of classes in the dataset
                :return:
                """
        super(COVIDEfficientnet, self).__init__()
        self.n_classes = n_classes
        self.efficientnet = EfficientNet(7, pretrained=True)

    def forward(self, input):
        """
                Runs input through model.

                :param input: input tensor
                :return:
                """
        return self.efficientnet(input)

    def probability(self, logits):
        """
                Computes a Softmax of given logits

                :param logits: prediction tensor
                :return:
                """
        return nn.functional.softmax(logits, dim=-1)
