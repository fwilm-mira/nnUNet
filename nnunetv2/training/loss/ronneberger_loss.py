from torch.nn.functional import one_hot
from line_profiler import profile
from torch import nn
import torch

class RonnebergerLoss(nn.Module):
    def __init__(self, ce_kwargs):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(RonnebergerLoss, self).__init__()

        # using BCEWithLogitsLoss with no reduction to preserve the channel information for each class such that it is possible to apply the class-wise and pixel-wise weight map
        self.loss_function = torch.nn.BCEWithLogitsLoss(reduction='none')

    @profile
    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        labels = target[:, 0, :, :]
        weights = target[:, 1:, :, :]

        # get labels as one-hot encoded and convert to float
        n_classes = net_output.shape[1]
        labels_one_hot = one_hot(labels.long(), n_classes)
        labels_one_hot = labels_one_hot.squeeze(1).permute(0, 3, 1, 2).float() 

        adapted_weights = torch.stack((weights[:, 0], weights[:, 1], weights[:, 1]), dim = 1)
        bce_loss = self.loss_function(net_output, labels_one_hot)
        return torch.sum(bce_loss * adapted_weights)