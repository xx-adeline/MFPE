import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class MNLoss(nn.Module):
    def __init__(self):
        super(MNLoss, self).__init__()
        pass

    def forward(self, sim_neg, train_mn_label, mn_length):
        mn_loss = self.calculate_mn_loss(sim_neg, train_mn_label, mn_length)
        return mn_loss

    def calculate_mn_loss(self, sim_neg, train_mn_label, mn_length):
        """
            sim_neg: 2N*100
            train_label: 2N
        """
        aligned_label = np.where(train_mn_label == 1)[0]
        misaligned_label = np.where(train_mn_label == 0)[0]

        # N
        aligned_length = mn_length[aligned_label]
        misaligned_length = mn_length[misaligned_label]

        # N*100
        aligned_sim_neg = sim_neg[aligned_label, :]
        misaligned_sim_neg = sim_neg[misaligned_label, :]

        aligned_loss = []
        for i in range(aligned_sim_neg.shape[0]):
            # 1*35 --> 1*1
            aligned_loss_i = torch.mean(F.relu(- aligned_sim_neg[i, :aligned_length[i]] + 0.001)).reshape(1, 1)
            aligned_loss.append(aligned_loss_i)

        misaligned_loss = []
        for i in range(misaligned_sim_neg.shape[0]):
            # 1*35 --> 1*1
            misaligned_sim_neg_i = torch.min(misaligned_sim_neg[i, :misaligned_length[i]])
            misaligned_loss.append(F.leaky_relu(misaligned_sim_neg_i + 0.15).reshape(1, 1))

        aligned_loss = torch.cat(aligned_loss, dim=0)
        misaligned_loss = torch.cat(misaligned_loss, dim=0)

        aligned_loss = torch.sum(aligned_loss, dim=0)
        misaligned_loss = torch.sum(misaligned_loss, dim=0)

        MN_loss = aligned_loss + misaligned_loss

        return MN_loss
