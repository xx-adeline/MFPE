# -*- coding: utf-8 -*-

import torch
import numpy as np
from math import ceil


def calculate_sim_neg_for_train(image_mn, text_mn, caption_length, labels):
    # N*N*100*6
    similarity = calculate_sim_neg(image_mn, text_mn)
    # N*N*100
    similarity, _ = torch.max(similarity, dim=3)
    # 2N*2        2N
    train_data_index, train_mn_label = get_data_for_train(labels)
    # 2N*100
    sim_neg = similarity[train_data_index[:, 0], train_data_index[:, 1], :]
    # 2N
    caption_length = caption_length[train_data_index[:, 1]]
    # N*N*100
    similarity[similarity > 0] = 0
    # N'*1
    similarity = torch.sum(similarity, 2)
    return sim_neg, train_mn_label, caption_length, similarity


def calculate_sim_neg_for_test(image_mn, text_mn, shard_size=64):
    """
        image_mn: N*512*6
        text_mn: M*512*100
    """
    n_img = image_mn.shape[0]
    n_txt = text_mn.shape[0]

    n_img_shard = ceil(n_img / shard_size)
    n_txt_shard = ceil(n_txt / shard_size)
    # N*M
    sim_neg = np.zeros((n_img, n_txt))

    for i in range(n_img_shard):
        img_start, img_end = shard_size * i, min(shard_size * (i + 1), n_img)
        for j in range(n_txt_shard):
            txt_start, txt_end = shard_size * j, min(shard_size * (j + 1), n_txt)

            # N'*M'*100*6
            sim_neg_i = calculate_sim_neg(image_mn[img_start:img_end, :, :], text_mn[txt_start:txt_end, :, :])

            # N'*M'*100
            sim_neg_i, _ = torch.max(sim_neg_i, dim=3)
            sim_neg_i = sim_neg_i.data.cpu().numpy()
            sim_neg_i[sim_neg_i > 0] = 0
            # N'*M'
            sim_neg_i = np.sum(sim_neg_i, axis=2)
            sim_neg[img_start:img_end, txt_start:txt_end] = sim_neg_i
    # N*M
    return sim_neg


def get_data_for_train(labels):
    labels = labels.cpu().data.numpy()
    N = labels.shape[0]

    # sample negative data
    train_data_index = []
    for idx, label in enumerate(labels):
        # remove same id
        negative = np.where(labels != label)[0]
        negetive = np.random.choice(negative)
        train_data_index.append([idx, negetive])
    # sample postive data
    for idx in range(len(labels)):
        train_data_index.append([idx, idx])
    train_data_index = np.vstack(train_data_index)
    train_label = np.vstack((np.zeros((N, 1)), np.ones((N, 1))))

    return train_data_index, train_label


def calculate_sim_neg(image_mn, text_mn):
    """
        image_mn: N*512*6
        text_mn: M*512*100
    """
    image_mn = image_mn / (image_mn.norm(dim=1, keepdim=True) + 1e-8)
    text_mn = text_mn / (text_mn.norm(dim=1, keepdim=True) + 1e-8)

    # image N*6*512
    image_mn = image_mn.permute(0, 2, 1)

    # image N*1*6*512
    image_mn = image_mn.unsqueeze(1)
    # text 1*M*512*100
    text_mn = text_mn.unsqueeze(0)
    # N*M*100*6
    similarity = torch.matmul(image_mn, text_mn).permute(0, 1, 3, 2).contiguous()

    return similarity
