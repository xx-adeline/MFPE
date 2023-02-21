# -*- coding: utf-8 -*-
from torch import nn
from model.text_feature_extract import TextExtract_lstm, TextExtract_Bert_lstm
from torchvision import models
import torch
from torch.nn import init


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1)
        init.constant_(m.bias.data, 0)


class conv(nn.Module):

    def __init__(self, input_dim, output_dim, relu=False, BN=False):
        super(conv, self).__init__()

        block = []
        block += [nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False)]

        if BN:
            block += [nn.BatchNorm2d(output_dim)]
        if relu:
            block += [nn.LeakyReLU(0.25, inplace=True)]

        self.block = nn.Sequential(*block)
        self.block.apply(weights_init_kaiming)

    def forward(self, x):
        x = self.block(x)
        x = x.squeeze(3).squeeze(2)
        return x


class TextImgPersonReidNet(nn.Module):

    def __init__(self, opt):
        super(TextImgPersonReidNet, self).__init__()

        self.opt = opt
        resnet50 = models.resnet50(pretrained=True)
        if opt.stride:
            resnet50.layer4[0].downsample[0].stride = (1, 1)
            resnet50.layer4[0].conv2.stride = (1, 1)
        self.ImageExtract = nn.Sequential(*(list(resnet50.children())[:-2]))

        if self.opt.wordtype == 'bert':
            self.TextExtract = TextExtract_Bert_lstm(opt)
        elif self.opt.wordtype == 'lstm':
            self.TextExtract = TextExtract_lstm(opt)

        self.global_avgpool = nn.AdaptiveMaxPool2d((1, 1))
        self.local_avgpool = nn.AdaptiveMaxPool2d((opt.part, 1))

        conv_local = []
        for i in range(opt.part):
            conv_local.append(conv(2048, opt.feature_length))
        self.conv_local = nn.Sequential(*conv_local)

        self.conv_global = conv(2048, opt.feature_length)

        self.conv_word_classifier = nn.Sequential(
            nn.Conv2d(2048, 6, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # MN conv 降维
        self.conv_img_for_mn = conv(2048, 256)
        self.conv_txt_for_mn = conv(2048, 256)

    def forward(self, image, caption_id, caption_mask, text_length):

        img_global, img_local, img_mn = self.img_embedding(image)
        txt_global, txt_local, txt_mn = self.txt_embedding(caption_id, caption_mask, text_length)

        return img_global, img_local, txt_global, txt_local, img_mn, txt_mn

    def img_embedding(self, image):

        image_feature = self.ImageExtract(image)
        
        image_feature_global = self.global_avgpool(image_feature)
        image_global = self.conv_global(image_feature_global).unsqueeze(2)

        image_feature_local = self.local_avgpool(image_feature)

        image_feature_mn = self.conv_img_for_mn(image_feature_local)

        image_local = []
        for i in range(self.opt.part):
            image_feature_local_i = image_feature_local[:, :, i, :]
            image_feature_local_i = image_feature_local_i.unsqueeze(2)
            image_embedding_local_i = self.conv_local[i](image_feature_local_i).unsqueeze(2)
            image_local.append(image_embedding_local_i)

        image_local = torch.cat(image_local, 2)

        return image_global, image_local, image_feature_mn

    def txt_embedding(self, caption_id, caption_mask, text_length):
        if self.opt.wordtype == 'bert':
            text_feature = self.TextExtract(caption_id, caption_mask)
        elif self.opt.wordtype == 'lstm':
            text_feature = self.TextExtract(caption_id, text_length)

        text_feature_mn = self.conv_txt_for_mn(text_feature)

        text_global, _ = torch.max(text_feature, dim=2, keepdim=True)
        text_global = self.conv_global(text_global).unsqueeze(2)

        text_feature_local = []
        for text_i in range(text_feature.size(0)):
            text_feature_local_i = text_feature[text_i, :, :text_length[text_i]].unsqueeze(0)

            word_classifier_score_i = self.conv_word_classifier(text_feature_local_i)

            word_classifier_score_i = word_classifier_score_i.permute(0, 3, 2, 1).contiguous()
            text_feature_local_i = text_feature_local_i.repeat(1, 1, 1, 6).contiguous()

            text_feature_local_i = text_feature_local_i * word_classifier_score_i

            text_feature_local_i, _ = torch.max(text_feature_local_i, dim=2)

            text_feature_local.append(text_feature_local_i)

        text_feature_local = torch.cat(text_feature_local, dim=0)

        text_local = []
        for p in range(self.opt.part):
            text_feature_local_conv_p = text_feature_local[:, :, p].unsqueeze(2).unsqueeze(2)
            text_feature_local_conv_p = self.conv_local[p](text_feature_local_conv_p).unsqueeze(2)
            text_local.append(text_feature_local_conv_p)
        text_local = torch.cat(text_local, dim=2)

        return text_global, text_local, text_feature_mn

