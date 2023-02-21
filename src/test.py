# -*- coding: utf-8 -*-
from option.options import options, config
from data.dataloader import get_dataloader
import torch
from model.model import TextImgPersonReidNet
import os
from test_during_train import test


def main(opt):
    opt.device = torch.device('cuda:{}'.format(opt.GPU_id))
    opt.save_path = './checkpoints/{}/'.format(opt.dataset) + opt.model_name
    config(opt)

    opt.mode = 'test'
    test_img_dataloader, test_txt_dataloader = get_dataloader(opt)
    network = TextImgPersonReidNet(opt).to(opt.device)

    if opt.resume == 'restart':
        check_point = torch.load(os.path.join(opt.save_path, 'model', opt.resume))
        network.load_state_dict(check_point["network"])

    network.eval()
    test(opt, 0, network, test_img_dataloader, test_txt_dataloader, 0)
    network.train()


if __name__ == '__main__':
    opt = options().opt
    main(opt)





