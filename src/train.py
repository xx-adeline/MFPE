# -*- coding: utf-8 -*-
from option.options import options, config
from data.dataloader import get_dataloader
import torch
from model.model import TextImgPersonReidNet
from loss.Id_loss import Id_Loss
from loss.RankingLoss import RankingLoss
from torch import optim
import logging
import os, random
import numpy as np
from test_during_train import test
from torch.autograd import Variable
from model.single_flow import calculate_sim_neg_for_train
from loss.MN_loss import MNLoss


logger = logging.getLogger()
logger.setLevel(logging.INFO)


def save_checkpoint(state, opt):

    filename = os.path.join(opt.save_path, 'model/best.pth.tar')
    torch.save(state, filename)

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    
def train(opt):
    opt.device = torch.device('cuda:{}'.format(opt.GPU_id))

    opt.save_path = './checkpoints/{}/'.format(opt.dataset) + opt.model_name

    config(opt)
    train_dataloader = get_dataloader(opt)
    opt.mode = 'test'
    test_img_dataloader, test_txt_dataloader = get_dataloader(opt)
    opt.mode = 'train'

    id_loss_fun_global = Id_Loss(opt, 1, opt.feature_length).to(opt.device)
    id_loss_fun_local = Id_Loss(opt, opt.part, opt.feature_length).to(opt.device)
    ranking_loss_fun = RankingLoss(opt)
    mn_loss_fun = MNLoss()

    network = TextImgPersonReidNet(opt).to(opt.device)    
    if opt.resume == 'restart':
        epoch_start = 1
    else:
        check_point = torch.load(os.path.join(opt.save_path, 'model', opt.resume))
        network.load_state_dict(check_point["network"])
        id_loss_fun_global.load_state_dict(check_point['id_global'])
        id_loss_fun_local.load_state_dict(check_point['id_local'])
        epoch_start = check_point["epoch"] + 1

    cnn_params = list(map(id, network.ImageExtract.parameters()))
    other_params = filter(lambda p: id(p) not in cnn_params, network.parameters())
    other_params = list(other_params)
    other_params.extend(list(id_loss_fun_global.parameters()))
    other_params.extend(list(id_loss_fun_local.parameters()))
    param_groups = [{'params': other_params, 'lr': opt.lr},
                    {'params': network.ImageExtract.parameters(), 'lr': opt.lr * 0.1}]

    optimizer = optim.Adam(param_groups, betas=(opt.adam_alpha, opt.adam_beta))

    test_best = 0
    test_history = 0

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, opt.epoch_decay)

    for epoch in range(epoch_start, opt.epoch):

        id_loss_sum = 0
        ranking_loss_sum = 0
        mn_loss_sum = 0

        for param in optimizer.param_groups:
            logging.info('lr:{}'.format(param['lr']))

        for times, [image, label, caption_code, caption_length, caption_mask] in enumerate(
                train_dataloader):

            image = Variable(image.to(opt.device))
            label = Variable(label.to(opt.device))
            caption_code = Variable(caption_code.to(opt.device).long())
            caption_mask = Variable(caption_mask.to(opt.device))
            caption_length = caption_length.to(opt.device)

            img_global, img_local, txt_global, txt_local, img_mn, txt_mn = network(image, caption_code, caption_mask, caption_length)

            sim_neg, train_mn_label, mn_length, sim_neg_for_rank = calculate_sim_neg_for_train(img_mn, txt_mn, caption_length, label)

            mn_loss = mn_loss_fun(sim_neg, train_mn_label, mn_length)

            id_loss_global = id_loss_fun_global(img_global, txt_global, label)
            id_loss_local = id_loss_fun_local(img_local, txt_local, label)
            id_loss = id_loss_global + id_loss_local * 0.5

            ranking_loss_global = ranking_loss_fun(img_global, txt_global, label)
            ranking_loss_local = ranking_loss_fun(img_local, txt_local, label)
            ranking_loss_neg = ranking_loss_fun(img_local, txt_local, label, sim_neg=sim_neg_for_rank)

            ranking_loss = ranking_loss_global + 0.5 * ranking_loss_local + 0.25 * ranking_loss_neg

            optimizer.zero_grad()
            loss = (id_loss + ranking_loss + mn_loss)
            loss.backward()
            optimizer.step()

            if (times + 1) % 50 == 0:
                logging.info("Epoch: %d/%d Setp: %d, ranking_loss: %.2f, id_loss: %.2f, mn_loss: %.5f"
                             % (epoch, opt.epoch, times + 1, ranking_loss, id_loss, mn_loss))

            ranking_loss_sum += ranking_loss
            id_loss_sum += id_loss
            mn_loss_sum += mn_loss
        ranking_loss_avg = ranking_loss_sum / (times + 1)
        id_loss_avg = id_loss_sum / (times + 1)
        mn_loss_avg = mn_loss_sum / (times + 1)

        logging.info("Epoch: %d/%d , ranking_loss: %.2f, id_loss: %.2f, mn_loss: %.5f"
                     % (epoch, opt.epoch, ranking_loss_avg, id_loss_avg, mn_loss_avg))

        print(opt.model_name)
        network.eval()
        test_best = test(opt, epoch, network, test_img_dataloader, test_txt_dataloader, test_best)
        network.train()

        if test_best > test_history:
            test_history = test_best
            state = {
                'network': network.cpu().state_dict(),
                'test_best': test_best,
                'epoch': epoch,
                'id_local': id_loss_fun_local.cpu().state_dict(),
                'id_global': id_loss_fun_global.cpu().state_dict()
            }
            save_checkpoint(state, opt)
            network.to(opt.device)
            id_loss_fun_local.to(opt.device)
            id_loss_fun_global.to(opt.device)

        scheduler.step()

    logging.info('Training Done')


if __name__ == '__main__':
    set_seed(3407)
    opt = options().opt
    train(opt)
