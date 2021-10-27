
import torch
import torch.utils.data
from torch import nn

from Nets.model import Model
from Configs.trainConf import configs
from Utils.utils import *
import torch.optim as optim

import sys
sys.path.insert(0, ".")

import numpy as np
import random
from reprod_log import ReprodLogger


def train_some_iters(model,
                     criterion,
                     optimizer,
                     fake_data,
                     fake_label,
                     device,
                     epoch,
                     apex=False,
                     max_iter=2):
    # needed to avoid network randomness
    #model.eval()

    loss_list = []
    for idx in range(max_iter):
        image = torch.from_numpy(fake_data).to(device)
        target = torch.from_numpy(fake_label).long().to(device)

        output = model(image)
        reprod_logger = ReprodLogger()

        #optimizer.update_lr(idx)
        #print(optimizer.lr)
        #criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=0)  # ignore pad
        loss = criterion(output.view(-1, output.size(-1)), target.view(-1))
        loss_list.append(loss)
        #output.retain_grad()
        #for name, param in model.named_parameters():
            #if param.requires_grad:
               # param.retain_grad()

        loss.backward()
        #parameters=filter(lambda x: x.requires_grad, model.named_parameters())
        #print(output.grad.sum())
        optimizer.step()
        #count = 0
        #for name, param in model.named_parameters():
            #if param.requires_grad:
                #p_grad = param.grad.sum()
                #param = param.sum()
                #if 'running_mean' in name:
                    #name = name.replace('running_mean','_mean')
                #elif 'running_var' in name:
                    #name = name.replace('running_var','_variance')
                #reprod_logger.add(name, param.detach().cpu().numpy())
                #count += 1
        #print(count)
        #reprod_logger.save("./val/bp_param_torch.npy")
        #print(output.grad.sum())

        optimizer.zero_grad()

    return loss_list

def main():
    random.seed(configs.random_seed)
    np.random.seed(configs.random_seed)
    torch.manual_seed(configs.random_seed)
    torch.cuda.manual_seed(configs.random_seed)

    device = torch.device(configs.device)


    print("Creating model")
    model = Model(configs.net)
    # model = load_part_of_model(model , path=' ******************************')
    param = torch.load('./models/pren.pth')
    model.load_state_dict(param['state_dict'])
    model.to(device)


    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=0)

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.999995)
    optimizer = ScheduledOptim(optim.Adadelta(filter(lambda x: x.requires_grad, model.parameters())),
                                    init_lr=configs.lr, milestones=configs.lr_milestones,
                                    gammas=configs.lr_gammas)
#,lr=configs.lr, weight_decay=configs.weight_decay


    print("Start training")

    fake_data = np.load("fake_data.npy")
    fake_label = np.load("fake_label.npy")

    loss_list = train_some_iters(
        model,
        criterion,
        optimizer,
        fake_data,
        fake_label,
        device,
        0,
        max_iter=5)

    print(loss_list)
    return loss_list


if __name__ == "__main__":
    loss_list = main()
    #reprod_logger = ReprodLogger()
    #for idx, loss in enumerate(loss_list):
        #reprod_logger.add(f"loss_{idx}", loss.detach().cpu().numpy())
    #reprod_logger.save("./val/backward_torch.npy")