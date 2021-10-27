import numpy as np
import torch
import torch.nn as nn
from model import Model
from utils import *
import argparse
from reprod_log import ReprodLogger
from Configs.trainConf import configs


# parser = argparse.ArgumentParser()
# # data I/O
# parser.add_argument('-i', '--data_dir', type=str,
#                     default='data', help='Location for the dataset')
# parser.add_argument('-o', '--save_dir', type=str, default='models',
#                     help='Location for parameter checkpoints and samples')
# parser.add_argument('-d', '--dataset', type=str,
#                     default='cifar', help='Can be either cifar|mnist')
# parser.add_argument('-p', '--print_every', type=int, default=50,
#                     help='how many iterations between print statements')
# parser.add_argument('-t', '--save_interval', type=int, default=10,
#                     help='Every how many epochs to write checkpoint/samples?')
# parser.add_argument('-r', '--load_params', type=str, default=None,
#                     help='Restore training from previous model checkpoint?')
# # model
# parser.add_argument('-q', '--nr_resnet', type=int, default=5,
#                     help='Number of residual blocks per stage of the model')
# parser.add_argument('-n', '--nr_filters', type=int, default=160,
#                     help='Number of filters to use across the model. Higher = larger model.')
# parser.add_argument('-m', '--nr_logistic_mix', type=int, default=10,
#                     help='Number of logistic components in the mixture. Higher = more flexible model')
# parser.add_argument('-l', '--lr', type=float,
#                     default=0.0002, help='Base learning rate')
# parser.add_argument('-e', '--lr_decay', type=float, default=0.999995,
#                     help='Learning rate decay, applied every step of the optimization')
# parser.add_argument('-b', '--batch_size', type=int, default=64,
#                     help='Batch size during training per GPU')
# parser.add_argument('-x', '--max_epochs', type=int,
#                     default=5000, help='How many epochs to run in total?')
# parser.add_argument('-s', '--seed', type=int, default=1,
#                     help='Random seed to use')
# args = parser.parse_args()


# torch.manual_seed(args.seed)
# np.random.seed(args.seed)
torch.manual_seed(1)
np.random.seed(1)

if __name__ == "__main__":
    # load model
    # the model is save into ~/.cache/torch/hub/checkpoints/alexnet-owt-4df8aa71.pth

    # def logger
    reprod_logger = ReprodLogger()

    # criterion = discretized_mix_logistic_loss_1d()

    model = Model(configs.net)
    param = torch.load('./models/pren.pth')
    model.load_state_dict(param['state_dict'])
    model.eval()
    # # read or gen fake data
    fake_data = np.load("fake_data.npy")
    fake_data = torch.from_numpy(fake_data)

    fake_label = np.load("fake_label.npy")
    fake_label = torch.from_numpy(fake_label)

    # forward
    out = model(fake_data)

    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=0)  # ignore pad
    loss = criterion(out, fake_label)
    #
    reprod_logger.add("loss", loss.cpu().detach().numpy())
    reprod_logger.save("loss_torch.npy")