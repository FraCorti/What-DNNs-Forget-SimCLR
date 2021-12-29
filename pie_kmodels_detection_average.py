import numpy as np
import argparse

import utils
from utils import data, pruning
from datetime import datetime, date
import logging
import torch
import os

import glob
import random

import torch
from torch import nn
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--models_number', default=30, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--model_depth', default=16, type=int)
parser.add_argument('--debug', default=False, type=bool)
parser.add_argument('--cuda_device', default=0, type=int)
parser.add_argument('--dropout', default=0.0, type=float)
parser.add_argument('--workers', default=0, type=int)
parser.add_argument('--widen_factor', default=2, type=int)
parser.add_argument('--cuda_deterministic', default=False, type=bool)
parser.add_argument('--seed', default=1, type=int)

args = parser.parse_args()
depth = args.model_depth
seed = args.seed
cuda_deterministic = args.cuda_deterministic
batch_size = args.batch_size
widen_factor = args.widen_factor
dropout = args.dropout
cuda_device = args.cuda_device
debug = args.debug
models_number = args.models_number
workers = args.workers

if cuda_deterministic:
    torch.set_printoptions(profile='full')
    torch.cuda.manual_seed(seed)
    print("Setting CUBLAS to deterministic behaviour")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)

        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth, num_classes=10, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)


class WideResNetPIE(nn.Module):
    def __init__(self, depth=10, widen_factor=2, dropout=0.0):
        super().__init__()
        self.base = WideResNet(depth=depth, num_classes=10, widen_factor=widen_factor, dropRate=dropout)

    def forward(self, x):
        return self.base(x)


device = torch.device("cuda:{}".format(cuda_device) if torch.cuda.is_available() else "cpu")

if debug:
    today = date.today()
    day = today.strftime("%d_%m_%Y")
    now = datetime.now()
    time = now.strftime("%H_%M_%S")
    logging.basicConfig(
        filename="/home/f/fraco1997/compressed_model_v2/logs/PIEs_DEBUG_wideResnet_day:{}_time:{}.log".format(
            day, time),
        level=logging.INFO)
    print(torch.cuda.memory_stats(device=device))

loaders = utils.data.get_ordered_cifar10_validation(batch_size=batch_size, workers=workers)

models_notpruned_paths = []

# wideResNet_0.3pruning_205epochs_16depth_0.0dropout
not_pruned_models_loaded = 0
for file in glob.glob(
        "/home/f/fraco1997/compressed_model_v2/models/wideResNet_0.0pruning_205epochs_16depth_0.0dropout*.pt"):
    if not_pruned_models_loaded < models_number:
        models_notpruned_paths.append(file)
        not_pruned_models_loaded += 1

if not debug:
    print("Found {} NOT pruned models".format(len(models_notpruned_paths)))
else:
    logging.info("Found {} NOT pruned models".format(len(models_notpruned_paths)))

models = []

# load NOT PRUNED models
for path in models_notpruned_paths:
    model = WideResNetPIE(depth=depth, widen_factor=widen_factor, dropout=dropout)
    model.load_state_dict(
        torch.load(path,
                   map_location=device))
    model.to(device)
    models.append(model)

if not debug:
    print("Loaded {} NOT pruned models".format(len(models_notpruned_paths)))
else:
    logging.info("Loaded {} NOT pruned models".format(len(models_notpruned_paths)))

for models_pruning in [0.3, 0.5, 0.7, 0.9]:

    models_pruned_paths = []
    pruned_models_loaded = 0

    # load PRUNED models
    for file in glob.glob(
            "/home/f/fraco1997/compressed_model_v2/models/wideResNet_{}pruning_205epochs_16depth_0.0dropout*.pt".format(
                models_pruning)):
        if pruned_models_loaded < models_number:
            pruned_models_loaded += 1
            models_pruned_paths.append(file)

    if not debug:
        print("Found {} pruned models: {}".format(models_pruning, len(models_pruned_paths)))
    else:
        logging.info("Found {} pruned models: {}".format(models_pruning, len(models_pruned_paths)))

    pruned_models = []

    for path in models_pruned_paths:
        model = WideResNetPIE(depth=depth, widen_factor=widen_factor, dropout=dropout)
        model.load_state_dict(
            torch.load(path,
                       map_location=device))
        model_sparsity = utils.pruning.get_model_sparsity(model)
        if models_pruning != round(model_sparsity / 100, 1):
            print("Model loaded with a different sparsity of {}!".format(model_sparsity))
        model.to(device)
        pruned_models.append(model)

    if not debug:
        print("Loaded {} pruned models: {}".format(models_pruning, len(models_pruned_paths)))
    else:
        logging.info("Loaded {} pruned models: {}".format(models_pruning, len(models_pruned_paths)))
    total_pies = 0

    for i in range(30):

        pie = 0

        for input_, label in loaders["valid"]:
            # print("Current input: ")
            # print(input_)
            # print("Current label: ")
            # print(label)
            # if debug:
            # logging.info("Current label: {}".format(label))
            # logging.info("Length of input samples: {}".format(len(input_)))

            input_ = input_.to(device)
            label = label.to(device)

            # lists to store the results of the models
            results_model_notpruned = []
            results_model_pruned = []

            most_frequent_labels_not_pruned = []
            most_frequent_labels_pruned = []

            with torch.set_grad_enabled(False):

                # iterate over the not pruned model and save most outputs
                for model in models:
                    output_model = model(input_)
                    # print("Output of the model:")
                    # print(output_model)
                    _, preds_output_model = torch.max(output_model, 1)
                    # if debug and batch_size < 3:
                    # logging.info("Output shape model NOT PRUNED: {}".format(preds_output_model.shape))
                    # if batch_size < 3:
                    # logging.info("Output model NOT PRUNED: {}".format(output_model))
                    results_model_notpruned.append(preds_output_model.cpu().detach().numpy())

                # if debug:
                # logging.info("Output NOT PRUNED models:")
                # logging.info(results_model_notpruned)
                # logging.info("All models NOT PRUNED output len: {}".format(len(results_model_notpruned)))

                for i in range(len(input_)):  # - 1
                    models_results = []

                    for model_output in range(len(results_model_notpruned)):  # - 1
                        models_results.append(results_model_notpruned[model_output][i])

                    most_frequent_labels_not_pruned.append(np.bincount(np.array(models_results)).argmax())

                if debug:
                    # logging.info("Result models not pruned: {}".format(results_model_notpruned))
                    logging.info("Most frequent NOT PRUNED la: {}".format(most_frequent_labels_not_pruned))

                for model in pruned_models:
                    output_model = model(input_)
                    _, preds_output_model = torch.max(output_model, 1)
                    # if debug:
                    # logging.info("Output shape model PRUNED: {}".format(preds_output_model.shape))
                    # if batch_size < 3:
                    # logging.info("Output model PRUNED: {}".format(output_model))
                    results_model_pruned.append(preds_output_model.cpu().detach().numpy())

                if debug:
                    pass
                    # logging.info("All models PRUNED output len: {}".format(len(results_model_notpruned)))

                for i in range(len(input_)):  # - 1
                    models_results = []

                    if debug:
                        pass
                        # logging.info("Numpy arrays in results_model_notpruned: {}".format(results_model_notpruned))
                        # logging.info(results_model_notpruned)
                        # logging.info("Models number: {}".format(len(models)))

                    for model_output in range(len(results_model_pruned)):  # - 1
                        models_results.append(results_model_pruned[model_output][i])

                    most_frequent_labels_pruned.append(np.bincount(np.array(models_results)).argmax())

                if debug:
                    # logging.info("Result models PRUNED: {}".format(results_model_pruned))
                    logging.info("Most frequent PRUNED labels: {}".format(most_frequent_labels_pruned))

                for i in range(len(input_)):  # - 1
                    if most_frequent_labels_not_pruned[i] != most_frequent_labels_pruned[i]:
                        logging.info("not_pruned: {} pruned: {}".format(most_frequent_labels_not_pruned[i],
                                                                        most_frequent_labels_pruned[i]))
                        pie += 1
                logging.info("current PIEs: {}".format(pie))
        if debug:
            logging.info(print(
                "--------------------------------- Pruned: {} Models used: {} Found PIEs: {} ---------------------------------".format(
                    models_pruning, models_number, pie)))
        else:
            print("Pruned: {} Models used: {} Found PIEs: {}".format(models_pruning, models_number, pie))

        total_pies += pie

    print("Pruned: {} CUDA deterministic: {} Averaged Found PIEs over 30 runs: {}".format(models_pruning,
                                                                                          cuda_deterministic,
                                                                                          total_pies / 30))
