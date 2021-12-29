import torch
from torch.optim import lr_scheduler

import utils
import torch.optim as optim
import random
import argparse
from torch import nn
import logging
from datetime import datetime

from utils import models, data, pruning

parser = argparse.ArgumentParser()
parser.add_argument('--pruning', default=0.0, type=float)
parser.add_argument('--model_depth', default=16, type=int)
parser.add_argument('--begin_step', default=1000, type=int)
parser.add_argument('--end_step', default=20000, type=int)
parser.add_argument('--frequency', default=500, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--cuda_device', default=0, type=int)
parser.add_argument('--debug', default=False, type=bool)
parser.add_argument('--shuffle', default=True, type=bool)
parser.add_argument('--dropout', default=0.0, type=float)
parser.add_argument('--workers', default=0, type=int)
parser.add_argument('--learning_rate', default=1.0, type=float)
parser.add_argument('--weight_decay', default=0.0005, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--index_start', type=int)
parser.add_argument('--index_end', type=int)
parser.add_argument('--epochs', default=205, type=int)

args = parser.parse_args()
final_sparsity = args.pruning
epochs = args.epochs
batch_size = args.batch_size
cuda_device = args.cuda_device
begin_step = args.begin_step
end_step = args.end_step
frequency = args.frequency
dropout = args.dropout
depth = args.model_depth
shuffle = args.shuffle
debug = args.debug
workers = args.workers
weight_decay = args.weight_decay
learning_rate = args.learning_rate
momentum = args.momentum
index_start = args.index_start
index_end = args.index_end

initial_sparsity = 0.0

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device("cuda:{}".format(cuda_device) if torch.cuda.is_available() else "cpu")

if debug:
    print(torch.cuda.memory_stats(device=device))

loaders = utils.data.get_cifar10(batch_size=batch_size, workers=workers, shuffle=shuffle)

# logging file
if not debug:
    now = datetime.now()
    time = now.strftime("%H_%M_%S")
    logging.basicConfig(
        filename="/home/f/fraco1997/compressed_model_v2/logs/wideResnet_pruning{}_indexStart{}_indexEnd{}_time:{}.log".format(
            final_sparsity, index_start, index_end,
            time),
        level=logging.INFO)

    logging.info("------------------- Setup Configuration -------------------")
    logging.info(
        "Models to train: {} Final sparsity: {} Pruning begin step: {} Pruning end step: {} Pruning frequency: {} Start seed index: {} End seed index: {} Models depth: {} Epochs: {} Batch size: {} Weight Decay: {} Dropout: {} Momentum: {} ".format(
            index_end - index_start, final_sparsity, begin_step, end_step, frequency, index_start, index_end,
            depth,
            epochs, batch_size, weight_decay, dropout, momentum))

for model_number in range(index_start, index_end):
    print("Training model number: {}".format(model_number))

    torch.manual_seed(model_number)
    torch.cuda.manual_seed(model_number)
    # np.random.seed(args.seed)
    random.seed(model_number)

    model = utils.models.get_wide_resnet(depth=depth, dropout=dropout)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[5, 30, 50, 80], gamma=0.1)

    loss_train = []
    loss_valid = []
    accuracy_train = []
    accuracy_valid = []

    step = 0

    for epoch in range(epochs):

        if debug:
            print("--------------- Epoch {} , Learning rate {} ---------------".format(epoch,
                                                                                       scheduler.get_last_lr()))

        step_train = 0
        step_validation = 0
        loss_training = 0
        loss_validation = 0
        correct_validation = 0
        correct_training = 0
        total_training = 0
        total_validation = 0

        # alternate training and validation phase
        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            validation_pred = []
            validation_true = []

            # cycle on the batches of the train and validation dataset
            for i, data in enumerate(loaders[phase]):

                if phase == "train":
                    step += 1
                    step_train += 1

                if phase == "valid":
                    step_validation += 1

                images, labels = data
                images, labels = images.to(device), labels.to(device)

                # reset the gradient
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):

                    # forward phase of the net
                    outputs = model(images)

                    # compute loss
                    loss = criterion(outputs, labels)

                    # compute accuracy
                    _, predicted = torch.max(outputs.data, 1)

                    if phase == "train":
                        loss_training += loss.item()
                        total_training += labels.size(0)
                        correct_training += (predicted == labels).sum().item()
                        loss.backward()
                        optimizer.step()

                    if phase == "valid":
                        loss_validation += loss.item()
                        total_validation += labels.size(0)
                        correct_validation += (predicted == labels).sum().item()

        current_train_loss = loss_training / step_train
        current_validation_loss = loss_validation / step_validation
        current_train_accuracy = 100 * correct_training / total_training
        current_validation_accuracy = 100 * correct_validation / total_validation

        # scheduler update
        scheduler.step()
        loss_train.append(current_train_loss)
        accuracy_train.append(current_train_accuracy)

        loss_valid.append(current_validation_loss)
        accuracy_valid.append(current_validation_accuracy)

        if debug:
            print(
                "loss: {}  accuracy: {}  val_loss: {} val_accuracy: {} current_steps: {} model_sparsity: {:.2f}% ".format(
                    current_train_loss,
                    current_train_accuracy,
                    current_validation_loss,
                    current_validation_accuracy,
                    step, utils.pruning.get_model_sparsity(model)))

        if epoch == epochs - 1 and not debug:
            logging.info(
                "loss: {}  accuracy: {}  val_loss: {} val_accuracy: {} total_steps: {} final_sparsity: {:.2f}%".format(
                    current_train_loss,
                    current_train_accuracy,
                    current_validation_loss,
                    current_validation_accuracy,
                    step, utils.pruning.get_model_sparsity(model)))

        loss_training = 0
        loss_validation = 0
        correct_training = 0
        correct_validation = 0
        step_train = 0
        step_validation = 0
        total_training = 0
        total_validation = 0

    if not debug:
        utils.data.plot_loss_curves(loss_train=loss_train, loss_valid=loss_valid, final_sparsity=final_sparsity,
                                    epochs=epochs, depth=depth, dropout=dropout, model_id=model_number)
        utils.data.plot_accuracy_curves(accuracy_train=accuracy_train, accuracy_valid=accuracy_valid, epochs=epochs,
                                        depth=depth, dropout=dropout, final_sparsity=final_sparsity,
                                        model_id=model_number)

    if not debug:
        torch.save(model.state_dict(),
                   '/home/f/fraco1997/compressed_model_v2/models/wideResNet_{}pruning_{}epochs_{}depth_{}dropout_id{}.pt'.format(
                       final_sparsity, epochs, depth, dropout, model_number))
