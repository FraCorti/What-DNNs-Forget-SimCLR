import torch
import torch.nn.utils.prune as prune
import glob

""" 
       l1_unstructured prune on the unpruned parameters, the final pruned percentage of weights are: 
       1 - (1 - gamma)^n where gamma is the percentage passed every iteration and n is the number of 
       iteration 
"""


# iterate over the model and prune weights and biases
def magnitude_pruning(model, theta):
    for name, module in model.named_modules():

        # prune connections in 2D-Conv modules
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=theta)

        # prune connections in Linear modules
        elif isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=theta)
            prune.l1_unstructured(module, name='bias', amount=theta)


# remove the weight_mask and bias_mask to make pruning permanent
def remove_pruning_masks(model):
    for name, module in model.named_modules():

        if isinstance(module, torch.nn.Conv2d):
            prune.remove(module, 'weight')

        if isinstance(module, torch.nn.Linear):
            prune.remove(module, 'bias')
            prune.remove(module, 'weight')


def check_pruning(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            print("Unpruned parameters of the module:")
            print(list(module.named_parameters()))
            print("Pruning masks of the module:")
            print(list(module.named_buffers()))
            print("Module weights by merging the pruning mask:")
            print(module.weight)
            print("Module biases by merging the pruning mask:")
            print(module.bias)


# Returns the gamma parameters to be passed inside Pytorch's pruning method
def get_gammas(initial_sparsity, final_sparsity, begin_step, end_step, frequency):
    polynomial_pruning_percentage = []
    for step in range(begin_step, end_step + 1, frequency):
        current_pruning_percentage = final_sparsity + (initial_sparsity - final_sparsity) * pow(
            1 - ((step - begin_step) / (end_step - begin_step)), 3)
        polynomial_pruning_percentage.append(current_pruning_percentage)

    # obtain the gamma parameters to pass in the Pytorch pruning method
    gamma_parameters = []

    first = False

    for pruning_percentage in polynomial_pruning_percentage:
        if first:
            denominator = 1
            for gamma in gamma_parameters:
                denominator *= (1 - gamma)
            gamma_parameters.append(1 - ((1 - pruning_percentage) / denominator))
        else:
            gamma_parameters.append(pruning_percentage)
            if pruning_percentage != 0.0:
                first = True
    return gamma_parameters


def get_pruned_models_number(pruning_percentage):
    pruned_models_found = 0
    for file in glob.glob(
            "/home/f/fraco1997/compressed_model_v2/models/wideResNet_{}pruning*.pt".format(
                pruning_percentage)):
        pruned_models_found += 1
    return pruned_models_found


def get_model_sparsity(model):
    zeros_weight = 0
    total_weight = 0

    for name, module in model.named_modules():

        if isinstance(module, torch.nn.Conv2d):
            zeros_weight += float(torch.sum(module.weight == 0))
            total_weight += float(module.weight.nelement())

        elif isinstance(module, torch.nn.Linear):
            zeros_weight += float(torch.sum(module.weight == 0))
            total_weight += float(module.weight.nelement())
    # print("Model current sparsity: {:.2f}%".format(100. * zeros_weight / total_weight))
    return 100. * zeros_weight / total_weight
