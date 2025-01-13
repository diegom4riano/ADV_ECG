import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.nn as nn
from model.CNN import CNN
from utils.DataLoader import ECGDataset, ecg_collate_func
import sys
import datetime
import os

data_dirc = 'data/'
RAW_LABELS = np.load(data_dirc+'raw_labels.npy', allow_pickle=True)
PERMUTATION = np.load(data_dirc+'random_permutation.npy', allow_pickle=True)
BATCH_SIZE = 16
MAX_SENTENCE_LENGTH = 18000
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

LEARNING_RATE = 0.001
NUM_EPOCHS = 200  # number epoch to train

data = np.load(data_dirc+'raw_data.npy', allow_pickle=True)
data = data[PERMUTATION]
RAW_LABELS = RAW_LABELS[PERMUTATION]
mid = int(len(data)*0.995)
val_data = data[mid:]
val_label = RAW_LABELS[mid:]
val_dataset = ECGDataset(val_data, val_label)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=BATCH_SIZE,
                                           collate_fn=ecg_collate_func,
                                           shuffle=False)
model = CNN(num_classes=4)
model = nn.DataParallel(model)
model = model.to(device)
model.load_state_dict(torch.load('saved_model/best_model.pth', map_location=lambda storage, loc: storage))

def pgd(inputs, lengths, targets, model, criterion, eps = None, step_alpha = None, num_steps = None, pseudo_inv = False, technique = None):
    """
    :param inputs: Clean samples (Batch X Size)
    :param targets: True labels
    :param model: Model
    :param criterion: Loss function
    :param gamma:
    :return:
    """
    crafting_input = torch.autograd.Variable(inputs.clone(), requires_grad=True)
    crafting_target = torch.autograd.Variable(targets.clone())

    for i in range(num_steps):
        output = model(crafting_input)
        loss = criterion(output, crafting_target)
        if crafting_input.grad is not None:
            crafting_input.grad.data.zero_()
        loss.backward(retain_graph=True)
        
        ####precision paradox
        if pseudo_inv == True:     
            # Get input dimensions
            batch_size = crafting_input.size(0)
            
            # Precision calculation
            jacobian = crafting_input.grad.data.clone()
            
            # Reshape jacobian to match the expected dimensions
            jacobian = jacobian.view(batch_size, -1)  # Flatten the jacobian
            
            # Convert output to probabilities and get target one-hot encoding
            output_probs = F.softmax(output, dim=1)
            target_one_hot = F.one_hot(crafting_target, num_classes=output.size(1)).float()
            delta = output_probs - target_one_hot
            
            # Reshape delta to match jacobian dimensions
            delta = delta.view(batch_size, -1)
            
            # Calculate pseudo-inverse and minimum perturbation
            jacobian_pseudo_inv = torch.pinverse(jacobian)
            min_perturbation = torch.matmul(jacobian_pseudo_inv, delta)
            
            #padding min_perturbation with 1's
            # Create tensor of ones with target shape
            target_shape = (batch_size, 1, MAX_SENTENCE_LENGTH)  # (16, 1, 18000)
            padded_perturbation = torch.ones(target_shape, device=min_perturbation.device)

            # Modify the reshaping code
            min_perturbation = min_perturbation.view(-1)  # Flatten
            target_length = batch_size * MAX_SENTENCE_LENGTH
            if min_perturbation.numel() > target_length:
                min_perturbation = min_perturbation[:target_length]  # Trim to needed size

            # Reshape with explicit size calculation
            final_length = min_perturbation.numel() // batch_size
            min_perturbation = min_perturbation[:batch_size*final_length]
            min_perturbation = min_perturbation.view(batch_size, 1, final_length)

            # Create padded version with correct dimensions
            padded_perturbation = torch.ones(
                (batch_size, 1, MAX_SENTENCE_LENGTH), 
                device=min_perturbation.device
            )

            # Copy values ensuring dimensions match
            copy_length = min(min_perturbation.size(2), MAX_SENTENCE_LENGTH)
            padded_perturbation[:, :, :copy_length] = min_perturbation[:, :, :copy_length]

            # Update min_perturbation
            min_perturbation = padded_perturbation

            ####
            added = torch.sign(crafting_input.grad.data * min_perturbation)
        else:
            added = torch.sign(crafting_input.grad.data)
        
        # for targeted attack we need to minimize the target class ( - instead of + )
        step_output = crafting_input - step_alpha * added
        
        total_adv = step_output - inputs
        total_adv = torch.clamp(total_adv, -eps, eps)
        crafting_output = inputs + total_adv
        crafting_input = torch.autograd.Variable(crafting_output.clone(), requires_grad=True)
    added = crafting_output - inputs
    crafting_output = inputs+ added
    crafting_output_clamp = crafting_output.clone()
    # remove pertubations on the padding
    for i in range(crafting_output_clamp.size(0)):
        remainder = MAX_SENTENCE_LENGTH - lengths[i]
        if remainder > 0:
            crafting_output_clamp[i][0][:int(remainder / 2)] = 0
            crafting_output_clamp[i][0][-(remainder - int(remainder / 2)):] = 0
    return crafting_output_clamp

def success_rate(data_loader, model, eps = 1, step_alpha = None, num_steps = None, pseudo_inv = False):
    model.eval()
    correct = 0.0
    correct_clamp = 0.0
    adv_exps = []
    adv_probs = []
    adv_classes = []
    pred_classes = []
    pred_probs = []
    pred_exps = []

    for bi, (inputs, lengths, targets) in enumerate(data_loader):
        inputs_batch, lengths_batch, targets_batch = inputs.to(device), lengths.to(device), targets.to(device)
        
        #try to specify the targets to be different from the original targets
        targets_wanted = targets_batch.clone()
        #targets_wanted = (targets_wanted + 1) % 3
        targets_wanted[:] = 2

        crafted_clamp = pgd(inputs_batch, lengths_batch, targets_wanted, model, F.cross_entropy, eps, step_alpha, num_steps, pseudo_inv)
        output = model(inputs_batch)
        output_clamp = model(crafted_clamp)
        pred = output.data.max(1, keepdim=True)[1].view_as(
            targets_batch)  # get the index of the max log-probability
        pred_clamp = output_clamp.data.max(1, keepdim=True)[1].view_as(targets_batch)
        
        #idx1 = (pred == targets_batch)
        #idx2 = (pred != pred_clamp)
        idx1 = (pred_clamp == targets_wanted)
        idx2 = (pred_clamp != targets_batch)

        idx = idx1 & idx2
        correct_clamp += pred_clamp.eq(targets_batch.view_as(pred_clamp)).cpu().numpy().sum()
        #pred_exps.append(inputs_batch[idx].detach().cpu().numpy())
        adv_classes.append(pred_clamp[idx].detach().cpu().numpy())
        pred_classes.append(pred[idx].cpu().numpy())
        #adv_probs.append(F.softmax(output_clamp, dim=1)[idx].detach().cpu().numpy())
        #pred_probs.append(F.softmax(output, dim=1)[idx].detach().cpu().numpy())
        #adv_exps.append(crafted_clamp[idx].detach().cpu().numpy())

    #adv_exps = np.concatenate(adv_exps)
    #adv_probs = np.concatenate(adv_probs)
    adv_classes = np.concatenate(adv_classes)
    pred_classes = np.concatenate(pred_classes)
    #pred_probs = np.concatenate(pred_probs)
    #pred_exps = np.concatenate(pred_exps)
    path = 'adv_exp/pgd'
    try:
        os.makedirs(path)
    except FileExistsError:
        pass
    
    # Add before the save operations
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    print("pseudo-inv:", pseudo_inv)
    print("adv_classes:", adv_classes)
    print("pred_classes:", pred_classes)

    #np.save(path+f'/p_adv_exps_{timestamp}.npy', adv_exps)
    #np.save(path+f'/p_adv_probs_{timestamp}.npy', adv_probs)
    np.save(path+f'/p_adv_classes_{timestamp}.npy', adv_classes)
    np.save(path+f'/p_pred_classes_{timestamp}.npy', pred_classes)
    #np.save(path+f'/p_pred_probs_{timestamp}.npy', pred_probs)
    #np.save(path+f'/p_pred_exps_{timestamp}.npy', pred_exps)
    correct_clamp /= len(data_loader.sampler)
    return correct_clamp

srpgd = success_rate(val_loader, model, eps = 10, step_alpha = 1, num_steps = 20, pseudo_inv = False)
print('success rate PGD 10,1,20:', srpgd)

srpgd = success_rate(val_loader, model, eps = 10, step_alpha = 1, num_steps = 20, pseudo_inv = True)
print('success rate pseudo-inverse PGD 10,1,20:', srpgd)


sys.stdout.flush()