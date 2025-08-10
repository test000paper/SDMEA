import torch

print(torch.__version__)
print(torch.cuda.is_available())

import numpy as np
import csv
import math
import random
import gzip
from scipy.stats import bernoulli
import torch
from sklearn import metrics
import matplotlib.pyplot as plt

nummotif = 16  # number of motifs to discover
bases = 'ACGT'  # DNA bases
basesRNA = 'ACGU'  # RNA bases
batch_size = 64  # fixed batch size -> see notes to problem about it
dictReverse = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}  # dictionary to implement reverse-complement mode
reverse_mode = False


def seqtopad(sequence, motlen, kind='DNA'):
    rows = len(sequence) + 2 * motlen - 2
    S = np.empty([rows, 4])
    base = bases if kind == 'DNA' else basesRNA
    for i in range(rows):
        for j in range(4):
            if i - motlen + 1 < len(sequence) and sequence[i - motlen + 1] == 'N' or i < motlen - 1 or i > len(
                    sequence) + motlen - 2:
                S[i, j] = np.float32(0.25)
            elif sequence[i - motlen + 1] == base[j]:
                S[i, j] = np.float32(1)
            else:
                S[i, j] = np.float32(0)
    return np.transpose(S)


def dinucshuffle(sequence):
    b = [sequence[i:i + 2] for i in range(0, len(sequence), 2)]
    random.shuffle(b)
    d = ''.join([str(x) for x in b])
    return d


def complement(seq):
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
    complseq = [complement[base] for base in seq]
    return complseq


def reverse_complement(seq):
    seq = list(seq)
    seq.reverse()
    return ''.join(complement(seq))


class Chip():
    def __init__(self, filename, motiflen=24, reverse_complemet_mode=reverse_mode):
        self.file = filename
        self.motiflen = motiflen
        self.reverse_complemet_mode = reverse_complemet_mode

    def openFile(self):
        train_dataset = []

        with open(self.file, 'r') as data:
            next(data)
            reader = csv.reader(data, delimiter='\t')
            if not self.reverse_complemet_mode:
                for row in reader:
                    train_dataset.append([seqtopad(row[2], self.motiflen), [1]])
                    train_dataset.append([seqtopad(dinucshuffle(row[2]), self.motiflen), [0]])
            else:
                for row in reader:
                    train_dataset.append([seqtopad(row[2], self.motiflen), [1]])
                    train_dataset.append([seqtopad(reverse_complement(row[2]), self.motiflen), [1]])
                    train_dataset.append([seqtopad(dinucshuffle(row[2]), self.motiflen), [0]])
                    train_dataset.append([seqtopad(dinucshuffle(reverse_complement(row[2])), self.motiflen), [0]])

        # random.shuffle(train_dataset)

        train_dataset_pad = train_dataset

        size = int(len(train_dataset_pad) / 3)
        firstvalid = train_dataset_pad[:size]
        secondvalid = train_dataset_pad[size:size + size]
        thirdvalid = train_dataset_pad[size + size:]

        firsttrain = secondvalid + thirdvalid
        secondtrain = firstvalid + thirdvalid
        thirdtrain = firstvalid + secondvalid

        return firsttrain, firstvalid, secondtrain, secondvalid, thirdtrain, thirdvalid, train_dataset_pad


import os
import argparse

parser = argparse.ArgumentParser(description='Train a model with a given transcription factor name.')
parser.add_argument('tf_name', type=str, help='Name of the transcription factor')
args = parser.parse_args()

tf_name = args.tf_name

base_dir = os.path.join('data', 'myExperiment_change', tf_name)

best_hyperparameters_path = os.path.join(base_dir, 'model', 'best_hyperparameters.pth')
os.makedirs(os.path.dirname(best_hyperparameters_path), exist_ok=True)

mymodel_path = os.path.join(base_dir, 'model', 'MyModel_2.pth')


def find_file_with_suffix(directory, suffix):
    for file_name in os.listdir(directory):
        if file_name.endswith(suffix):
            return os.path.join(directory, file_name)
    return None


chipseq_seq_path = find_file_with_suffix(base_dir, '_Stanford_AC.seq')
if chipseq_seq_path is None:
    raise FileNotFoundError(f"No file ending with '_Stanford_AC.seq' was found，Please check the directory: {base_dir}")

chipseq_test_seq_path = find_file_with_suffix(base_dir, '_Stanford_B.seq')
if chipseq_test_seq_path is None:
    raise FileNotFoundError(f"No file ending with '_Stanford_B.seq'was found，Please check the directory: {base_dir}")

chipseq = Chip(chipseq_seq_path)

train1, valid1, train2, valid2, train3, valid3, alldataset = chipseq.openFile()

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class chipseq_dataset(Dataset):

    def __init__(self, xy=None):
        self.x_data = np.asarray([el[0] for el in xy], dtype=np.float32)
        self.y_data = np.asarray([el[1] for el in xy], dtype=np.float32)
        self.x_data = torch.from_numpy(self.x_data)
        self.y_data = torch.from_numpy(self.y_data)
        self.len = len(self.x_data)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


train1_dataset = chipseq_dataset(train1)
train2_dataset = chipseq_dataset(train2)
train3_dataset = chipseq_dataset(train3)
valid1_dataset = chipseq_dataset(valid1)
valid2_dataset = chipseq_dataset(valid2)
valid3_dataset = chipseq_dataset(valid3)
alldataset_dataset = chipseq_dataset(alldataset)

batchSize = 64
if reverse_mode:
    train_loader1 = DataLoader(dataset=train1_dataset, batch_size=batchSize, shuffle=False)
    train_loader2 = DataLoader(dataset=train2_dataset, batch_size=batchSize, shuffle=False)
    train_loader3 = DataLoader(dataset=train3_dataset, batch_size=batchSize, shuffle=False)
    valid1_loader = DataLoader(dataset=valid1_dataset, batch_size=batchSize, shuffle=False)
    valid2_loader = DataLoader(dataset=valid2_dataset, batch_size=batchSize, shuffle=False)
    valid3_loader = DataLoader(dataset=valid3_dataset, batch_size=batchSize, shuffle=False)
    alldataset_loader = DataLoader(dataset=alldataset_dataset, batch_size=batchSize, shuffle=False)
else:
    train_loader1 = DataLoader(dataset=train1_dataset, batch_size=batchSize, shuffle=True)
    train_loader2 = DataLoader(dataset=train2_dataset, batch_size=batchSize, shuffle=True)
    train_loader3 = DataLoader(dataset=train3_dataset, batch_size=batchSize, shuffle=True)
    valid1_loader = DataLoader(dataset=valid1_dataset, batch_size=batchSize, shuffle=False)
    valid2_loader = DataLoader(dataset=valid2_dataset, batch_size=batchSize, shuffle=False)
    valid3_loader = DataLoader(dataset=valid3_dataset, batch_size=batchSize, shuffle=False)
    alldataset_loader = DataLoader(dataset=alldataset_dataset, batch_size=batchSize, shuffle=False)

train_dataloader = [train_loader1, train_loader2, train_loader3]
valid_dataloader = [valid1_loader, valid2_loader, valid3_loader]

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import torch.nn.functional as F

num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001


def logsampler(a, b):
    x = np.random.uniform(low=0, high=1)
    y = 10 ** ((math.log10(b) - math.log10(a)) * x + math.log10(a))
    return y


def sqrtsampler(a, b):
    x = np.random.uniform(low=0, high=1)
    y = (b - a) * math.sqrt(x) + a
    return y


class ConvNet(nn.Module):
    def __init__(self, nummotif, motiflen, poolType, neuType, mode, dropprob, learning_rate, momentum_rate, sigmaConv,
                 sigmaNeu, beta1, beta2, beta3, reverse_complemet_mode=reverse_mode):

        super(ConvNet, self).__init__()
        self.poolType = poolType
        self.neuType = neuType
        self.mode = mode
        self.reverse_complemet_mode = reverse_complemet_mode
        self.dropprob = dropprob
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.sigmaConv = sigmaConv
        self.sigmaNeu = sigmaNeu
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.wConv = torch.randn(nummotif, 4, motiflen).to(device)
        torch.nn.init.normal_(self.wConv, mean=0, std=self.sigmaConv)
        self.wConv.requires_grad = True

        self.wRect = torch.randn(nummotif).to(device)
        torch.nn.init.normal_(self.wRect)
        self.wRect = -self.wRect
        self.wRect.requires_grad = True

        if neuType == 'nohidden':

            if poolType == 'maxavg':
                self.wNeu = torch.randn(2 * nummotif, 1).to(device)
            else:
                self.wNeu = torch.randn(nummotif, 1).to(device)
            self.wNeuBias = torch.randn(1).to(device)
            torch.nn.init.normal_(self.wNeu, mean=0, std=self.sigmaNeu)
            torch.nn.init.normal_(self.wNeuBias, mean=0, std=self.sigmaNeu)

        else:
            if poolType == 'maxavg':
                self.wHidden = torch.randn(2 * nummotif, 32).to(device)
            else:

                self.wHidden = torch.randn(nummotif, 32).to(device)
            self.wNeu = torch.randn(32, 1).to(device)
            self.wNeuBias = torch.randn(1).to(device)
            self.wHiddenBias = torch.randn(32).to(device)
            torch.nn.init.normal_(self.wNeu, mean=0, std=self.sigmaNeu)
            torch.nn.init.normal_(self.wNeuBias, mean=0, std=self.sigmaNeu)
            torch.nn.init.normal_(self.wHidden, mean=0, std=0.3)
            torch.nn.init.normal_(self.wHiddenBias, mean=0, std=0.3)

            self.wHidden.requires_grad = True
            self.wHiddenBias.requires_grad = True

        self.wNeu.requires_grad = True
        self.wNeuBias.requires_grad = True

    def divide_two_tensors(self, x):
        l = torch.unbind(x)

        list1 = [l[2 * i] for i in range(int(x.shape[0] / 2))]
        list2 = [l[2 * i + 1] for i in range(int(x.shape[0] / 2))]
        x1 = torch.stack(list1, 0)
        x2 = torch.stack(list2, 0)
        return x1, x2

    def forward_pass(self, x, mask=None, use_mask=False):

        conv = F.conv1d(x, self.wConv, bias=self.wRect, stride=1, padding=0)
        rect = conv.clamp(min=0)
        maxPool, _ = torch.max(rect, dim=2)
        if self.poolType == 'maxavg':
            avgPool = torch.mean(rect, dim=2)
            pool = torch.cat((maxPool, avgPool), 1)
        else:
            pool = maxPool
        if (self.neuType == 'nohidden'):
            if self.mode == 'training':
                if not use_mask:
                    mask = bernoulli.rvs(self.dropprob, size=len(pool[0]))
                    mask = torch.from_numpy(mask).float().to(device)
                pooldrop = pool * mask
                out = pooldrop @ self.wNeu
                out.add_(self.wNeuBias)
            else:
                out = self.dropprob * (pool @ self.wNeu)
                out.add_(self.wNeuBias)
        else:
            hid = pool @ self.wHidden
            hid.add_(self.wHiddenBias)
            hid = hid.clamp(min=0)
            if self.mode == 'training':
                if not use_mask:
                    mask = bernoulli.rvs(self.dropprob, size=len(hid[0]))
                    mask = torch.from_numpy(mask).float().to(device)
                hiddrop = hid * mask
                out = self.dropprob * (hid @ self.wNeu)
                out.add_(self.wNeuBias)
            else:
                out = self.dropprob * (hid @ self.wNeu)
                out.add_(self.wNeuBias)
        return out, mask

    def forward(self, x):

        if not self.reverse_complemet_mode:
            out, _ = self.forward_pass(x)

        else:

            x1, x2 = self.divide_two_tensors(x)
            out1, mask = self.forward_pass(x1)
            out2, _ = self.forward_pass(x2, mask, True)
            out = torch.max(out1, out2)

        return out


best_AUC = 0
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
# device='cpu'
learning_steps_list = [4000, 8000, 12000, 16000, 20000]
for number in range(5):

    pool_List = ['maxavg']
    random_pool = random.choice(pool_List)

    # neuType_list = ['hidden', 'nohidden']
    neuType_list = ['hidden']
    random_neuType = random.choice(neuType_list)
    dropoutList = [0.5, 0.75, 1.0]

    dropprob = random.choice(dropoutList)

    learning_rate = logsampler(0.0005, 0.05)

    momentum_rate = sqrtsampler(0.95, 0.99)

    sigmaConv = logsampler(10 ** -7, 10 ** -3)

    sigmaNeu = logsampler(10 ** -5, 10 ** -2)
    beta1 = logsampler(10 ** -15, 10 ** -3)
    beta2 = logsampler(10 ** -10, 10 ** -3)
    beta3 = logsampler(10 ** -10, 10 ** -3)

    model_auc = [[], [], []]
    for kk in range(3):
        model = ConvNet(16, 24, random_pool, random_neuType, 'training', dropprob, learning_rate, momentum_rate,
                        sigmaConv, sigmaNeu, beta1, beta2, beta3, reverse_complemet_mode=reverse_mode).to(device)
        if random_neuType == 'nohidden':
            optimizer = torch.optim.SGD([model.wConv, model.wRect, model.wNeu, model.wNeuBias], lr=model.learning_rate,
                                        momentum=model.momentum_rate, nesterov=True)

        else:
            optimizer = torch.optim.SGD(
                [model.wConv, model.wRect, model.wNeu, model.wNeuBias, model.wHidden, model.wHiddenBias],
                lr=model.learning_rate, momentum=model.momentum_rate, nesterov=True)

        train_loader = train_dataloader[kk]
        valid_loader = valid_dataloader[kk]
        learning_steps = 0
        while learning_steps <= 20000:
            model.mode = 'training'
            auc = []
            for i, (data, target) in enumerate(train_loader):
                data = data.to(device)
                target = target.to(device)
                if model.reverse_complemet_mode:
                    target_2 = torch.randn(int(target.shape[0] / 2), 1)
                    for i in range(target_2.shape[0]):
                        target_2[i] = target[2 * i]
                    target = target_2.to(device)

                # Forward pass
                output = model(data)
                if model.neuType == 'nohidden':
                    loss = F.binary_cross_entropy(torch.sigmoid(output),
                                                  target) + model.beta1 * model.wConv.norm() + model.beta3 * model.wNeu.norm()

                else:
                    loss = F.binary_cross_entropy(torch.sigmoid(output),
                                                  target) + model.beta1 * model.wConv.norm() + model.beta2 * model.wHidden.norm() + model.beta3 * model.wNeu.norm()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                learning_steps += 1

                if learning_steps % 4000 == 0:

                    with torch.no_grad():
                        model.mode = 'test'
                        auc = []
                        for i, (data, target) in enumerate(valid_loader):
                            data = data.to(device)
                            target = target.to(device)
                            if model.reverse_complemet_mode:
                                target_2 = torch.randn(int(target.shape[0] / 2), 1)
                                for i in range(target_2.shape[0]):
                                    target_2[i] = target[2 * i]
                                target = target_2.to(device)
                            # Forward pass
                            output = model(data)
                            pred_sig = torch.sigmoid(output)
                            pred = pred_sig.cpu().detach().numpy().reshape(output.shape[0])
                            labels = target.cpu().numpy().reshape(output.shape[0])

                            auc.append(metrics.roc_auc_score(labels, pred))
                        #                         print(np.mean(auc))
                        model_auc[kk].append(np.mean(auc))
                        print('AUC performance when training fold number ', kk + 1, 'using ',
                              learning_steps_list[len(model_auc[kk]) - 1], 'learning steps = ', np.mean(auc))

    print('                   ##########################################               ')
    for n in range(5):
        AUC = (model_auc[0][n] + model_auc[1][n] + model_auc[2][n]) / 3
        # print(AUC)
        if AUC > best_AUC:
            best_AUC = AUC
            best_learning_steps = learning_steps_list[n]
            best_LearningRate = model.learning_rate
            best_LearningMomentum = model.momentum_rate
            best_neuType = model.neuType
            best_poolType = model.poolType
            best_sigmaConv = model.sigmaConv
            best_dropprob = model.dropprob
            best_sigmaNeu = model.sigmaNeu
            best_beta1 = model.beta1
            best_beta2 = model.beta2
            best_beta3 = model.beta3

print('best_poolType=', best_poolType)
print('best_neuType=', best_neuType)
print('best_AUC=', best_AUC)
print('best_learning_steps=', best_learning_steps)
print('best_LearningRate=', best_LearningRate)
print('best_LearningMomentum=', best_LearningMomentum)
print('best_sigmaConv=', best_sigmaConv)
print('best_dropprob=', best_dropprob)
print('best_sigmaNeu=', best_sigmaNeu)
print('best_beta1=', best_beta1)
print('best_beta2=', best_beta2)
print('best_beta3=', best_beta3)

best_hyperparameters = {'best_poolType': best_poolType, 'best_neuType': best_neuType,
                        'best_learning_steps': best_learning_steps, 'best_LearningRate': best_LearningRate,
                        'best_LearningMomentum': best_LearningMomentum, 'best_sigmaConv': best_sigmaConv,
                        'best_dropprob': best_dropprob,
                        'best_sigmaNeu': best_sigmaNeu, 'best_beta1': best_beta1, 'best_beta2': best_beta2,
                        'best_beta3': best_beta3}
# torch.save(best_hyperparameters, 'result/best_hyperpamarameters.pth')
torch.save(best_hyperparameters, best_hyperparameters_path)


# input of shape(batch_size,inp_chan,iW)
class ConvNet_test(nn.Module):
    def __init__(self, nummotif, motiflen, poolType, neuType, mode, learning_steps, learning_rate, learning_Momentum,
                 sigmaConv, dropprob, sigmaNeu, beta1, beta2, beta3, reverse_complemet_mode):
        super(ConvNet_test, self).__init__()
        self.poolType = poolType
        self.neuType = neuType
        self.mode = mode
        self.learning_rate = learning_rate
        self.reverse_complemet_mode = reverse_complemet_mode
        self.momentum_rate = learning_Momentum
        self.sigmaConv = sigmaConv
        self.wConv = torch.randn(nummotif, 4, motiflen).to(device)
        torch.nn.init.normal_(self.wConv, mean=0, std=self.sigmaConv)
        self.wConv.requires_grad = True
        self.wRect = torch.randn(nummotif).to(device)
        torch.nn.init.normal_(self.wRect)
        self.wRect = -self.wRect
        self.wRect.requires_grad = True
        self.dropprob = dropprob
        self.sigmaNeu = sigmaNeu
        self.wHidden = torch.randn(2 * nummotif, 32).to(device)
        self.wHiddenBias = torch.randn(32).to(device)
        if neuType == 'nohidden':

            if poolType == 'maxavg':
                self.wNeu = torch.randn(2 * nummotif, 1).to(device)
            else:
                self.wNeu = torch.randn(nummotif, 1).to(device)
            self.wNeuBias = torch.randn(1).to(device)
            torch.nn.init.normal_(self.wNeu, mean=0, std=self.sigmaNeu)
            torch.nn.init.normal_(self.wNeuBias, mean=0, std=self.sigmaNeu)

        else:
            if poolType == 'maxavg':
                self.wHidden = torch.randn(2 * nummotif, 32).to(device)
            else:

                self.wHidden = torch.randn(nummotif, 32).to(device)
            self.wNeu = torch.randn(32, 1).to(device)
            self.wNeuBias = torch.randn(1).to(device)
            self.wHiddenBias = torch.randn(32).to(device)
            torch.nn.init.normal_(self.wNeu, mean=0, std=self.sigmaNeu)
            torch.nn.init.normal_(self.wNeuBias, mean=0, std=self.sigmaNeu)
            torch.nn.init.normal_(self.wHidden, mean=0, std=0.3)
            torch.nn.init.normal_(self.wHiddenBias, mean=0, std=0.3)

            self.wHidden.requires_grad = True
            self.wHiddenBias.requires_grad = True

        self.wNeu.requires_grad = True
        self.wNeuBias.requires_grad = True

        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3

    def divide_two_tensors(self, x):
        l = torch.unbind(x)

        list1 = [l[2 * i] for i in range(int(x.shape[0] / 2))]
        list2 = [l[2 * i + 1] for i in range(int(x.shape[0] / 2))]
        x1 = torch.stack(list1, 0)
        x2 = torch.stack(list2, 0)
        return x1, x2

    def forward_pass(self, x, mask=None, use_mask=False):

        conv = F.conv1d(x, self.wConv, bias=self.wRect, stride=1, padding=0)
        rect = conv.clamp(min=0)
        maxPool, _ = torch.max(rect, dim=2)
        if self.poolType == 'maxavg':
            avgPool = torch.mean(rect, dim=2)
            pool = torch.cat((maxPool, avgPool), 1)
        else:
            pool = maxPool
        if (self.neuType == 'nohidden'):
            if self.mode == 'training':
                if not use_mask:
                    mask = bernoulli.rvs(self.dropprob, size=len(pool[0]))
                    mask = torch.from_numpy(mask).float().to(device)
                pooldrop = pool * mask
                out = pooldrop @ self.wNeu
                out.add_(self.wNeuBias)
            else:
                out = self.dropprob * (pool @ self.wNeu)
                out.add_(self.wNeuBias)
        else:
            hid = pool @ self.wHidden
            hid.add_(self.wHiddenBias)
            hid = hid.clamp(min=0)
            if self.mode == 'training':
                if not use_mask:
                    mask = bernoulli.rvs(self.dropprob, size=len(hid[0]))
                    mask = torch.from_numpy(mask).float().to(device)
                hiddrop = hid * mask
                out = self.dropprob * (hid @ self.wNeu)
                out.add_(self.wNeuBias)
            else:
                out = self.dropprob * (hid @ self.wNeu)
                out.add_(self.wNeuBias)
        return out, mask

    def forward(self, x):

        if not self.reverse_complemet_mode:
            out, _ = self.forward_pass(x)
        else:

            x1, x2 = self.divide_two_tensors(x)
            out1, mask = self.forward_pass(x1)
            out2, _ = self.forward_pass(x2, mask, True)
            out = torch.max(out1, out2)

        return out


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
learning_steps_list = [4000, 8000, 12000, 16000, 20000]
best_AUC = 0

# best_hyperparameters = torch.load('result/best_hyperpamarameters.pth')
best_hyperparameters = torch.load(best_hyperparameters_path)

best_poolType = best_hyperparameters['best_poolType']
best_neuType = best_hyperparameters['best_neuType']
best_learning_steps = best_hyperparameters['best_learning_steps']
best_LearningRate = best_hyperparameters['best_LearningRate']
best_dropprob = best_hyperparameters['best_dropprob']
best_LearningMomentum = best_hyperparameters['best_LearningMomentum']
best_sigmaConv = best_hyperparameters['best_sigmaConv']
best_sigmaNeu = best_hyperparameters['best_sigmaNeu']
best_beta1 = best_hyperparameters['best_beta1']
best_beta2 = best_hyperparameters['best_beta2']
best_beta3 = best_hyperparameters['best_beta3']

train_losses = []
best_AUC = 0
best_model_losses = None

for number_models in range(6):

    model = ConvNet_test(16, 24, best_poolType, best_neuType, 'training', best_learning_steps, best_LearningRate,
                         best_LearningMomentum, best_sigmaConv, best_dropprob, best_sigmaNeu, best_beta1, best_beta2,
                         best_beta3, reverse_complemet_mode=False).to(device)

    if model.neuType == 'nohidden':
        optimizer = torch.optim.SGD([model.wConv, model.wRect, model.wNeu, model.wNeuBias], lr=model.learning_rate,
                                    momentum=model.momentum_rate, nesterov=True)

    else:
        optimizer = torch.optim.SGD(
            [model.wConv, model.wRect, model.wNeu, model.wNeuBias, model.wHidden, model.wHiddenBias],
            lr=model.learning_rate, momentum=model.momentum_rate, nesterov=True)

    train_loader = alldataset_loader
    valid_loader = alldataset_loader
    learning_steps = 0
    while learning_steps <= best_learning_steps:

        for i, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            if model.reverse_complemet_mode:
                target_2 = torch.randn(int(target.shape[0] / 2), 1)
                for i in range(target_2.shape[0]):
                    target_2[i] = target[2 * i]
                target = target_2.to(device)
            # Forward pass
            output = model(data)

            if model.neuType == 'nohidden':
                loss = F.binary_cross_entropy(torch.sigmoid(output),
                                              target) + model.beta1 * model.wConv.norm() + model.beta3 * model.wNeu.norm()

            else:
                loss = F.binary_cross_entropy(torch.sigmoid(output),
                                              target) + model.beta1 * model.wConv.norm() + model.beta2 * model.wHidden.norm() + model.beta3 * model.wNeu.norm()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            learning_steps += 1

            train_losses.append(loss.item())

            if learning_steps % 1000 == 0:
                print(f"Model {number_models + 1}, Learning Step: {learning_steps}, Loss: {loss.item()}")

    with torch.no_grad():
        model.mode = 'test'
        auc = []
        for i, (data, target) in enumerate(valid_loader):
            data = data.to(device)
            target = target.to(device)
            if model.reverse_complemet_mode:
                target_2 = torch.randn(int(target.shape[0] / 2), 1)
                for i in range(target_2.shape[0]):
                    target_2[i] = target[2 * i]
                target = target_2.to(device)
            # Forward pass
            output = model(data)
            pred_sig = torch.sigmoid(output)
            pred = pred_sig.cpu().detach().numpy().reshape(output.shape[0])
            labels = target.cpu().numpy().reshape(output.shape[0])

            auc.append(metrics.roc_auc_score(labels, pred))
        #
        AUC_training = np.mean(auc)
        print('AUC for model ', number_models, ' = ', AUC_training)

        if AUC_training > best_AUC:
            best_AUC = AUC_training
            best_model_losses = train_losses
            state = {'conv': model.wConv, 'rect': model.wRect, 'wHidden': model.wHidden,
                     'wHiddenBias': model.wHiddenBias, 'wNeu': model.wNeu, 'wNeuBias': model.wNeuBias}
            # torch.save(state, 'result/MyModel_2.pth')
            torch.save(state, mymodel_path)

if best_model_losses is not None:
    plt.figure(figsize=(10, 6))
    plt.plot(best_model_losses, label='Best Model Training Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve for Best Model (Best Hyperparameters)')
    plt.legend()

    # 保存损失函数曲线图
    loss_curve_path = os.path.join(base_dir, 'result', 'best_model_training_loss_curve.png')
    plt.savefig(loss_curve_path)
    plt.close()

    print(f"Loss Curve for Best Model save to: {loss_curve_path}")
else:
    print("Can't find Best Model")

# checkpoint = torch.load('result/MyModel_2.pth')
checkpoint = torch.load(mymodel_path)
model = ConvNet_test(16, 24, best_poolType, best_neuType, 'test', best_learning_steps, best_LearningRate,
                     best_LearningMomentum, best_sigmaConv, best_dropprob, best_sigmaNeu, best_beta1, best_beta2,
                     best_beta3, reverse_complemet_mode=reverse_mode).to(device)
model.wConv = checkpoint['conv']
model.wRect = checkpoint['rect']
model.wHidden = checkpoint['wHidden']
model.wHiddenBias = checkpoint['wHiddenBias']
model.wNeu = checkpoint['wNeu']
model.wNeuBias = checkpoint['wNeuBias']

with torch.no_grad():
    model.mode = 'test'
    auc = []

    for i, (data, target) in enumerate(valid_loader):
        data = data.to(device)
        target = target.to(device)
        if model.reverse_complemet_mode:
            target_2 = torch.randn(int(target.shape[0] / 2), 1)
            for i in range(target_2.shape[0]):
                target_2[i] = target[2 * i]
            target = target_2.to(device)
        # Forward pass
        output = model(data)
        pred_sig = torch.sigmoid(output)
        pred = pred_sig.cpu().detach().numpy().reshape(output.shape[0])
        labels = target.cpu().numpy().reshape(output.shape[0])

        auc.append(metrics.roc_auc_score(labels, pred))
    #
    AUC_training = np.mean(auc)
    print(AUC_training)


class Chip_test():
    def __init__(self, filename, motiflen=24, reverse_complemet_mode=reverse_mode):
        self.file = filename
        self.motiflen = motiflen
        self.reverse_complemet_mode = reverse_complemet_mode

    def openFile(self):
        test_dataset = []
        with open(self.file, 'r') as data:
            next(data)
            reader = csv.reader(data, delimiter='\t')
            if not self.reverse_complemet_mode:
                for row in reader:
                    test_dataset.append([seqtopad(row[2], self.motiflen), [int(row[3])]])
            else:
                for row in reader:
                    test_dataset.append([seqtopad(row[2], self.motiflen), [int(row[3])]])
                    test_dataset.append([seqtopad(reverse_complement(row[2]), self.motiflen), [int(row[3])]])

        return test_dataset


chipseq_test = Chip_test(chipseq_test_seq_path)
test_data = chipseq_test.openFile()
test_dataset = chipseq_dataset(test_data)
batchSize = test_dataset.__len__()
test_loader = DataLoader(dataset=test_dataset, batch_size=batchSize, shuffle=False)

with torch.no_grad():
    model.mode = 'test'
    auc = []

    for i, (data, target) in enumerate(test_loader):
        data = data.to(device)
        target = target.to(device)
        if model.reverse_complemet_mode:
            target_2 = torch.randn(int(target.shape[0] / 2), 1)
            for i in range(target_2.shape[0]):
                target_2[i] = target[2 * i]
            target = target_2.to(device)
        # Forward pass
        output = model(data)
        pred_sig = torch.sigmoid(output)
        pred = pred_sig.cpu().detach().numpy().reshape(output.shape[0])
        labels = target.cpu().numpy().reshape(output.shape[0])

        auc.append(metrics.roc_auc_score(labels, pred))
    #
    AUC_training = np.mean(auc)
    print('AUC on test data = ', AUC_training)

with torch.no_grad():
    model.mode = 'test'
    auc = []
    acc = []
    specificity = []
    sensitivity = []

    for i, (data, target) in enumerate(test_loader):
        data = data.to(device)
        target = target.to(device)
        if model.reverse_complemet_mode:
            target_2 = torch.randn(int(target.shape[0] / 2), 1)
            for i in range(target_2.shape[0]):
                target_2[i] = target[2 * i]
            target = target_2.to(device)

        output = model(data)
        pred_sig = torch.sigmoid(output)
        pred = pred_sig.cpu().detach().numpy().reshape(output.shape[0])
        labels = target.cpu().numpy().reshape(output.shape[0])

        auc.append(metrics.roc_auc_score(labels, pred))

        pred_binary = (pred >= 0.5).astype(int)

        tn, fp, fn, tp = metrics.confusion_matrix(labels, pred_binary).ravel()

        acc.append((tp + tn) / (tp + tn + fp + fn))
        specificity.append(tn / (tn + fp))
        sensitivity.append(tp / (tp + fn))

    AUC_test = np.mean(auc)
    ACC_test = np.mean(acc)
    SPEC_test = np.mean(specificity)
    SENS_test = np.mean(sensitivity)

    print('AUC on test data = ', AUC_test)
    print('Accuracy on test data = ', ACC_test)
    print('Specificity on test data = ', SPEC_test)
    print('Sensitivity on test data = ', SENS_test)
