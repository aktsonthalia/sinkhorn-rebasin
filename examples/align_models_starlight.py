import torch
from models.vgg import VGG
import sys
import copy
sys.path.append('..')
sys.path.append('../..') 
from dataloaders import datasets_dict
from sinkhorn_rebasin.loss import RndLoss, DistL1Loss
from sinkhorn_rebasin import RebasinNet
from copy import deepcopy
from datasets.classification import MNistDataset, SmallMNistDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import train, lerp, eval_loss_acc
from time import time
import wandb
from utils.utils import load_model_from_wandb_id

ENTITY = "mode-connect"
PROJECT = "star-domain"
WANDB1 = "xt1fokcv"
WANDB2 = "2ihj5vqx"

# this code is similar to experiment 2 of our paper
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device == torch.device("cpu"):
    print("Consider using GPU, if available, for a significant speed up.")

# preparing dataset

dataset_train, dataset_val, dataset_test = datasets_dict["cifar10"]()
# model A - trained on Mnist
modelA = VGG("VGG11", in_channels=3, out_features=10, h_in=32, w_in=32)
print("\nLoading network A")
modelA.load_state_dict(load_model_from_wandb_id(ENTITY, PROJECT, WANDB1))

loss, acc = eval_loss_acc(modelA, dataset_test, torch.nn.CrossEntropyLoss(), device)
print("Model A: test loss {:1.3f}, test accuracy {:1.3f}".format(loss, acc))
modelA.eval()

modelB = VGG("VGG11", in_channels=3, out_features=10, h_in=32, w_in=32)
print("\nLoading network B")
modelB.load_state_dict(load_model_from_wandb_id(ENTITY, PROJECT, WANDB2))

loss, acc = eval_loss_acc(modelB, dataset_test, torch.nn.CrossEntropyLoss(), device)
print("Model B: test loss {:1.3f}, test accuracy {:1.3f}".format(loss, acc))
modelB.eval()

# rebasin network for model A
pi_modelA = RebasinNet(copy.deepcopy(modelA), input_shape=(1, 3, 32, 32))
pi_modelA.identity_init()
pi_modelA.to(device)
pi_modelA.train()
print("\nMaking sure we initialize the permutation matrices to I")
print(pi_modelA.p[0].data.clone().cpu().numpy().astype("uint8"))
print("\n")

# rand point loss
criterion =  DistL1Loss(modelB)

# optimizer for rebasin network
optimizer = torch.optim.AdamW(pi_modelA.p.parameters(), lr=0.1)

t1 = time()
print("training rebasin network")
for iteration in range(50):
    # training step
    pi_modelA.train()  # this uses soft permutation matrices
    rebased_model = pi_modelA()
    loss_training = criterion(rebased_model)  # this compared rebased_model with modelB

    optimizer.zero_grad()
    loss_training.backward()
    optimizer.step()  # only updates the permutation matrices

    # validation step
    pi_modelA.eval()  # this uses hard permutation matrices
    rebased_model = pi_modelA()
    loss_validation = criterion(rebased_model)
    print(
        "Iteration {:02d}: loss training {:1.3f}, loss validation {:1.3f}".format(
            iteration, loss_training, loss_validation
        )
    )
    if loss_validation == 0:
        break

print("Elapsed time {:1.3f} secs".format(time() - t1))

pi_modelA.update_batchnorm(modelA)
pi_modelA.eval()
rebased_model = deepcopy(pi_modelA())
rebased_model.eval()

lambdas = torch.linspace(0, 1, 50)
costs_naive = torch.zeros_like(lambdas)
costs_lmc = torch.zeros_like(lambdas)
acc_naive = torch.zeros_like(lambdas)
acc_lmc = torch.zeros_like(lambdas)

print("\nComputing interpolation for LMC visualization")
for i in tqdm(range(lambdas.shape[0])):
    l = lambdas[i]

    temporal_model = lerp(rebased_model, modelB, l)

    # temporal_model.train()
    # for x, y in dataset_train:
    #     x = x.cuda()
    #     _ = temporal_model(x)
    # temporal_model.eval()

    costs_lmc[i], acc_lmc[i] = eval_loss_acc(
        temporal_model, dataset_test, torch.nn.CrossEntropyLoss(), device
    )

    temporal_model = lerp(modelA, modelB, l)
    # temporal_model.train()
    # for x, y in dataset_train:
    #     x = x.cuda()
    #     _ = temporal_model(x)
    # temporal_model.eval()
    
    costs_naive[i], acc_naive[i] = eval_loss_acc(
        temporal_model, dataset_test, torch.nn.CrossEntropyLoss(), device
    )

    print(i, acc_lmc[i], acc_naive[i])

plt.figure()
plt.plot(lambdas, costs_naive, label="Naive")
plt.plot(lambdas, costs_lmc, label="Sinkhorn Re-basin")
plt.title("Loss")
plt.xticks([0, 1], ["ModelA", "ModelB"])
plt.legend()
# plt.show()
plt.savefig("lmc_cnn_loss.png")

plt.figure()
plt.plot(lambdas, acc_naive, label="Naive")
plt.plot(lambdas, acc_lmc, label="Sinkhorn Re-basin")
plt.title("Accuracy")
plt.xticks([0, 1], ["ModelA", "ModelB"])
plt.legend()
# plt.show()
plt.savefig("lmc_cnn_accuracy.png")

print("LMC for VGG!")
