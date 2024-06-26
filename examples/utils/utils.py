import torch
from copy import deepcopy
import time
import os
import wandb


def lerp(model1, model2, l, temporal_model=None):
    if temporal_model is None:
        temporal_model = deepcopy(model1)
    for p, p1, p2 in zip(
        temporal_model.parameters(), model1.parameters(), model2.parameters()
    ):
        p.data.copy_((1 - l) * p1.data + l * p2.data)

    for m, m1, m2 in zip(temporal_model.modules(), model1.modules(), model2.modules()):
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean = None
            m.running_var = None
            m.track_running_stats = False

    return temporal_model


def eval(model, dataset, criterion, device):
    model.to(device)
    cumulative_test_loss = 0
    total_test = 0
    model.eval()
    for x, y in dataset:
        z = model(x.to(device))
        loss_test = criterion(z, y.to(device))

        cumulative_test_loss += loss_test.item() * x.shape[0]
        total_test += x.shape[0]

    cumulative_test_loss /= total_test
    return cumulative_test_loss


def eval_loss_acc(model, dataset, criterion, device):
    model.to(device)
    cumulative_test_loss = 0
    cumulative_test_acc = 0
    total_test = 0
    model.eval()
    param_precision = next(iter(model.parameters())).data.dtype
    for x, y in dataset:
        z, loss = estep(x, y, model, criterion, device, param_precision)
        acc_test = sum([1 if y[i] == z[i] else 0 for i in range(y.shape[0])])

        cumulative_test_loss += loss * x.shape[0]
        cumulative_test_acc += acc_test
        total_test += x.shape[0]

    cumulative_test_loss /= total_test
    cumulative_test_acc /= total_test
    return cumulative_test_loss, cumulative_test_acc


def estep(x, y, model, criterion, device, param_precision):
    z = model(x.to(device).to(param_precision))
    loss_test = criterion(z, y.to(device))
    return z.detach().argmax(1), loss_test.detach().item()

def load_model_from_wandb_id(entity, project, wandb_id):

    for _ in range(10):
        try:
            api = wandb.Api()
            run = api.run(f"{entity}/{project}/{wandb_id}")
            checkpoints = [
                artifact
                for artifact in run.logged_artifacts()
                if artifact.type == "model-weights"
            ][0]
            num_epochs = run.config["training"]["num_epochs"]
            last_epoch_file = [
                f
                for f in checkpoints.files()
                if f.name == f"{wandb_id}_checkpoint{num_epochs-1}.pt"
            ][0]
            download_path = last_epoch_file.download(
                replace=True, root=os.environ["SCRATCH"] if "SCRATCH" in os.environ else "/workspace"
            ).name
            state_dict = torch.load(download_path)
            return state_dict
        except:
            print("Failed to download model, trying again...")
            time.sleep(10)
            continue
