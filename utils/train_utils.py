from utils.optimize_matrices import get_matrices
import torch
import torch.nn as nn
import numpy as np
import os
import pandas as pd
from utils.get_data import Synthetic
from utils.algorithms import ISTA, FISTA
from time import time
from utils.wavelet import WT
from utils.fft import fft2, ifft2
import utils.conf as conf
from torchvision.transforms import Grayscale, ToTensor, Compose, RandomVerticalFlip, RandomHorizontalFlip, Resize, \
    RandomAffine, CenterCrop, RandomResizedCrop
from torchvision import datasets
from torch.utils.data import DataLoader
from utils.algorithms import soft_threshold

device = conf.device


def train_model(m, n, s, k, p, model_fn, noise_fn, epochs, initial_lr, name, model_dir='res/models/',
                matrix_dir='res/matrices/'):
    if not os.path.exists(model_dir + name):
        os.makedirs(model_dir + name)

    if os.path.isfile(model_dir + name + "/train_log"):
        print("Results for " + name + " are already available. Skipping computation...")
        return None

    phi, W_frob = get_matrices(m, n, matrix_dir=matrix_dir)

    L = np.max(np.linalg.eigvals(np.dot(phi, phi.T)).astype(np.float32))
    phi = torch.Tensor(phi).to(device)
    W_frob = torch.Tensor(W_frob).to(device)
    forward_op = lambda x: torch.matmul(phi, x.T).T
    backward_op = lambda x: torch.matmul(W_frob.T, x.T).T

    data = Synthetic(m, n, s, s)
    # put W_frob = reverse tv norm operator ...
    model = model_fn(m, n, s, k, p, forward_op, backward_op, L).to(device)

    if type(model) not in [ISTA, FISTA]:
        opt = torch.optim.Adam(model.parameters(), lr=initial_lr)
    else:
        model.backward_op = lambda x: torch.matmul(phi.T, x.T).T

    train_losses = []
    train_dbs = []
    test_losses = []
    test_dbs = []

    if type(model) in [ISTA, FISTA]:
        epochs = 1
    for i in range(epochs):
        if type(model) not in [ISTA, FISTA]:
            train_loss, train_db = train_one_epoch(model, data.train_loader, noise_fn, opt)
        else:
            train_loss = 0
            train_db = 0
        test_loss, test_db = test_one_epoch(model, data.test_loader, noise_fn)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_dbs.append(train_db)
        test_dbs.append(test_db)

        if test_dbs[-1] == min(test_dbs) and type(model) not in [ISTA, FISTA]:
            print("saving!")
            model.save(model_dir + name + "/checkpoint")

        data.train_data.reset()

        print(i, train_db, test_db)

    print("saving results to " + model_dir + name + "/train_log")
    pd.DataFrame(
        {
            "epoch": range(epochs),
            "train_loss": train_losses,
            "test_loss": test_losses,
            "train_dbs": train_dbs,
            "test_dbs": test_dbs,
        }
    ).to_csv(model_dir + name + "/train_log")


def train_model(m, n, s, k, p, model_fn, noise_fn, epochs, initial_lr, name, model_dir='res/models/',
                matrix_dir='res/matrices/'):
    if not os.path.exists(model_dir + name):
        os.makedirs(model_dir + name)

    if os.path.isfile(model_dir + name + "/train_log"):
        print("Results for " + name + " are already available. Skipping computation...")
        return None

    torch.manual_seed(3)
    P_omega = torch.zeros((128, 128))
    P_omega[(torch.randn(64) * 21 + 64).long().clamp(0, 127)] = 1
    print(P_omega.sum()/128)

    import matplotlib.pyplot as plt
    plt.imshow(P_omega)
    plt.show()
    P_omega = P_omega.to(device)

    L = 1
    # fft2 = lambda x: x
    # ifft2 = lambda x: x
    wavelet = WT()
    forward_op = lambda x: P_omega * fft2(wavelet.iwt(x))
    backward_op = lambda x: wavelet.wt(ifft2(P_omega * x))

    train_transform = Compose([Grayscale(), RandomAffine((0, 0), translate=(0, 0.1), scale=(0.8, 1.2)),
                               RandomResizedCrop((128, 128), scale=(1.0, 1.0)), RandomHorizontalFlip(),
                               RandomVerticalFlip(), ToTensor()])
    # train_dataset = datasets.CIFAR100('.', train=True, transform=train_transform, target_transform=None, download=True)

    # for i in os.listdir('realdata/yes'):
    #    os.system("mv realdata/yes/" + str(i.replace(" ", "\\ ")) + " realdata/yes/" + str(i.replace(" ", "").lower()))
    # for i in os.listdir('realdata/no'):
    #    os.system("mv realdata/no/" + str(i.replace(" ", "\\ ")) + " realdata/no/" + str(i.replace(" ", "").lower()))

    train_dataset = datasets.ImageFolder('realdata/train', transform=train_transform)
    test_transform = Compose([Resize(128), CenterCrop(128), Grayscale(), ToTensor()])
    test_dataset = datasets.ImageFolder('realdata/test', transform=test_transform)
    # test_dataset = datasets.CIFAR100('.', train=False, transform=test_transform, target_transform=None, download=True)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    n = 64 * 64

    model = model_fn(m, n, s, k, p, forward_op, backward_op, L).to(device)

    if type(model) not in [ISTA, FISTA]:
        opt = torch.optim.Adam(model.parameters(), lr=initial_lr)

    train_losses = []
    train_dbs = []
    test_losses = []
    test_dbs = []

    if type(model) in [ISTA, FISTA]:
        epochs = 1
    for i in range(epochs):
        if type(model) not in [ISTA, FISTA]:
            train_loss, train_db = train_one_epoch(model, train_loader, noise_fn, opt, transform=wavelet)
        else:
            train_loss = 0
            train_db = 0
        test_loss, test_db = test_one_epoch(model, test_loader, noise_fn, transform=wavelet)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_dbs.append(train_db)
        test_dbs.append(test_db)

        if test_dbs[-1] == min(test_dbs) and type(model) not in [ISTA, FISTA]:
            print("saving!")
            model.save(model_dir + name + "/checkpoint")

        print(i, train_db, test_db)

    print("saving results to " + model_dir + name + "/train_log")
    pd.DataFrame(
        {
            "epoch": range(epochs),
            "train_loss": train_losses,
            "test_loss": test_losses,
            "train_dbs": train_dbs,
            "test_dbs": test_dbs,
        }
    ).to_csv(model_dir + name + "/train_log")


def train_one_epoch(model, loader, noise_fn, opt, transform=None):
    train_loss = 0
    train_normalizer = 0
    for i, (X, info) in enumerate(loader):
        X = X.to(device)

        if transform is not None:
            # if i == 0:
            #  import matplotlib.pyplot as plt
            #  fig, ax = plt.subplots(2, 2, figsize=(10, 10))
            #  ax[0, 0].imshow(X[0, 0].detach().cpu().numpy())
            X = transform.wt(X)
            # Xnorm = torch.norm(X.reshape(X.shape[0], -1), dim=1).reshape(X.shape[0], 1, 1, 1)
            # X = X/Xnorm

            # if i == 0:
            #  ax[0,1].imshow(transform.iwt(model.backward_op(model.forward_op(X)))[0,0].detach().cpu().numpy())

        info = info.to(device)
        opt.zero_grad()

        y = model.forward_op(X)
        X_hat, gammas, thetas = model(noise_fn(y), info)

        # Xhatnorm = torch.norm(X.reshape(X.shape[0], -1), dim=1).reshape(X.shape[0], 1, 1, 1)
        if transform is not None:
            loss = ((transform.iwt(X_hat) - transform.iwt(X)) ** 2).mean()
        else:
            loss = ((X_hat - X) ** 2).mean()
        loss.backward()

        # if transform is not None:
        # if i == 0:
        #  ax[1, 1].imshow(X_hat[0][0].detach().cpu().numpy())
        #  ax[1,0].imshow(transform.iwt(X_hat)[0][0].detach().cpu().numpy())
        #  plt.show()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        opt.step()
        train_normalizer += (X ** 2).mean().item()
        train_loss += loss.item()
    return train_loss / len(loader), 10 * np.log10(train_loss / train_normalizer)


def test_one_epoch(model, loader, noise_fn, transform=None):
    test_loss = 0
    test_loss_no_recon = 0
    test_normalizer = 0
    with torch.no_grad():
        for i, (X, info) in enumerate(loader):
            X = X.to(device)
            if transform is not None:
                if i == 0:
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(3, 2, figsize=(10, 10))
                    ax[0, 0].imshow(X[0, 0].detach().cpu().numpy())
                X = transform.wt(X)
                if i == 0:
                    import matplotlib.pyplot as plt
                    ax[0, 1].imshow(X[0, 0].detach().cpu().numpy())

                # Xnorm = torch.norm(X.reshape(X.shape[0], -1), dim=1).reshape(X.shape[0], 1, 1, 1)
                # X = X/Xnorm
            info = info.to(device)
            y = model.forward_op(X)
            if i == 0:
                import matplotlib.pyplot as plt
                ax[1, 0].imshow(torch.sqrt(torch.abs(y))[0, 0].detach().cpu().numpy())
                ax[1, 1].imshow(transform.iwt(model.backward_op(y))[0, 0].detach().cpu().numpy())

            X_hat, gammas, thetas = model(noise_fn(y), info)

            Xbackward = model.backward_op(y)
            # Xbackwardnorm = torch.norm(Xbackward.reshape(X.shape[0], -1), dim=1).reshape(X.shape[0], 1, 1, 1) / 0.01
            # Xhatnorm = torch.norm(X.reshape(X.shape[0], -1), dim=1).reshape(X.shape[0], 1, 1, 1) / 0.01
            if transform is not None:
                if i == 0:
                    ax[2, 0].imshow(X_hat[0][0].detach().cpu().numpy())
                    ax[2, 1].imshow(transform.iwt(X_hat)[0][0].detach().cpu().numpy())
                    plt.show()

            if transform is not None:
                test_loss += ((transform.iwt(X_hat) - transform.iwt(X)) ** 2).mean().item()
                test_loss_no_recon += ((transform.iwt(X) - transform.iwt(Xbackward)) ** 2).mean().item()
            else:
                test_loss += ((X_hat - X) ** 2).mean().item()
            test_normalizer += (X ** 2).mean().item()
    print("NO RECON:", 10 * np.log10(test_loss_no_recon / test_normalizer))
    return test_loss / len(loader), 10 * np.log10(test_loss / test_normalizer)


def evaluate_model(m, n, s, k, p, model_fn, noise_fn, name, model_dir='res/models/', transform=None):
    phi, W_soft_gen, W_frob = get_matrices(m, n)
    data = Synthetic(m, n, s, s)
    model = model_fn(m, n, s, k, p, phi, W_soft_gen, W_frob).to(device)
    model.load(model_dir + name + "/checkpoint")

    test_loss = []
    test_normalizer = []
    sparsities = []
    t1 = time()
    with torch.no_grad():
        for epoch in range(1):
            for i, (X, info) in enumerate(data.train_loader):
                if i == 0:
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
                    ax[0, 0].imshow(X[0, 0].detach().cpu().numpy())

                sparsities.extend(list((X != 0).int().sum(dim=1).detach().numpy()))
                X = X.to(device)

                info = info.to(device)
                y = model.forward_op(X)
                X_hat, gammas, thetas = model(noise_fn(y), info)

                test_loss.extend(list(((X_hat - X) ** 2).cpu().detach().numpy()))
                test_normalizer.extend(list((X ** 2).cpu().detach().numpy()))
            data.train_data.reset()
    t2 = time()
    runtime_evaluation = t2 - t1

    test_loss = np.array(test_loss)
    test_normalizer = np.array(test_normalizer)
    sparsities = np.array(sparsities)

    keys = []
    counts = []
    values = []
    for s in sorted(np.unique(sparsities)):
        count = (sparsities == s).mean()
        if count > 10e-5:
            keys.append(s)
            counts.append(count)
            values.append(
                10
                * np.log10(
                    np.sum(test_loss[sparsities == s]) / np.sum(test_normalizer[sparsities == s])
                )
            )

    return keys, counts, values, 10 * np.log10(np.sum(test_loss) / np.sum(test_normalizer))
