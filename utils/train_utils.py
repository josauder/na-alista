from utils.optimize_matrices import get_matrices
import torch
import torch.nn as nn
import numpy as np
import os
import pandas as pd
from utils.get_data import Synthetic
from utils.algorithms import ISTA, FISTA
from time import time

import utils.conf as conf

device = conf.device


def train_model(m, n, s, k, p, model_fn, noise_fn, epochs, initial_lr, name, model_dir='res/models/',matrix_dir='res/matrices/'):
    if not os.path.exists(model_dir + name):
        os.makedirs(model_dir + name)

    if os.path.isfile(model_dir + name + "/train_log"):
        print("Results for " + name + " are already available. Skipping computation...")
        return None

    phi, W_frob = get_matrices(m, n, matrix_dir=matrix_dir)

    L = np.max(np.linalg.eigvals(np.dot(phi, phi.T)).astype(np.float32))
    phi = torch.Tensor(phi).to(device)
    W_frob = torch.Tensor(phi).to(device)
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
            model.save(model_dir+ name + "/checkpoint")

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


def train_one_epoch(model, loader, noise_fn, opt):
    train_loss = 0
    train_normalizer = 0
    for i, (X, info) in enumerate(loader):
        X = X.to(device)
        info = info.to(device)
        opt.zero_grad()
        y = model.forward_op(X)
        X_hat, gammas, thetas = model(noise_fn(y), info)
        loss = ((X_hat - X) ** 2).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        opt.step()
        train_normalizer += (X ** 2).mean().item()
        train_loss += loss.item()
    return train_loss / len(loader), 10 * np.log10(train_loss / train_normalizer)


def test_one_epoch(model, loader, noise_fn):
    test_loss = 0
    test_normalizer = 0
    with torch.no_grad():
        for i, (X, info) in enumerate(loader):
            X = X.to(device)
            info = info.to(device)
            y = model.forward_op(X)
            X_hat, gammas, thetas = model(noise_fn(y), info)
            test_loss += ((X_hat - X) ** 2).mean().item()
            test_normalizer += (X ** 2).mean().item()
    return test_loss / len(loader), 10 * np.log10(test_loss / test_normalizer)


def evaluate_model(m, n, s, k, p, model_fn, noise_fn, name, model_dir='res/models/'):
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
