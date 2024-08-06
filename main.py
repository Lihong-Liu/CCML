import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from data import Scene, PIE, CUB, CNIST, Cal101_20
from loss_function import get_loss
from model import CCML

import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)

def calculate_mean_and_std(numbers):
    mean = sum(numbers) / len(numbers)
    variance = sum((x - mean) ** 2 for x in numbers) / len(numbers)
    import math
    std = math.sqrt(variance)
    return mean, std


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=200, metavar='N',
                        help='input batch size for training [default: 100]')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of epochs to train [default: 500]')
    parser.add_argument('--annealing_step', type=int, default=50, metavar='N',
                        help='gradually increase the value of lambda from 0 to 1')
    parser.add_argument('--lr', type=float, default=0.003, metavar='LR',
                        help='learning rate')
    args = parser.parse_args()

    dataset = PIE()
    num_samples = len(dataset)
    num_classes = dataset.num_classes
    num_views = dataset.num_views
    dims = dataset.dims


    delta = 1
    gamma = 1
    beta = 1.25

    test_time = 5

    evidence_vir = []
    results = []
    for test_t in range(test_time):

        index = np.arange(num_samples)
        np.random.shuffle(index)
        train_index, test_index = index[:int(0.8 * num_samples)], index[int(0.8 * num_samples):]
        train_loader = DataLoader(Subset(dataset, train_index), batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(Subset(dataset, test_index), batch_size=args.batch_size, shuffle=False)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = CCML(num_views, dims, num_classes, device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        model.to(device)

        print(f'{test_t+1}: training...')

        model.train()
        for epoch in range(1, args.epochs + 1):
            if epoch % (args.epochs/10) == 0:
                print(f'====> {epoch}')
            for X, Y, indexes in train_loader:
                for v in range(num_views):
                    X[v] = X[v].to(device)
                Y = Y.to(device)
                evidences, evidence_a, evidence_con, evidence_div = model(X, beta)
                loss = get_loss(evidences, evidence_a, evidence_con, evidence_div, Y, epoch, num_classes, args.annealing_step, gamma, delta, device)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()
        num_correct, num_sample = 0, 0
        for X, Y, indexes in test_loader:
            for v in range(num_views):
                X[v] = X[v].to(device)
            Y = Y.to(device)
            with torch.no_grad():
                evidences, evidence_a, evidence_con, evidence_div = model(X, beta)
                _, Y_pre = torch.max(evidence_a, dim=1)
                num_correct += (Y_pre == Y).sum().item()
                num_sample += Y.shape[0]
                evidence_vir.append(torch.mean(evidence_con).item())

        print('====> acc: {:.4f}'.format(num_correct / num_sample))
        results.append(num_correct / num_sample)

    mean, stdDev = calculate_mean_and_std(results)
    print('===========================================')
    print(f'{test_time} times test finish.')
    print('delta = {:.2f}'.format(delta))
    print('gamma = {:.2f}'.format(gamma))
    print('beta = {:.2f}'.format(beta))

    print(results)
    print('mean = {:.4f}'.format(mean))
    print('std = {:.4f}'.format(stdDev))


