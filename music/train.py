# from logger import Logger
import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary

from music.torch_model import nn_model


# config device


class TrainDataset(Dataset):
    def __init__(self):
        self.x = torch.from_numpy(np.load("Training_Data/train_x.npy").astype(np.float32))
        self.y = torch.from_numpy(
            np.load("Training_Data/train_y.npy").reshape(-1, 1).astype(np.float32)
        )
        self.n_samples = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


class TestDataset(Dataset):
    def __init__(self):
        self.x = torch.from_numpy(np.load("Training_Data/test_x.npy").astype(np.float32))
        self.y = torch.from_numpy(
            np.load("Training_Data/test_y.npy").reshape(-1, 1).astype(np.float32)
        )
        self.n_samples = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


def adjust_learning_rate(optimizer, epoch, learning_rate):
    """Sets the learning rate to the initial LR decayed by 10 every 12 epochs"""
    lr = learning_rate * (0.1 ** (epoch // 12))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
        print(param_group["lr"])


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 16
    num_classes = 8
    batch_size = 128
    learning_rate = 0.0001
    trainDataset = TrainDataset()
    train_loader = DataLoader(
        dataset=trainDataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    testDataset = TestDataset()
    test_loader = DataLoader(dataset=testDataset, batch_size=batch_size, shuffle=True)
    # Initialize model
    model = nn_model.to(device)
    summary(model, input_size=(1, 128, 128))
    # logger = Logger('./logs')
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    total_samples = len(trainDataset)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    n_iterations = math.ceil(total_samples / batch_size)
    # Train the model
    best_accuracy = 0
    model.train()  # Set model to train mode
    for epoch in range(num_epochs):
        adjust_learning_rate(optimizer, epoch, learning_rate)
        for i, (images, labels) in enumerate(train_loader):
            model.train()
            images = (images / 255.0).reshape(-1, 1, 128, 128).to(device)
            labels = (
                F.one_hot(labels.type(torch.LongTensor), num_classes)
                .reshape(-1, num_classes)
                .to(device)
            )

            # Forward pass
            outputs = model(images)

            loss = criterion(outputs, torch.max(labels, 1)[1])

            # Backward and optimize
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Compute accuracy
            _, argmax = torch.max(outputs, 1)
            n = argmax.size(0)
            accuracy = (torch.max(labels, 1)[1] == argmax).sum().item() / n

            if accuracy > best_accuracy:
                best_model = copy.deepcopy(model)

            if (i + 1) % 100 == 0:
                print(
                    f"epoch: {epoch+1}/{num_epochs}, step: {i+1}/{n_iterations}, loss: {loss.item():.4f}, accuracy: {accuracy:.4f}"
                )
                # print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))
                # ================================================================== #
                #                        Tensorboard Logging                         #
                # ================================================================== #

                # place for logs
                # 1. Log scalar values (scalar summary)
                # 2. Log values and gradients of the parameters (histogram summary)
                # 3. Log training images (image summary)

    # Test the model
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = (images / 255.0).reshape(-1, 1, 128, 128).to(device)
            labels = (
                F.one_hot(labels.type(torch.LongTensor), num_classes)
                .reshape(-1, num_classes)
                .to(device)
            )
            out = model(images)
            _, predicted = torch.max(out.data, 1)
            total += labels.size(0)
            correct += (predicted == torch.max(labels, 1)[1]).sum().item()

        print("Test Accuracy: {:.4f} %".format(100 * correct / total))

    # Save the best model checkpoint
    torch.save(best_model.state_dict(), "model.pt")
