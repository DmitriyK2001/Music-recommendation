import torch
from model import model
from torchtext.data import BucketIterator

def train_model(model, iterator, optimizer, criterion, clip, train_history=None, valid_history=None):
    model.train()

    epoch_loss = 0
    history = []
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg
        optimizer.zero_grad()
        output = model(src, trg)
        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

        history.append(loss.cpu().data.numpy())


    return epoch_loss / len(iterator)

def train(num_layes, hid_dim):
    model = model(num_layers, hid_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
    critetion = torch.nn.CrossEntropyLoss()
    CLIP = 1
    iterator = BucketIterator(data, BATCH_SIZE = 128)
    train_model(model, iterator, optimizer, critetion, CLIP)
