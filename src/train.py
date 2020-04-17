import os
import json
import math
import torch
from tqdm import tqdm

def pretrain_epoch(model, dataloader, criterion, optimizer, model_path, use_cuda=False):
    total_loss = 0.
    total_words = 0.
    for batch in tqdm(dataloader):
        data = batch["x"]
        target = batch["x"][:,:,0]
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        target = target.contiguous().view(-1)
        pred = model.forward(data)
        loss = criterion(pred, target)
        total_loss += loss.item()
        total_words += data.size(0) * data.size(1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    target_ = target.detach().cpu().numpy()
    _, pred_ = torch.max(pred, axis=-1)
    pred_ = pred_.cpu().numpy()
    target_query = []
    pred_query = []
    for i in range(72):
        target_query.append(index_map[target_[i]])
        pred_query.append(index_map[pred_[i]])
    print("[INFO] Target query: ", target_query)
    print("[INFO] Predicted query: ", pred_query)
    torch.save(model.state_dict(), model_path)
    return math.exp(total_loss / total_words)

def pretest_epoch(model, dataloader, criterion, optimizer, model_path, use_cuda=False):
    total_loss = 0.
    total_words = 0.
    for batch in tqdm(dataloader):
        data = batch["x"]
        target = batch["x"][:,:,0]
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        target = target.contiguous().view(-1)
        with torch.no_grad():
            pred = model.forward(data)
            loss = criterion(pred, target)
            total_loss += loss.item()
            total_words += data.size(0) * data.size(1)
    target_ = target.detach().cpu().numpy()
    _, pred_ = torch.max(pred, axis=-1)
    pred_ = pred_.cpu().numpy()
    target_query = []
    pred_query = []
    for i in range(72):
        target_query.append(index_map[target_[i]])
        pred_query.append(index_map[pred_[i]])
    print("[INFO] Target query: ", target_query)
    print("[INFO] Predicted query: ", pred_query)
    torch.save(model.state_dict(), model_path)
    return math.exp(total_loss / total_words)

def pretrain_G(generator, train_loader, test_loader, gen_criterion, gen_optimizer, G_path, USE_CUDA, PRE_EPOCH_NUM):
    print('[INFO] Pretrain generator with MLE ...')
    train_loss_list = []
    test_loss_list = []
    for epoch in range(PRE_EPOCH_NUM):
        print('[INFO] Start epoch [%d] ...'% (epoch))
        train_loss = pretrain_epoch(generator, train_loader, gen_criterion, gen_optimizer, G_path, USE_CUDA)
        test_loss = pretest_epoch(generator, test_loader, gen_criterion, gen_optimizer, G_path, USE_CUDA)
        print('[INFO] End epoch [%d], train Loss: %.4f, test loss: %.4f'% (epoch, train_loss, test_loss))
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)

    import matplotlib.pyplot as plt
    plt.plot(train_loss_list)
    plt.plot(test_loss_list)
    plt.legend(["train", "test"])
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("pretrained generator (training loss)")
    plt.savefig("../result/pretrained_generator_loss.png")
