import os
import json
import math
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
'''
Define pretrain module for generator
'''
def pretrain_gen_epoch(model, dataloader, criterion, optimizer, model_path, use_cuda=False):
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

def pretest_gen_epoch(model, dataloader, criterion, optimizer, model_path, use_cuda=False):
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

def pretrain_gen(generator, train_loader, test_loader, gen_criterion, gen_optimizer, GEN_PATH, USE_CUDA, PRE_EPOCH_NUM):
    print('[INFO] Pretrain generator with MLE ...')
    train_loss_list = []
    test_loss_list = []
    for epoch in range(PRE_EPOCH_NUM):
        print('[INFO] Start epoch [%d] ...'% (epoch))
        train_loss = pretrain_gen_epoch(generator, train_loader, gen_criterion, gen_optimizer, GEN_PATH, USE_CUDA)
        test_loss = pretest_gen_epoch(generator, test_loader, gen_criterion, gen_optimizer, GEN_PATH, USE_CUDA)
        print('[INFO] End epoch [%d], train Loss: %.4f, test loss: %.4f'% (epoch, train_loss, test_loss))
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
    # plot results
    plt.plot(train_loss_list)
    plt.plot(test_loss_list)
    plt.legend(["train", "test"])
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("pretrained generator (training loss)")
    plt.savefig("../result/pretrained_generator_loss.png")

'''
Define pretrain module for adversary
'''
def pretrain_adv_epoch(model, dataloader, criterion, optimizer, model_path, use_cuda=False):
    total_loss = 0.0
    total_acc = 0.0
    for batch in tqdm(dataloader):
        data, target = batch['x'], batch['u'].squeeze()
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        pred = model.forward(data)
        loss = criterion(pred, target)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, pred_ = torch.max(pred, axis=-1)
        total_acc += (pred==target).sum().item() / target.shape[0]
    torch.save(model.state_dict(), model_path)
    return total_loss, total_acc

def pretest_adv_epoch(model, dataloader, criterion, optimizer, model_path, use_cuda=False):
    total_loss = 0.0
    total_acc = 0.0
    for batch in tqdm(dataloader):
        data, target = batch['x'], batch['u'].squeeze()
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            pred = model.forward(data)
            loss = criterion(pred, target)
            total_loss += loss.item()
            _, pred_ = torch.max(pred, axis=-1)
            total_acc += (pred==target).sum().item() / target.shape[0]
    torch.save(model.state_dict(), model_path)
    return total_loss, total_acc

def pretrain_adv(adversary, train_loader, test_loader, adv_criterion, adv_optimizer, ADV_PATH, USE_CUDA, PRE_EPOCH_NUM):
    print('[INFO] Pretrain adversary with CNN ...')
    train_loss_list = []
    test_loss_list = []
    train_acc_list = []
    test_acc_list = []
    for epoch in range(PRE_EPOCH_NUM):
        print('[INFO] Start epoch [%d] ...'% (epoch))
        train_loss, train_acc = pretrain_adv_epoch(adversary, train_loader, gen_criterion, gen_optimizer, ADV_PATH, USE_CUDA)
        test_loss, test_acc = pretest_adv_epoch(adversary, test_loader, gen_criterion, gen_optimizer, ADV_PATH, USE_CUDA)
        print('[INFO] End epoch [%d], \
                      loss (train, test): (%.4f, %.4f), \
                      accuracy (train, test): (%.4f, %.4f)'% \
                      (epoch, train_loss, test_loss, train_acc, test_acc))
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
    # plot results
    fig, ax = plt.subplots(1, 2, figsize=(14,5))
    ax[0].plot(train_loss_list)
    ax[0].plot(test_loss_list)
    ax[0].legend(["train", "test"])
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("loss")
    ax[0].set_title("pretrained generator (training loss)")
    ax[1].plot(train_acc_list)
    ax[1].plot(test_acc_list)
    ax[1].legend(["train", "test"])
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("accuracy")
    ax[1].set_title("pretrained generator (training accuracy)")
    fig.savefig("../result/pretrained_adversary_result.png")