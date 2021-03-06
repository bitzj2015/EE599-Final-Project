import os
import csv
import json
import math
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from rollout_mp import Rollout
from utils import GANLoss, CumReward
import numpy as np
'''
Define pretrain module for generator
'''
def pretrain_gen_epoch(model, 
                       dataloader, 
                       criterion, 
                       optimizer, 
                       index_map,
                       model_path, 
                       MAX_SEQ_LEN,
                       USE_CUDA=False):
    total_loss = 0.
    total_words = 0.
    for batch in tqdm(dataloader):
        data = batch["x"]
        target = batch["x"][:,:,0]
        if USE_CUDA:
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
    for i in range(MAX_SEQ_LEN):
        target_query.append(index_map[target_[i]])
        pred_query.append(index_map[pred_[i]])
    print("[INFO] Target query: ", target_query)
    print("[INFO] Predicted query: ", pred_query)
    torch.save(model.state_dict(), model_path)
    return math.exp(total_loss / total_words)

def pretest_gen_epoch(model, 
                      dataloader, 
                      criterion, 
                      optimizer, 
                      index_map,
                      model_path, 
                      MAX_SEQ_LEN,
                      USE_CUDA=False):
    total_loss = 0.
    total_words = 0.
    for batch in tqdm(dataloader):
        data = batch["x"]
        target = batch["x"][:,:,0]
        if USE_CUDA:
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
    for i in range(MAX_SEQ_LEN):
        target_query.append(index_map[target_[i]])
        pred_query.append(index_map[pred_[i]])
    print("[INFO] Target query: ", target_query)
    print("[INFO] Predicted query: ", pred_query)
    return math.exp(total_loss / total_words)

def pretrain_gen(generator, 
                 train_loader, 
                 test_loader, 
                 gen_criterion, 
                 gen_optimizer, 
                 index_map,
                 MAX_SEQ_LEN,
                 GEN_PATH, 
                 USE_CUDA, 
                 EPOCH_NUM,
                 PLOT=True):
    print('[INFO] Pretrain generator with MLE ...')
    train_loss_list = []
    test_loss_list = []
    csvFile = open("../param/pretrain_generator.csv", 'a', newline='')
    writer = csv.writer(csvFile)
    writer.writerow(["epoch", "train_loss", "test_loss"])
    csvFile.close()
    for epoch in range(EPOCH_NUM):
        print('[INFO] Start epoch [%d] ...'% (epoch))
        train_loss = pretrain_gen_epoch(generator, 
                                        train_loader, 
                                        gen_criterion, 
                                        gen_optimizer, 
                                        index_map,
                                        GEN_PATH,
                                        MAX_SEQ_LEN,
                                        USE_CUDA)
        test_loss = pretest_gen_epoch(generator, 
                                      test_loader, 
                                      gen_criterion, 
                                      gen_optimizer,
                                      index_map, 
                                      GEN_PATH, 
                                      MAX_SEQ_LEN,
                                      USE_CUDA)
        print('[INFO] End epoch [%d], \
                      train Loss: %.4f, \
                      test loss: %.4f'% \
                      (epoch, train_loss, test_loss))
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        csvFile = open("../param/pretrain_generator.csv", 'a', newline='')
        writer = csv.writer(csvFile)
        writer.writerow([epoch, train_loss, test_loss])
        csvFile.close()
        if epoch > 0:
            for param_group in gen_optimizer.param_groups:
                param_group['lr'] *= 0.95
                print(param_group['lr'])
    if PLOT:
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
def train_adv_epoch(model,
                    generator,
                    dataloader, 
                    criterion, 
                    optimizer, 
                    model_path, 
                    USE_CUDA=False):
    total_loss = 0.0
    total_acc = 0.0
    total_words = 0.0
    count = 0
    for batch in tqdm(dataloader):
        data, target = batch['x'], batch['u'].squeeze()
        # print(data.size(), target.size(0), target)
        if USE_CUDA:
            data, target = data.cuda(), target.cuda()
        if generator != None:
            pred = generator.forward(data)
            _, pred_ = torch.max(pred, axis=-1)
            pred_ = pred_.view(data.size(0), -1)
            samples = torch.stack([pred_, data[:,:,1]], axis=2)
        else:
            samples = data
        _, pred = model.forward(samples)
        loss = criterion(pred, target)
        total_loss += loss.item()
        total_words += data.size(0) * data.size(1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, pred_ = torch.max(pred, axis=-1)
        total_acc += (pred_ == target).sum().item() / target.size(0)
        count += 1
    torch.save(model.state_dict(), model_path)
    return math.exp(total_loss / total_words), total_acc / count

def test_adv_epoch(model, 
                   generator,
                   dataloader, 
                   criterion, 
                   optimizer, 
                   model_path, 
                   USE_CUDA=False):
    total_loss = 0.0
    total_acc = 0.0
    total_words = 0.0
    count = 0
    for batch in tqdm(dataloader):
        data, target = batch['x'], batch['u'].squeeze()
        if USE_CUDA:
            data, target = data.cuda(), target.cuda()
        if generator != None:
            pred = generator.forward(data)
            _, pred_ = torch.max(pred, axis=-1)
            pred_ = pred_.view(data.size(0), -1)
            samples = torch.stack([pred_, data[:,:,1]], axis=2)
        else:
            samples = data
        with torch.no_grad():
            _, pred = model.forward(samples)
            loss = criterion(pred, target)
            total_loss += loss.item()
            total_words += data.size(0) * data.size(1)
            _, pred_ = torch.max(pred, axis=-1)
            total_acc += (pred_ == target).sum().item() / target.size(0)
            count += 1
    return math.exp(total_loss / total_words), total_acc / count

def train_adv(adversary,
              generator,
              train_loader, 
              test_loader, 
              adv_criterion, 
              adv_optimizer, 
              ADV_PATH, 
              USE_CUDA, 
              EPOCH_NUM,
              PHASE="pretrain", 
              PLOT=False):
    print('[INFO] Pretrain adversary with CNN ...')
    train_loss_list = []
    test_loss_list = []
    train_acc_list = []
    test_acc_list = []
    for epoch in range(EPOCH_NUM):
        print('[INFO] Start epoch [%d] ...'% (epoch))
        train_loss, train_acc = train_adv_epoch(adversary,
                                                generator, 
                                                train_loader, 
                                                adv_criterion, 
                                                adv_optimizer, 
                                                ADV_PATH, 
                                                USE_CUDA)
        test_loss, test_acc = test_adv_epoch(adversary,
                                             generator,
                                             test_loader, 
                                             adv_criterion, 
                                             adv_optimizer, 
                                             ADV_PATH, 
                                             USE_CUDA)
        print('[INFO] End epoch [%d], \
                      loss (train, test): (%.4f, %.4f), \
                      accuracy (train, test): (%.4f, %.4f)'% \
                      (epoch, train_loss, test_loss, train_acc, test_acc))
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
    if PLOT:
        # plot results
        fig, ax = plt.subplots(1, 2, figsize=(14,5))
        ax[0].plot(train_loss_list)
        ax[0].plot(test_loss_list)
        ax[0].legend(["train", "test"])
        ax[0].set_xlabel("epoch")
        ax[0].set_ylabel("loss")
        ax[0].set_title(PHASE + " generator (training loss)")
        ax[1].plot(train_acc_list)
        ax[1].plot(test_acc_list)
        ax[1].legend(["train", "test"])
        ax[1].set_xlabel("epoch")
        ax[1].set_ylabel("accuracy")
        ax[1].set_title(PHASE + " generator (training accuracy)")
        fig.savefig("../result/" + PHASE + "_adversary_result.png")

'''
Define train discriminator
'''
def train_dis(discriminator, 
              generator,
              train_loader,
              test_loader,
              dis_criterion,
              dis_optimizer,
              DIS_PATH,
              USE_CUDA, 
              EPOCH_NUM,
              PHASE="pretrain", 
              PLOT=False):
    train_loss_list = []
    train_acc_list = []

    for epoch in range(EPOCH_NUM):
        print('[INFO] Start training epoch [%d] ...'% (epoch))   
        total_loss = 0.0
        total_acc = 0.0
        total_words = 0.0
        count = 0
        for batch in tqdm(train_loader):
            data = batch['x']
            batch_size = data.size(0)
            seq_len = data.size(1)
            label = torch.ones((batch_size, 1)).long()
            label_ = torch.zeros((batch_size, 1)).long()
            if USE_CUDA:
                data = data.cuda()
                label = label.cuda()
                label_ = label_.cuda()
            pred = generator.forward(data)
            _, pred_ = torch.max(pred, axis=-1)
            pred_ = pred_.view(batch_size, -1)
            if pred_.size(1) != seq_len:
                print("[ERR] Check dimension!!!")
            data_ = torch.stack([pred_, data[:, :, 1]], axis=2)
            data_batch = torch.cat([data, data_], axis=0)
            label_batch = torch.cat([label, label_], axis=0)
            label_batch = label_batch.contiguous().view(-1)
            _, pred_batch = discriminator(data_batch)
            loss = dis_criterion(pred_batch, label_batch)
            total_loss += loss.item()
            total_words += data_batch.size(0) * data_batch.size(1)
            dis_optimizer.zero_grad()
            loss.backward()
            dis_optimizer.step()
            _, pred_batch_ = torch.max(pred_batch, axis=-1)
            total_acc += (pred_batch_ == label_batch).sum().item() / (batch_size * 2)
            count += 1
        total_acc = total_acc / count
        total_loss = math.exp(total_loss / total_words)
        train_loss_list.append(total_loss)
        train_acc_list.append(total_acc)
        torch.save(discriminator.state_dict(), DIS_PATH)
        print('[INFO] End training epoch [%d], \
                      loss: %.4f, accuracy: %.4f'% \
                      (epoch, total_loss, total_acc))
    test_loss, test_acc = test_dis(discriminator, 
                                   generator,
                                   test_loader,
                                   dis_criterion,
                                   dis_optimizer,
                                   DIS_PATH,
                                   USE_CUDA)
    return train_loss_list, train_acc_list, test_loss, test_acc

def test_dis(discriminator, 
             generator,
             test_loader,
             dis_criterion,
             dis_optimizer,
             DIS_PATH,
             USE_CUDA):

    print('[INFO] Start testing ...')   
    test_loss = 0.0
    test_acc = 0.0
    test_words = 0.0
    count = 0
    for batch in tqdm(test_loader):
        data = batch['x']
        batch_size = data.size(0)
        seq_len = data.size(1)
        label = torch.ones((batch_size, 1)).long()
        label_ = torch.zeros((batch_size, 1)).long()
        if USE_CUDA:
            data = data.cuda()
            label = label.cuda()
            label_ = label_.cuda()
        pred = generator.forward(data)
        _, pred_ = torch.max(pred, axis=-1)
        pred_ = pred_.view(batch_size, -1)
        if pred_.size(1) != seq_len:
            print("[ERR] Check dimension!!!")
        data_ = torch.stack([pred_, data[:, :, 1]], axis=2)
        data_batch = torch.cat([data, data_], axis=0)
        label_batch = torch.cat([label, label_], axis=0)
        label_batch = label_batch.contiguous().view(-1)
        _, pred_batch = discriminator(data_batch)
        loss = dis_criterion(pred_batch, label_batch)
        test_loss += loss.item()
        test_words += data_batch.size(0) * data_batch.size(1)
        dis_optimizer.zero_grad()
        loss.backward()
        dis_optimizer.step()
        _, pred_batch_ = torch.max(pred_batch, axis=-1)
        test_acc += (pred_batch_ == label_batch).sum().item() / (batch_size * 2)
        count += 1 
    test_loss = math.exp(test_loss / test_words)
    test_acc = test_acc / count
    print('[INFO] Testing, \
                    loss: %.4f, accuracy: %.4f'% \
                    (test_loss, test_acc))
    return test_loss, test_acc

'''
Define training gap
'''
def train_gap(model,
              criterion,
              optimizer,
              train_loader,
              test_loader,
              index_map,
              PATH,
              USE_CUDA,
              EPOCH_NUM,
              MC_NUM=16,
              W=[0.2,0.2,0.6],
              V="0"):
    generator, discriminator, adversary = model
    gen_criterion, dis_criterion, adv_criterion = criterion
    gen_optimizer, dis_optimizer, adv_optimizer = optimizer
    GEN_PATH, DIS_PATH, ADV_PATH = PATH

    # Adversarial Training
    print("[INFO] Start training GAP ...")
    gen_sim_loss = GANLoss()
    gen_dis_loss = GANLoss()
    gen_adv_loss = GANLoss()
    W = torch.Tensor(W)
    if USE_CUDA:
        gen_sim_loss = gen_sim_loss.cuda()
        gen_dis_loss = gen_dis_loss.cuda()
        gen_adv_loss = gen_adv_loss.cuda()
        W = W.cuda()
    csvFile = open("../param/train_gap_seq_loss" + V +".csv", 'a', newline='')
    writer = csv.writer(csvFile)
    writer.writerow(["epoch", "step", "gen_loss", "gen_mle_loss", \
                     "gen_dis_loss", "gen_adv_loss", "dis_reward", "adv_reward"])
    csvFile.close()
    for epoch in range(EPOCH_NUM):
        # Train the generator for one step
        if epoch % 5 == 0:
            train_dis(discriminator=discriminator, 
                      generator=generator,
                      train_loader=train_loader,
                      test_loader=test_loader,
                      dis_criterion=dis_criterion,
                      dis_optimizer=dis_optimizer,
                      DIS_PATH=DIS_PATH,
                      USE_CUDA=USE_CUDA, 
                      EPOCH_NUM=5,
                      PHASE="train_ep_"+str(epoch), 
                      PLOT=False)
            train_adv(adversary=adversary, 
                      generator=generator,
                      train_loader=train_loader, 
                      test_loader=test_loader, 
                      adv_criterion=adv_criterion,
                      adv_optimizer=adv_optimizer, 
                      ADV_PATH=ADV_PATH, 
                      USE_CUDA=USE_CUDA, 
                      EPOCH_NUM=5,
                      PHASE="train_ep_"+str(epoch), 
                      PLOT=True)
        # step = 0
        dis_r_bias = 0
        adv_r_bias = 0
        # for batch in tqdm(test_loader):
        #     step += 1
        #     data, category = batch['x'], batch['u'].squeeze()
        #     batch_size = data.size(0)
        #     seq_len = data.size(1)
        #     dis_pred_label = torch.ones((batch_size)).long()
        #     if USE_CUDA:
        #         data, category, dis_pred_label = \
        #         data.cuda(), category.cuda(), dis_pred_label.cuda()
        #     # noise_out, noise_hidden = privatizer.forward(input=data)
        #     with torch.no_grad():
        #         output = generator.forward(input=data)
        #         _, pred_ = torch.max(output, axis=-1)
        #         samples = pred_.view(batch_size, -1)
        #         samples_ = torch.stack([samples, data[:,:,1]], axis=2)
        #         dis_pred, dis_out = discriminator(samples_)
        #         adv_pred, adv_out = adversary(samples_)
        #         dis_acc = torch.exp(dis_out)[:,1].sum() / batch_size
        #         adv_acc = 1 - torch.exp(torch.gather(adv_out, 1, category.view(batch_size,1)).view(-1)).sum() / batch_size
        #         dis_r_bias += dis_acc.data.cpu().numpy()
        #         adv_r_bias += adv_acc.data.cpu().numpy()
        # dis_r_bias /= step
        # adv_r_bias /= step

        step = 0
        total_gen_loss = 0
        total_gen_mle_loss = 0
        total_gen_dis_loss = 0
        total_gen_adv_loss = 0
        total_dis_acc = 0
        total_adv_acc = 0
        for batch in tqdm(train_loader): 
            step += 1
            data, category = batch['x'], batch['u'].squeeze()
            batch_size = data.size(0)
            seq_len = data.size(1)
            dis_pred_label = torch.ones((batch_size)).long()
            if USE_CUDA:
                data, category, dis_pred_label = \
                data.cuda(), category.cuda(), dis_pred_label.cuda()
            # noise_out, noise_hidden = privatizer.forward(input=data)
            output = generator.forward(input=data)
            _, pred_ = torch.max(output, axis=-1)
            samples = pred_.view(batch_size, -1)
            samples_ = torch.stack([samples, data[:,:,1]], axis=2)
            dis_pred, dis_out = discriminator(samples_)
            adv_pred, adv_out = adversary(samples_)
            dis_acc = torch.exp(dis_out)[:,1].sum() / batch_size
            adv_acc = 1 - torch.exp(torch.gather(adv_out, 1, category.view(batch_size,1)).view(-1)).sum() / batch_size
            # dis_r = torch.exp(dis_pred)[:,:,1] / (1 - torch.exp(dis_pred)[:,:,1] + 1e-6)
            dis_r = (dis_r.view(batch_size, -1) - dis_r_bias) * data[:,:,1].float()
            category = category.view(batch_size,1).unsqueeze(1).repeat(1, seq_len, 1)
            adv_r = 1 - torch.exp(torch.gather(adv_pred, 2, category))
            adv_r = (adv_r.view(batch_size, -1) - adv_r_bias) * data[:,:,1].float() 
            # print(dis_r[0], adv_r[0], dis_r_bias, adv_r_bias)
            # print(dis_r[0])
            # dis_r = CumReward(dis_r, gamma=0.98, USE_CUDA=USE_CUDA) * torch.exp(dis_out)[:,1].unsqueeze(1)
            # print(dis_r[0])
            # adv_r = CumReward(adv_r, gamma=0.98, USE_CUDA=USE_CUDA)
            dis_loss = gen_dis_loss(output, samples.contiguous().view(-1), dis_r.contiguous().view(-1))
            adv_loss = gen_adv_loss(output, samples.contiguous().view(-1), adv_r.contiguous().view(-1))
            mle_loss = gen_criterion(output, data[:, :, 0].contiguous().view(-1)) / (data.size(0) * data.size(1))
            gen_loss = W[0] * mle_loss + W[1] * dis_loss + W[2] * adv_loss
            generator.zero_grad()
            gen_optimizer.zero_grad()
            gen_loss.backward()
            
            # for name, p in generator.named_parameters():
            #     print(name,p.grad)
            gen_optimizer.step()
            total_gen_loss += gen_loss.item()
            total_gen_mle_loss += mle_loss.item()
            total_gen_dis_loss += dis_loss.item()
            total_gen_adv_loss += adv_loss.item()
            total_dis_acc += dis_acc.data.cpu().numpy()
            total_adv_acc += adv_acc.data.cpu().numpy()
            # for name, p in generator.named_parameters():
            #     print(name,p.grad)
            if step % 10 == 0:
                csvFile = open("../param/train_gap_seq_loss" + V +".csv", 'a', newline='')
                writer = csv.writer(csvFile)
                writer.writerow([epoch, step, total_gen_loss / step, total_gen_mle_loss / step, \
                                    total_gen_dis_loss / step, total_gen_adv_loss / step, \
                                    total_dis_acc / step, total_adv_acc / step])
                csvFile.close()
                print("[INFO] Epoch: {}, step: {}, loss: {}, mle_loss: {}, dis_loss: {}, adv_loss: {}, \
                    dis_R: {}, adv_R: {}".\
                        format(epoch, step, total_gen_loss/step, total_gen_mle_loss/step, total_gen_dis_loss/step, \
                            total_gen_adv_loss/step, total_dis_acc/step, total_adv_acc/step))
        if epoch > 0:
            for param_group in gen_optimizer.param_groups:
                param_group['lr'] *= 0.98
                print(param_group['lr'])
        torch.save(generator.state_dict(), GEN_PATH)


'''
Define training privatizer
'''
def train_pri(model,
              criterion,
              optimizer,
              train_loader,
              test_loader,
              index_map,
              PATH,
              USE_CUDA,
              EPOCH_NUM,
              MC_NUM=16,
              W=[0.2,0.2,0.6]):
    generator, discriminator, adversary, privatizer = model
    gen_criterion, dis_criterion, adv_criterion, pri_criterion = criterion
    gen_optimizer, dis_optimizer, adv_optimizer, pri_optimizer = optimizer
    GEN_PATH, DIS_PATH, ADV_PATH, PRI_PATH = PATH

    # Adversarial Training

    print("[INFO] Start training GAP ...")
    gen_sim_loss = GANLoss()
    gen_dis_loss = GANLoss()
    gen_adv_loss = GANLoss()
    W = torch.Tensor(W)
    if USE_CUDA:
        gen_sim_loss = gen_sim_loss.cuda()
        gen_dis_loss = gen_dis_loss.cuda()
        gen_adv_loss = gen_adv_loss.cuda()
        W = W.cuda()
    csvFile = open("../param/train_privatizer_loss.csv", 'a', newline='')
    writer = csv.writer(csvFile)
    writer.writerow(["epoch", "step", "pri_loss", "pri_sim_loss", \
                     "pri_dis_loss", "pri_adv_loss", "dis_acc", "adv_acc"])
    csvFile.close()
    for epoch in range(EPOCH_NUM):
        ## Train the generator for one step
        dis_reward_bias = 0
        adv_reward_bias = 0
        step = 0
        total_pri_loss = 0
        total_pri_sim_loss = 0
        total_pri_dis_loss = 0
        total_pri_adv_loss = 0
        total_dis_acc = 0
        total_adv_acc = 0
        for batch in tqdm(train_loader): 
            step += 1
            data, category = batch['x'], batch['u'].squeeze()
            batch_size = data.size(0)
            seq_len = data.size(1)
            dis_pred_label = torch.ones((batch_size)).long()
            if USE_CUDA:
                data, category, dis_pred_label = \
                data.cuda(), category.cuda(), dis_pred_label.cuda()
            # noise_out, noise_hidden = privatizer.forward(input=data)
            samples, distance = generator.sample_with_noise(batch_size, input=data, privatizer=privatizer)
            samples_ = torch.stack([samples, data[:,:,1]], axis=2)
            dis_pred = discriminator(samples_)
            adv_pred = adversary(samples_)
            dis_acc = torch.exp(dis_pred)[:,1].sum() / batch_size
            adv_acc = torch.exp(torch.gather(adv_pred, 1, category.view(batch_size,1)).view(-1)).sum() / batch_size
            pri_dis_loss = dis_criterion(dis_pred, dis_pred_label) / batch_size
            pri_adv_loss = -adv_criterion(adv_pred, category) / batch_size
            pri_sim_loss = distance
            pri_loss = W[0] * pri_sim_loss + W[1] * pri_dis_loss + W[2] * pri_adv_loss
            gen_optimizer.zero_grad()
            pri_loss.backward()
            gen_optimizer.step()

            total_pri_loss += pri_loss.item()
            total_pri_sim_loss += pri_sim_loss.item()
            total_pri_dis_loss += pri_dis_loss.item()
            total_pri_adv_loss += pri_adv_loss.item()
            total_dis_acc += dis_acc.data.cpu().numpy()
            total_adv_acc += adv_acc.data.cpu().numpy()

        csvFile = open("../param/train_privatizer_loss.csv", 'a', newline='')
        writer = csv.writer(csvFile)
        writer.writerow([epoch, total_pri_loss / step, total_pri_sim_loss / step, \
                            total_pri_dis_loss / step, total_pri_adv_loss / step, \
                            total_dis_acc / step, total_adv_acc / step])
        csvFile.close()

        if epoch > 0:
            for param_group in gen_optimizer.param_groups:
                param_group['lr'] *= 0.99
                print(param_group['lr'])
        torch.save(generator.state_dict(), GEN_PATH)

        if (epoch + 1) % 5 == 0:
            train_dis(discriminator=discriminator, 
                      generator=generator,
                      privatizer=privatizer,
                      train_loader=train_loader,
                      test_loader=test_loader,
                      dis_criterion=dis_criterion,
                      dis_optimizer=dis_optimizer,
                      DIS_PATH=DIS_PATH,
                      USE_CUDA=USE_CUDA, 
                      EPOCH_NUM=2,
                      PHASE="train_ep_"+str(epoch), 
                      PLOT=False)
            train_adv(adversary=adversary, 
                      generator=generator,
                      privatizer=privatizer,
                      train_loader=train_loader, 
                      test_loader=test_loader, 
                      adv_criterion=adv_criterion,
                      adv_optimizer=adv_optimizer, 
                      ADV_PATH=ADV_PATH, 
                      USE_CUDA=USE_CUDA, 
                      EPOCH_NUM=2,
                      PHASE="train_ep_"+str(epoch), 
                      PLOT=True)