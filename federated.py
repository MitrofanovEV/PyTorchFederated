import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from visualization.plots import plot_results, plot_multiple, plot_train_val
import os
from logs.experiment_logs import save_data
from logs.training_logs import log
import datetime

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def copy_model(main, n_clients, fmod):

    # copying main model weights and saving to temporary files to load at next training epoch
    # main: main(centralized) model
    # n_clients: number of clients(nodes)
    # fmod: folder for temporary models

    for i in range(n_clients):
        name1 = fmod + 'federated_model_weights_' + str(i)
        torch.save(main.state_dict(), name1)
    """
        #averaging optimizer parameters
        name1 = fmod + 'federated_model_opt_' + str(i)
        torch.save(opt.state_dict(), name1)
    """


def get_loaders(n, main_ds, batch_size):
    # doing random split of dataset at N equal sized loaders to emulate clients
    # n: number of clients
    # main_ds: dataset to split
    # batch_size: batch size of each client data loader
    split = int(len(main_ds) / n)
    loaders = list()
    splits = [split for i in range(n)]
    splits[-1] = len(main_ds) - split * (len(splits) - 1)
    print('split_sizes: ', splits)
    datasets = torch.utils.data.random_split(main_ds, splits)
    for dataset in datasets:
        dl = torch.utils.data.DataLoader(dataset=dataset,
                                         pin_memory=True,
                                         batch_size=batch_size,
                                         shuffle=True)
        loaders.append(dl)
    return loaders


def fix_arch(model, num_classes):
    # fixing FC layer of ResNet to predict required number of classes
    # num_classes: number of classes to fix architecture
    if str(type(model)) == '<class \'torchvision.models.resnet.ResNet\'>':
        # model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = torch.nn.Linear(2048, num_classes)
    return


def fit_epoch(model, optimizer, criterion, train_dl, fn):
    # training 1 epoch of given model and returning metrics
    # fn: folder name to save all logs
    model.train()
    log('Fitting epoch', fn)
    running_loss = 0
    running_acc = 0
    total = len(train_dl.dataset)
    cumulative = 0
    for i, (images, labels) in enumerate(train_dl):
        cumulative += len(labels)
        images = images.to(device)
        labels = labels.to(device)
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        running_acc += (outputs.argmax(dim=1) == labels).float().cpu().sum()
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            log('Train loss: {:.3f}, Train acc: {:.3f}'.format(running_loss / cumulative,
                                                               running_acc / cumulative * 100.),
                fn)
    log('Train loss: {:.3f}, Train acc: {:.3f}'.format(running_loss / total, running_acc / total * 100.), fn)
    return running_loss / total, running_acc / total * 100.


def save_client(client, opt, i, fmod):
    # saving client training information to interrupt training for aggregation
    # i: index number of client
    # fmod: folder to save weights
    name1 = fmod + 'federated_model_weights_' + str(i)
    name2 = fmod + 'federated_model_opt_' + str(i)
    torch.save(client.state_dict(), name1)
    torch.save(opt.state_dict(), name2)
    del client
    del opt


def load_client(i, arch, lr, fmod, num_classes):
    # loading client model and optimizer to continue training
    # fmod: folder to load weights
    # num_classes: number of classes to fix architecture
    model = arch()
    fix_arch(model, num_classes)
    name1 = fmod + 'federated_model_weights_' + str(i)
    name2 = fmod + 'federated_model_opt_' + str(i)
    model.load_state_dict(torch.load(name1))
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    opt.load_state_dict(torch.load(name2))
    return model, opt


def validation(model, criterion, val_dl):
    # validating on given loader 1 epoch on given model and returning metrics
    model.eval()
    correct = 0.
    loss = 0.
    cumulative = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_dl):
            cumulative += len(labels)
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            correct += (outputs.argmax(dim=1) == labels).float().sum()
            if i % 100 == 0:
                print('loss: {:.3f}, Acc: {:.3f}'.format(loss / cumulative, correct / cumulative * 100.))

    # model.train()
    return correct / float(len(val_dl.dataset)) * 100., loss / float(len(val_dl.dataset))


def aggregation(main_model, n_clients, arch, lr, fmod, num_classes):
    # simple weights averaging aggregation

    # client_models = list()
    # client_models = [load_client(i,arch,lr,fmod,num_classes) for i in range(n_clients)]
    # global m1
    for name in main_model.state_dict():
        param = main_model.state_dict()[name]
        tmp = torch.zeros(param.shape, dtype=torch.float).to(device)
        fl = True
        for j in range(n_clients):
            model, opt = load_client(j, arch, lr, fmod, num_classes)
            tmp += model.state_dict()[name].data.float()
        param.data.copy_(tmp * (1. / n_clients))
    copy_model(main_model, n_clients, fmod)


"""
            if (j == 0) and fl:
                m1 = opt
                print("copied successfully")
            else:
                for i in opt.state_dict()['state']:
                    m1.state_dict()['state'][i]['exp_avg'].data += opt.state_dict()['state'][i]['exp_avg'].data
                    m1.state_dict()['state'][i]['exp_avg_sq'].data += opt.state_dict()['state'][i]['exp_avg_sq'].data
        for i in m1.state_dict()['state']:
            m1.state_dict()['state'][i]['exp_avg'].data /= n_clients
            m1.state_dict()['state'][i]['exp_avg_sq'].data /= n_clients

        fl = False
        param.data.copy_(tmp * (1. / n_clients))

    copy_model(main_model, m1, n_clients, fmod)

    for name in main_opt.state_dict():
        param = main_opt.state_dict()[name]
        tmp = torch.zeros(param.shape, dtype=torch.float).to(device)
        for j in range(n_clients):
            tmp += client_models[j][1].state_dict()[name].data.float()
        param.data.copy_(tmp * (1. / n_clients))
"""


# copy_model(main_model, n_clients, fmod)


def federated(n_clients, arch, train_dataset, val_dl, fn, epoch_step=1, lr=1e-3, n_epochs=30, num_classes=2,
              batch_size=128, train_dl=None):
    """

    :param n_clients: number of clients
    :param arch: model architecture
    :param train_dataset: PyTorch dataset
    :param val_dl: validation data loader
    :param fn: folder of experiment
    :param epoch_step: number of local epochs before averaging
    :param lr: learning rate
    :param n_epochs: total number of epochs
    :param num_classes: number of classes
    :param batch_size: batch size
    :param train_dl: train data loader
    :return: loss and accuracy metric of federated trained model
    """

    fmod = fn + '/models/'
    fplot = fn + '/plots/'
    try:
        os.mkdir(fmod)
    except FileExistsError:
        pass
    main_model = arch(pretrained=True)
    fix_arch(main_model, num_classes)
    main_model = main_model.to(device)
    torch.save(main_model.state_dict(), 'initial_model')
    criterion = nn.CrossEntropyLoss()
    train_loaders = get_loaders(n_clients, train_dataset, batch_size=batch_size)
    client_models = [arch() for i in range(n_clients)]
    for i, model in enumerate(client_models):
        fix_arch(model, num_classes)
    optimizers = [torch.optim.Adam(client_models[i].parameters(), lr=lr) for i in range(n_clients)]

    for i in range(n_clients):
        save_client(client_models[i], optimizers[i], i, fmod)
    val_loss = list()
    val_acc = list()
    train_loss = list()
    train_acc = list()
    copy_model(main_model, n_clients, fmod)
    for epoch in range(n_epochs):
        losstr = 0.
        acctr = 0.
        for i in range(n_clients):
            client, opt = load_client(i, arch, lr, fmod, num_classes)
            loss, acc = fit_epoch(client, opt, criterion, train_loaders[i], fn)
            losstr += loss
            acctr += acc
            save_client(client, opt, i, fmod)
        losstr /= n_clients
        acctr /= n_clients
        if (epoch + 1) % epoch_step == 0:
            aggregation(main_model, n_clients, arch, lr, fmod, num_classes)
            acc, loss = validation(main_model, criterion, val_dl)
            # acctr, losstr = validation(main_model, criterion, train_dl)
            log('Epoch [{}/{}], ValLoss: {:.4f}, ValAccuracy: {:.3f}'
                .format(epoch + 1, n_epochs, loss, acc), fn)
            log('Epoch [{}/{}], TrainLoss: {:.4f}, TrainAccuracy: {:.3f}'
                .format(epoch + 1, n_epochs, losstr, acctr), fn)
            val_loss.append(loss)
            val_acc.append(acc)
            train_loss.append(losstr)
            train_acc.append(acctr)
        else:
            log('Epoch [{}/{}]'.format(epoch + 1, n_epochs), fn)
    plot_train_val(fplot + 'federated_training.png', train_acc, val_acc, train_loss, val_loss)
    val_loss = val_loss
    val_acc = val_acc
    return val_loss, val_acc, main_model


def single(arch, train_dl, val_dl, fn, lr=1e-3, epoch_step=1, n_epochs=30, num_classes=2):

    """
    Training model with same architecture as clients but using whole training dataset
    :param arch: model architecture
    :param train_dl: train data loader
    :param val_dl: validation data loader
    :param fn: folder of experiment
    :param lr: learning rate
    :param epoch_step: number of local epochs before validation
    :param n_epochs: total number of epochs
    :param num_classes: number of classes
    :return: loss and accuracy
    """
    fplot = fn + '/plots/'
    model = arch(pretrained=True)
    fix_arch(model, 2)

    #model.load_state_dict(torch.load('unlabeled_model'))
    model.load_state_dict(torch.load('initial_model'))
    fix_arch(model, num_classes)
    model = model.to(device)
    # torch.save(model.state_dict(), 'initial_model_unlabeled')
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    val_loss = list()
    val_acc = list()
    train_loss = list()
    train_acc = list()
    for epoch in range(n_epochs):
        losstr, acctr = fit_epoch(model, opt, criterion, train_dl, fn)
        if (epoch + 1) % epoch_step == 0:
            acc, loss = validation(model, criterion, val_dl)
            val_loss.append(loss)
            val_acc.append(acc)
            train_loss.append(losstr)
            train_acc.append(acctr)
            log('Epoch [{}/{}], ValLoss: {:.4f}, ValAccuracy: {:.3f}'
                .format(epoch + 1, n_epochs, loss, acc), fn)
        else:
            log('Epoch [{}/{}]'.format(epoch + 1, n_epochs), fn)
    plot_train_val(fplot + 'single_training.png', train_acc, val_acc, train_loss, val_loss)
    return val_loss, val_acc, model


def experiment(arch, train_dataset, validation_dataset, n_epochs=30, n_clients=10, n_experiments=10,
               learning_rate=1e-3,
               batch_size=128,
               epoch_step=1,
               prefix='1'):
    """
    Performing series of experiments comparing federated and single model performance with same initial weights
    :param arch: model architecture
    :param train_dataset: train dataset
    :param validation_dataset: validation dataset
    :param n_epochs: total number of epochs
    :param n_clients: number of clients
    :param n_experiments: number of experiments
    :param learning_rate: learning rate
    :param batch_size: batch size
    :param epoch_step: number of local epochs before averaging
    :param prefix: prefix to add in folder name
    :return: void
    """
    path = os.getcwd()
    # fn = path + '/experiments/' + prefix + '_nex' + str(n_experiments) + '_nep' + str(n_epochs) + '_nc' + str(
    #    n_clients) + '_epst' + str(epoch_step)
    fn = path + '/experiments/' + str(datetime.datetime.now().date()) + '_' + str(
        datetime.datetime.now().time()) + '_' + prefix
    fplot = fn + '/plots'
    fn1 = fplot + '/average.png'
    fn2 = fplot + '/all.png'
    try:
        os.mkdir(fn)
        os.mkdir(fplot)
    except FileExistsError:
        pass
    train_dl = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           pin_memory=True,
                                           shuffle=True)
    val_dl = torch.utils.data.DataLoader(dataset=validation_dataset,
                                         batch_size=batch_size,
                                         pin_memory=True,
                                         shuffle=False)
    all_acc_s = list()
    all_loss_s = list()
    all_loss_f = list()
    all_acc_f = list()
    average_acc_f = np.zeros((n_epochs // epoch_step), dtype=float)
    average_loss_f = np.zeros((n_epochs // epoch_step), dtype=float)
    average_acc_s = np.zeros((n_epochs // epoch_step), dtype=float)
    average_loss_s = np.zeros((n_epochs // epoch_step), dtype=float)
    for i in range(n_experiments):
        log('Starting experiment [' + str(i + 1) + '/' + str(n_experiments) + ']', fn)
        log('Federated_model', fn)
        val_loss, val_acc, model_f = federated(n_clients, arch, train_dataset, val_dl, fn, lr=learning_rate,
                                               n_epochs=n_epochs,
                                               epoch_step=epoch_step,
                                               batch_size=batch_size,
                                               train_dl=train_dl)
        average_acc_f += np.array(val_acc, dtype=float)
        average_loss_f += np.array(val_loss, dtype=float)
        all_loss_f.append(val_loss)
        all_acc_f.append(val_acc)
        log('Single_model', fn)
        val_loss1, val_acc1, model_s = single(arch, train_dl, val_dl, fn, lr=learning_rate, n_epochs=n_epochs,
                                              epoch_step=epoch_step)
        all_loss_s.append(val_loss1)
        all_acc_s.append(val_acc1)
        average_acc_s += np.array(val_acc1, dtype=float)
        average_loss_s += np.array(val_loss1, dtype=float)
        torch.save(model_f.state_dict(), fn + '/federated_model_' + str(i))
        torch.save(model_s.state_dict(), fn + '/single_model_' + str(i))

        # plot_results(val_loss, val_acc, val_loss1, val_acc1, 'Experiment number ' + str(i + 1) + ':')

    average_acc_f *= (1. / n_experiments)
    average_loss_f *= (1. / n_experiments)
    average_acc_s *= (1. / n_experiments)
    average_loss_s *= (1. / n_experiments)

    save_data(arch, fn, epoch_step=epoch_step, n_epochs=n_epochs,
              n_clients=n_clients, n_experiments=n_experiments, learning_rate=learning_rate, batch_size=batch_size)
    plot_results(fn1, average_loss_f, average_acc_f, average_loss_s, average_acc_s,
                 'Average over all experiments ' + ':')
    plot_multiple(fn2, all_loss_s, all_loss_f, all_acc_s, all_acc_f, epoch_step=epoch_step, n_epochs=n_epochs,
                  n_clients=n_clients, n_experiments=n_experiments, arch=str(arch), lr=learning_rate)

    return
