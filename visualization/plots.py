from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

def plot_results(file_name, vl, va, vl1, va1, title):
    plt.rcParams.update({'font.size': 22})
    X = [i for i in range(len(vl))]
    fig = plt.figure(figsize=(40, 10))
    fig.suptitle(title, fontsize=30, fontstyle='italic', fontweight='bold')
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.set_title('validation acc')
    ax2.set_title('validation loss')
    ax2.plot(X, vl1, 'b', label='single model')
    ax2.plot(X, vl, 'y', label='federated model')
    ax1.plot(X, va1, 'b', label='single model')
    ax1.plot(X, va, 'y', label='federated model')
    custom_lines = [Line2D([0], [0], color='b', lw=4),
                    Line2D([0], [0], color='y', lw=4)]

    ax1.legend(custom_lines, ['single model', 'federated model'])
    ax2.legend(custom_lines, ['single model', 'federated model'])
    fig.savefig(file_name)


def plot_multiple(file_name, all_loss_s, all_loss_f, all_acc_s, all_acc_f, epoch_step=1, n_epochs=0, n_clients=0,
                  n_experiments=0,
                  arch='', lr=0.):
    X = [i for i in range(n_epochs // epoch_step)]
    fig = plt.figure(figsize=(40, 10))
    fig.suptitle('Experiments: ' + str(n_experiments) + ', Clients: ' + str(n_clients) + ', LR: ' + str(lr) +
                 ', Model: ' + arch + ', Epoch_step:' + str(epoch_step), fontsize=30, fontstyle='italic',
                 fontweight='bold')
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.set_title('validation acc')
    ax2.set_title('validation loss')
    for i in range(len(all_acc_s)):
        ax2.plot(X, all_loss_s[i], 'b')
        ax2.plot(X, all_loss_f[i], 'y')
        ax1.plot(X, all_acc_s[i], 'b')
        ax1.plot(X, all_acc_f[i], 'y')
    custom_lines = [Line2D([0], [0], color='b', lw=4),
                    Line2D([0], [0], color='y', lw=4)]

    ax1.legend(custom_lines, ['single model', 'federated model'])
    ax2.legend(custom_lines, ['single model', 'federated model'])
    fig.savefig(file_name)


def plot_train_val(file_name, train_acc, val_acc, train_loss, val_loss):
    X = [i for i in range(len(train_acc))]
    fig = plt.figure(figsize=(40, 10))
    fig.suptitle('Train/val performance')
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.set_title('Loss')
    ax2.set_title('Accuracy')
    ax2.plot(X, train_acc, 'b', label='train')
    ax2.plot(X, val_acc, 'y', label='validation')
    ax1.plot(X, train_loss, 'b', label='train')
    ax1.plot(X, val_loss, 'y', label='validation')
    custom_lines = [Line2D([0], [0], color='b', lw=4),
                    Line2D([0], [0], color='y', lw=4)]

    ax1.legend(custom_lines, ['train', 'validation'])
    ax2.legend(custom_lines, ['train', 'validation'])
    fig.savefig(file_name)


def plot_unlabeled(file_name, vl, va, vl1, va1, title):
    plt.rcParams.update({'font.size': 22})
    X = [i for i in range(len(vl))]
    fig = plt.figure(figsize=(40, 10))
    fig.suptitle(title, fontsize=30, fontstyle='italic', fontweight='bold')
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.set_title('validation acc')
    ax2.set_title('validation loss')
    ax2.plot(X, vl1, 'b', label='unsupervised model')
    ax2.plot(X, vl, 'y', label='supervised model')
    ax1.plot(X, va1, 'b', label='unsupervised model')
    ax1.plot(X, va, 'y', label='supervised model')
    custom_lines = [Line2D([0], [0], color='b', lw=4),
                    Line2D([0], [0], color='y', lw=4)]

    ax1.legend(custom_lines, ['unsupervised model', 'supervised model'])
    ax2.legend(custom_lines, ['unsupervised model', 'supervised model'])
    fig.savefig(file_name)