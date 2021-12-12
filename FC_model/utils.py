import os
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def ensure_directory_exists(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)


def save_test_val_acc_loss_plots(train_acc, val_acc, train_loss, val_loss):
    output_dir = os.path.join(os.curdir, 'output')
    ensure_directory_exists(output_dir)

    plt.title('Net Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    blue_patch = mpatches.Patch(color='blue', label='training')
    red_patch = mpatches.Patch(color='red', label='validation')
    plt.plot([i for i in range(len(train_acc))], train_acc, color='tab:blue')
    plt.plot([i for i in range(len(val_acc))], val_acc, color='tab:red')
    plt.legend(handles=[red_patch, blue_patch], loc=4)

    plt.savefig(os.path.join(output_dir, 'Accuracies.png'), dpi=100)

    plt.clf()

    plt.title('Net Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    blue_patch = mpatches.Patch(color='blue', label='training')
    red_patch = mpatches.Patch(color='red', label='validation')
    plt.plot([i for i in range(len(train_loss))], train_loss, color='tab:blue')
    plt.plot([i for i in range(len(val_loss))], val_loss, color='tab:red')
    plt.legend(handles=[red_patch, blue_patch], loc=1)

    plt.savefig(os.path.join(output_dir, 'Losses.png'), dpi=100)
