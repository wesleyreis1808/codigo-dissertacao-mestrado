import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class Plot:
    def __init__(this):
        fig = plt.figure(1)
        ax = fig.add_subplot(1, 1, 1)

        # Adjust Grid
        major_ticks = np.arange(0, 1, 0.1)
        ax.set_xticks(major_ticks)
        ax.set_yticks(major_ticks)
        ax.grid(which='both')
        plt.grid(True)

        plt.axis(np.array([0, 1, 0, 1]))
        plt.xlabel('Probabiliy of false alarm, P_fa')
        plt.ylabel('Probabiliy of detection, P_d')

        this.plt = plt

    def add_plot(this, Pfa, Pd, AUC, style, label="AUC"):
        legend = "%s(%.5f)" % (label, AUC)
        this.plt.plot(Pfa, Pd, style, label=legend)

    def show(this):
        this.plt.legend()
        this.plt.show()

    def save(this, fig_name):
        this.plt.legend()
        this.plt.savefig(fig_name + ".png")

def plot(Pfa, Pd, AUC):
    fig = plt.figure(1)
    ax = fig.add_subplot(1, 1, 1)

    # Adjust Grid
    major_ticks = np.arange(0, 1, 0.1)
    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)
    ax.grid(which='both')
    plt.grid(True)

    legend = "AUC(%.2f)" % (AUC)
    plt.plot(Pfa, Pd, 'r-^', label=legend)

    plt.axis(np.array([0, 1, 0, 1]))
    plt.xlabel('Probabiliy of false alarm, P_fa')
    plt.ylabel('Probabiliy of detection, P_d')
    plt.legend()
    plt.show()


def plotSCM_H0_H1(dataH0, dataH1):

    AcovH0 = np.abs(np.cov(dataH0, rowvar=True, bias=True))
    AcovH1 = np.abs(np.cov(dataH1, rowvar=True, bias=True))

    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(10, 8)

    # Choosing the colors
    cmap = sns.color_palette("GnBu", 20)

    ax0 = plt.subplot(2, 2, 1)
    sns.heatmap(AcovH0, cmap=cmap, vmin=0)
    ax2 = plt.subplot(2, 2, 3)
    sns.heatmap(AcovH1, cmap=cmap, vmin=0)

    # data can include the colors
    c = "#0A98BE"

    ax1 = plt.subplot(2, 2, 2)
    in_line = np.concatenate((dataH0))
    ax1.scatter(np.real(in_line), np.imag(in_line), c=c, s=40)

    ax3 = plt.subplot(2, 2, 4)
    in_line = np.concatenate((dataH1))
    ax3.scatter(np.real(in_line), np.imag(in_line), c=c, s=40)

    # Remove the top and right axes from the data plot
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)

    plt.show()
    plt.close()


def plotHistory(history):
    # print(history.history.keys())
    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(10, 4)

    # Choosing the colors
    cmap = sns.color_palette("GnBu", 20)

    ax0 = plt.subplot(1, 2, 1)
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.ylim(0, 1)
    plt.legend(['train', 'test'], loc='upper left')
    

    ax2 = plt.subplot(1, 2, 2)
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.show()
    plt.close()
