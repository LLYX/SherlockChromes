import numpy as np
import sys
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, '../models')
sys.path.insert(0, '../datasets')

from baseline_lstm_model import BaselineChromatogramPeakDetector
from visualizer import plot_confusion_matrix

if __name__ == "__main__":
    chromatograms, labels = (
        np.load('../../../data/working/skyline_exported_chromatograms.npy'),
        np.load('../../../data/working/skyline_exported_labels.npy'))

    model = BaselineChromatogramPeakDetector(6, 6, 1)
    loss_function = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    with torch.no_grad():
        inputs = torch.from_numpy(chromatograms[18]).float()
        peak_scores = model(inputs)

        #Accuracy
        peaks = torch.from_numpy(labels[18]).float()
        output = peak_scores
        output = (output > 0.5).float().view(len(peaks))
        correct = (output == peaks).float().sum()
        print("Accuracy: {:.3f}".format(correct/output.shape[0]))

    epochs = 50
    for epoch in range(epochs):  
        # for i in range(len(chromatograms)):
        for i in [18]:
            chromatogram, peaks = chromatograms[i], labels[i]

            model.zero_grad()

            model.hidden = model.init_hidden()

            chromatogram = torch.from_numpy(chromatogram).float()
            peaks = torch.from_numpy(peaks).float()

            peak_scores = model(chromatogram)

            loss = loss_function(peak_scores.view(len(peaks)), peaks)
            loss.backward()
            optimizer.step()

            #Accuracy
            output = peak_scores
            output = (output > 0.5).float().view(len(peaks))
            correct = (output == peaks).float().sum()
            print("Epoch {}/{}, Loss: {:.3f}, Accuracy: {:.3f}".format(
                epoch + 1, epochs, loss.data.item(), correct/output.shape[0]))

    with torch.no_grad():
        inputs = torch.from_numpy(chromatograms[18]).float()
        peak_scores = model(inputs)

        #Accuracy
        peaks = torch.from_numpy(labels[18]).float()
        output = peak_scores
        output = (output > 0.5).float().view(len(peaks))
        correct = (output == peaks).float().sum()
        print("Accuracy: {:.3f}".format(correct/output.shape[0]))
        plot_confusion_matrix(peaks, output, classes=['noPeak', 'Peak'], normalize=False,
                      title='Normalized confusion matrix')