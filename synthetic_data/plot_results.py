from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

def plot_training_progress(train_losses, test_losses, outfile):
    epoch_start = 500
    
    plt.plot(train_losses, label='train loss')
    plt.plot(test_losses, label='test loss')
    plt.xlim(epoch_start, plt.xlim()[1])
    ylims = list(plt.ylim())
    ylims[1] = max(train_losses[epoch_start], test_losses[epoch_start])
    plt.ylim(tuple(ylims))
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.savefig(outfile, bbox_inches='tight')
    plt.close()
    
# def plot_metrics
        
infile = Path('results/training_progress.csv')
training_progress = pd.read_csv(infile)

outfile = Path('results/training_progress.png')
plot_training_progress(training_progress['train_loss'], training_progress['test_loss'], outfile)
