import glob
import os
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == '__main__':
    csv_paths = glob.glob('*/*.csv')

    for file in csv_paths:
        df_history = pd.read_csv(file)
        print(os.path.abspath(os.path.join(file, os.pardir)))
        # do some fancy plots
        # Accuracy
        plt.figure()
        plt.plot(df_history.get('epoch'), df_history.get('val_categorical_accuracy'), label='val_categorical_accuracy')
        plt.plot(df_history.get('epoch'), df_history.get('categorical_accuracy'), label='categorical_accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend()
        plt.savefig(os.path.join(os.path.abspath(os.path.join(file, os.pardir)), 'accuracy.png'))
        # Loss
        plt.figure()
        plt.plot(df_history.get('epoch'), df_history.get('val_loss'), label='val_loss')
        plt.plot(df_history.get('epoch'), df_history.get('loss'), label='loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend()
        plt.savefig(os.path.join(os.path.abspath(os.path.join(file, os.pardir)), 'loss.png'))
