import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

def normalization(x, y):
    for i in range(x.shape[1]):
        x[:,i] = x[:,i]/x[:,i].max()
    return x,y/y.max()

def loss(y,yh):
    return sum((yh-y)**2)/(2*len(y))

def plot_logs(ytest,ypred):
    logs = pd.DataFrame({'validation': ytest, 'predicted': ypred})
    f,ax=plt.subplots(1,2,figsize=(16,6))
    ax[0].title.set_text('actual vs predicted results')
    ax[1].title.set_text('linearity of actual and predicted results')
    sb.scatterplot(data=logs,ax=ax[0])
    sb.scatterplot(x=ytest,y=ypred,ax=ax[1])
    sb.lineplot(x=ytest,y=ytest,ax=ax[1])
    print('Loss :',round(2*loss(ytest,ypred),ndigits=4))
    plt.show()

class LinearRegressionGD:

    def __init__(self, iterations = 10000, lr = 5e-2):
        self.iterations = iterations
        self.lr = lr
        self.weights = None
        self.bias = 0
        self.train_loss=[]
        self.val_loss=[]

    def fit(self,x,y,val_data=None):
        self.weights = np.zeros((x.shape[1],))+0.5
        iter_factor = self.iterations/10
        if val_data is not None:
            for i in range(self.iterations):
                yh = x@self.weights + self.bias
                self.weights = self.weights + self.lr*(x.T @ (y - yh) / len(y))
                self.bias = self.bias + self.lr*sum((y-yh)/len(y))
                pred = val_data[0]@self.weights + self.bias
                if self.iterations%1000==0 and (i+1)%iter_factor==0:
                    self.train_loss.append(2*loss(y, yh))
                    self.val_loss.append(2*loss(val_data[1],pred))
                    print(f'Epoch = {i+1}/{self.iterations}  '
                          f'Training Loss = {round(2*loss(y, yh),ndigits=4)}  '
                          f'Validation Loss = {round(2*loss(val_data[1],pred),ndigits=4)}')
        else:
            for i in range(self.iterations):
                yh = x@self.weights + self.bias
                self.weights = self.weights + self.lr*(x.T @ (y - yh) / len(y))
                self.bias = self.bias + self.lr*sum((y-yh)/len(y))
                if self.iterations%1000==0 and (i+1)%iter_factor==0:
                    self.train_loss.append(2*loss(y, yh))
                    print(f'Epoch = {i+1}/{self.iterations}  Training Loss = {round(2*loss(y, yh),ndigits=4)}')

    def predict(self,x):
        return x@self.weights + self.bias

    def plot_loss(self):
        plt.xlabel('epochs')
        plt.ylabel('loss')
        sb.lineplot(data=self.train_loss)
        sb.lineplot(data=self.val_loss)
        plt.legend(labels=['training', 'validation'])
        plt.show()













