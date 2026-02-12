import matplotlib.pyplot as plt
import numpy as np

def plot_data(X, y, title):
    plt.figure()
    plt.scatter(X[y==1][:,0], X[y==1][:,1], label="Class +1")
    plt.scatter(X[y==-1][:,0], X[y==-1][:,1], label="Class -1")
    plt.legend()
    plt.title(title)
    plt.show()

def plot_decision_boundary(w, b, X):
    x = np.linspace(X[:,0].min()-1, X[:,0].max()+1, 100)
    y = -(w[0]*x + b)/w[1]
    plt.plot(x, y, 'k')

def plot_margin_lines(w, b, X):
    x = np.linspace(X[:,0].min()-1, X[:,0].max()+1, 100)
    y1 = -(w[0]*x + b - 1)/w[1]
    y2 = -(w[0]*x + b + 1)/w[1]
    plt.plot(x, y1, 'r--')
    plt.plot(x, y2, 'r--')