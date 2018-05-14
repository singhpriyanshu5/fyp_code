import tensorflow as tf
import matplotlib.pyplot as plt
from IPython import display
import numpy as np

#to plot the losses
def Plot(x_val, p1, p2, title, name1, name2):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.figure(figsize=(5,4))
    plt.title(title)
    plt.plot(x_val,p1, label=name1)
    plt.plot(x_val,p2, label=name2)
    plt.legend()
    plt.show()
