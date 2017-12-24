import matplotlib.pyplot as plt
from numpy.ma import arange

from load_data import load_data


def draw_line(x,y_true,y_predit):
    plt.plot(x, y_true, 'r-', linewidth=1, label='path')
    # plt.yticks(arange(0,65,5))
    plt.plot(x, y_predit, 'g-', linewidth=1, label='path')
    plt.show()
