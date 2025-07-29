#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Python tools to create products/plots for the final version of the InSight
Marsquake service Mars event catalogue

:copyright:
    Simon Stähler (mail@simonstaehler.com), 2018
    Martin van Driel (Martin@vanDriel.de), 2018
    Luca Scarabello (luca.scarabello@sed.ethz.ch), 2024
    Savas Ceylan (savas.ceylan@eaps.ethz.ch), 2024
    Fabian Euchner (fabian.euchner@sed.ethz.ch), 2024
:license:
    GPLv3
"""

import matplotlib.pyplot as plt
import numpy as np


def scatter_annot(*args, names, fig=None, **kwargs):

    if fig is None:
        fig = plt.figure()

    ax = fig.gca()

    sc = plt.scatter(*args, **kwargs)

    annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def update_annot(ind):

        pos = sc.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        text = " ".join([names[n] for n in ind["ind"]])
        annot.set_text(text)


    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)

    return fig, ax



if __name__ == "__main__":


    N = 10
    x = np.random.randn(N)
    y = np.random.randn(N)
    names = x.astype('str')

    fig = scatter_annot(x, y, names=names, label='bla')
    scatter_annot(x, y, names=names, fig=fig, label='bli')
    plt.legend()
    plt.show()
