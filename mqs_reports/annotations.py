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

import csv

from obspy import UTCDateTime as utct


class Annotations:
    def __init__(self,
                 fnam_csv,
                 starttime=None,
                 endtime=None,
                 type='ONE-SIDED_PULSE'):
        self.glitches = []

        if starttime is None:
            starttime = utct('1970-01-01')
        if endtime is None:
            endtime = utct('2030-01-01')
        with open(fnam_csv, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file,
                                        fieldnames=['starttime',
                                                    'endtime',
                                                    'author',
                                                    'type',
                                                    'comment',
                                                    'annotation'
                                                    ])
            for row in csv_reader:
                if row['starttime'][0] != '#':
                    if type in row['type'] and \
                       len(row['endtime']) > 0 and \
                       utct(row['starttime']) > starttime and \
                       utct(row['endtime']) < endtime:
                        self.glitches.append([utct(row['starttime']),
                                              utct(row['endtime'])])

    def select(self,
               starttime=None,
               endtime=None):
        if starttime is None:
            starttime = utct('1970-01-01')
        if endtime is None:
            endtime = utct('2030-01-01')
        glitches_selected = []
        for glitch in self.glitches:
            if starttime < glitch[0] < endtime:
                glitches_selected.append(glitch)

        return glitches_selected


if __name__ == '__main__':
    ann = Annotations(fnam_csv='mqs_reports/data/annotated_epochs.csv')
    print(ann)
    pass
