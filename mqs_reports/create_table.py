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

import glob
import json
import os
import sys
import warnings

from argparse import ArgumentParser
from os.path import exists as pexists, join as pjoin
from sys import stdout as stdout

import obspy
from obspy import UTCDateTime as utct
from tqdm import tqdm

from fitter import Fitter, plot_spectra

from mqs_reports.annotations import Annotations
from mqs_reports.catalog import Catalog
from mqs_reports.snr import calc_SNR, calc_stalta
from mqs_reports.utils import solify


def create_row_header(list):
    row = '    <tr>\n'
    for li in zip(list):
        row += '<th>' + str(li) + '</th>\n'
    row += '</tr>\n'
    return row


def create_row(list, event=None, fmts=None, extras=None):
    if fmts is None:
        fmts = []
        for i in range(len(list)):
            fmts.append('%s')
    if event is None:
        row = '    <tr>\n'
    else:
        row = '    <tr id="ev_row_type_%s_quality_%s_name_%s">\n' % (event.mars_event_type_short, event.quality, event.name)
    ind_string = '      '
    if extras is None:
        for li, fmt in zip(list, fmts):
            if li is None:
                row += ind_string + '<td>-</td>\n'
            else:
                row += ind_string + '<td>' + fmt % (li) + '</td>\n'
    else:
        for li, fmt, extra in zip(list, fmts, extras):
            if li is None or (type(li) is tuple and not all(li)):
                row += ind_string + '<td>-</td>\n'
            else:
                if extra is None:
                    row += ind_string + '<td>' + fmt % (li) + '</td>\n'
                else:
                    try:
                        row += ind_string \
                               + '<td sorttable_customkey="%d">' % (extra * 100) + fmt % (li) + '</td>\n'
                    except(ValueError):
                        row += ind_string + '<td sorttable_customkey=-100000>' + fmt % (li) + '</td>\n'

    row += '    </tr>\n'
    return row


def add_information():
    string = '<H2>Distance types: &dagger;: alignment, *: Pg/Sg based; GUI-based otherwise</H2><br>\n\n'
    return string

def add_event_count_table(catalog):
    df = catalog.get_event_count_table(style='dataframe')
    html = ('<H1>MQS events until %s</H1>\n<br>\n' %
        utct().strftime('%Y-%m-%dT%H:%M (UTC)') )
    html += '''
<table border="1" class="dataframe" id="events_all">
  <thead>
    <tr style="text-align: right;">
      <th style="min-width: 40px;">event type</th>
      <th style="min-width: 40px;">abbr.</th>
      <th style="min-width: 40px;">total</th>
      <th style="min-width: 40px;"><input type="checkbox" id="cb_event_quality_A" onclick="eventFilteringToggle();" checked >A</th>
      <th style="min-width: 40px;"><input type="checkbox" id="cb_event_quality_B" onclick="eventFilteringToggle();" checked >B</th>
      <th style="min-width: 40px;"><input type="checkbox" id="cb_event_quality_C" onclick="eventFilteringToggle();" checked >C</th>
      <th style="min-width: 40px;"><input type="checkbox" id="cb_event_quality_D" onclick="eventFilteringToggle();" checked >D</th>
    </tr>
  </thead>
  <tbody>
'''
    for index, row in df.iterrows():
        checked = 'checked' if row['abbr.'] != 'SF' else ''
        html +='''
    <tr>
      <td>%s</td>
      <td><input type="checkbox" id="%s" onclick="eventFilteringToggle();" %s>%s</td>
      <td>%d</td>
      <td>%d</td>
      <td>%d</td>
      <td>%d</td>
      <td>%d</td>
    </tr>
''' % (row['event type'], f"cb_event_type_{row['abbr.']}", checked, row['abbr.'],
      row['total'], row['A'], row['B'], row['C'], row['D'])

    html += '''
  </tbody>
</table>
'''
    return html


def write_html(catalog, fnam_out, magnitude_version):
    output = create_html_header()
    output += add_event_count_table(catalog)
    output += add_information()
    output += create_table_head(
        column_names=(' ',
                      'name',
                      'type',
                      'LQ',
                      'Origin time<br>(UTC)',
                      'Start time<br>(UTC)',
                      'Start time<br>(LMST)',
                      'duration<br>[minutes]',
                      'distance<br>[degree]',
                      'BAZ<br>[degree]',
                      'SNR',
                      'P-amp<br>[m]',
                      'S-amp<br>[m]',
                      '2.4 Hz<br>pick [m]',
                      '2.4 Hz<br>fit [m]',
                      'A<sub>0</sub><sup>2</sup><br>[dB]',
                      'm<sub>b,P</sub>',
                      'm<sub>b,S</sub>',
                      'm<sub>2.4</sub>',
                      'M<sub>W,spec</sub>',
                      'M<sub>W</sub>',
                      'f<sub>c</sub><br>[Hz]',
                      't*<br>[s]',
                      'VBB<br>rate',
                      'SP1<br>rate',
                      'SPH<br>rate'))
    output_error = create_table_head(
        table_head='Events with errors',
        column_names=(' ',
                      'name',
                      'type',
                      'LQ',
                      'missing picks',
                      'picks in wrong order'))
    formats = ('%d', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s',
               '%s', '%8.2E', '%8.2E', '%8.2E', '%8.2E',
               '%4d&plusmn;%d',
               '%3.1f', '%3.1f',
               '%3.1f',
               '%3.1f&plusmn;%3.1f',
               '%3.1f&plusmn;%3.1f',
               '%3.1f', '%5.3f',
               '%s', '%s', '%s')
    time_string = {'GUI': '%s<sup>[O]</sup>',
                   'aligned': '%s<sup>[A]</sup>',
                   'PgSg': '%s',
                   'unknown': '%s'}
    dist_string = {'GUI': '{0.distance:.3g}&plusmn;{0.distance_sigma:.2g}',
                   'aligned': '<i>{0.distance:.3g}&plusmn;{0.distance_sigma:.2g}</i>&dagger;',
                   'PgSg': '<i>{0.distance:.3g}&plusmn;{0.distance_sigma:.2g}</i>*',
                   'unknown': '<i>-</i>'}
    baz_links = ' <br> <a href="{pol:s}" target="_blank">F</a> ' + \
                '<a href="{pol_zoom:s}" target="_blank">Z</a> ' + \
                '<a href="{pol_polar:s}" target="_blank">P</a> '
    baz_string = {'A': '{0.baz:5.1f}' + baz_links,
                  'B': '<i>-</i>' + baz_links,
                  'C': '<i>-</i>' + baz_links,
                  'D': '<i>-</i>'
                  }

    event_type_idx = {'LF': 1,
                      'BB': 2,
                      'HF': 3,
                      '24': 4,
                      'VF': 5,
                      'SF': 6,
                      'WB': 7}
    ievent = len(catalog)
    print('Filling HTML table with event entries')
    error_events = False
    for event in tqdm(catalog, file=stdout):
        picks_check = check_picks(ievent, event)
        if picks_check is not None:
            output_error += '<tr> ' + picks_check + '</tr>\n'
            error_events = True
        else:
            try:
                row = create_event_row(dist_string,
                                       time_string,
                                       baz_string,
                                       event,
                                       event_type_idx,
                                       formats,
                                       ievent,
                                       magnitude_version=magnitude_version)
            except (KeyError, IndexError) as e:
                print('Problem with event %s (%s-%s): %s' %
                      (event.name, event.mars_event_type_short, event.quality, e))
                error_row = '<td>%d</td>\n <td>%s</td>\n' % (ievent, event.name)
                error_row += '<td>%s</td>\n <td>%s</td>\n' % (event.mars_event_type_short, event.quality)
                error_row += '<td></td>\n <td></td>\n'
                output_error += '<tr> ' + error_row + '</tr>\n'
                error_events = True
                #raise
            else:
                output += row
        ievent -= 1
    footer = create_footer()
    output += footer
    if error_events:
        output += output_error + '    </tbody>\n </table>'
    with open(fnam_out, 'w') as f:
        f.write(output)


def check_picks(ievent, event):
    missing_picks = []
    wrong_pairs = ''
    mandatory_minimum = ['start', 'end', 'noise_start', 'noise_end']

    for pick in mandatory_minimum:
        if pick not in event.picks or event.picks[pick] == '':
            missing_picks.append(pick)

    mandatory_ABC = ['P_spectral_start', 'P_spectral_end']
    if event.quality in ['A', 'B', 'C']:
        for pick in mandatory_ABC:
            if pick not in event.picks or event.picks[pick] == '':
                missing_picks.append(pick)

    mandatory_LF_ABC = ['Peak_MbP']
    if event.quality in ['A', 'B', 'C'] and event.mars_event_type_short in ['LF', 'WB', 'BB']:
        for pick in mandatory_LF_ABC:
            if pick not in event.picks or event.picks[pick] == '':
                missing_picks.append(pick)

    mandatory_LF_AB = ['P', 'S', 'S_spectral_start', 'S_spectral_end', 'Peak_MbS']
    mandatory_LF_AB_alt = ['PP', 'SS']
    if event.quality in ['A', 'B'] and event.mars_event_type_short in ['LF', 'WB', 'BB']:
        for pick in mandatory_LF_AB:
            if pick not in event.picks or event.picks[pick] == '':
                missing_picks.append(pick)

    if 'P' in missing_picks and 'PP' in event.picks:
        #print('no P, but I found a PP')
        missing_picks.remove('P')

    if 'S' in missing_picks and 'SS' in event.picks:
        #print('no S, but I found a SS')
        missing_picks.remove('S')

    mandatory_HF_ABC = ['Pg', 'Sg', 'S_spectral_start', 'S_spectral_end', 'Peak_M2.4']
    if event.quality in ['A', 'B', 'C'] and event.mars_event_type_short in ['HF', 'VF', '24']:
        for pick in mandatory_HF_ABC:
            if pick not in event.picks or event.picks[pick] == '':

                # Pg/Sg re-labeled as P/S for VF and some HF events
                if event.mars_event_type_short in ['HF', 'VF']:
                    if pick == 'Pg':
                        if 'P' in event.picks and event.picks['P']:
                            continue
                        missing_picks.append('P')
                    elif pick == 'Sg':
                        if 'S' in event.picks and event.picks['S']:
                            continue
                        missing_picks.append('S')

                missing_picks.append(pick)

    pairs = [['P_spectral_start', 'P_spectral_end'],
             ['P', 'S'],
             #['PP', 'SS'],
             ['Pg', 'Sg'],
             ['noise_start', 'noise_end'],
             ['start', 'end']]
    for pair in pairs:
        if not (event.picks[pair[0]] == '' or event.picks[pair[1]] == ''):
            if utct(event.picks[pair[0]]) > utct(event.picks[pair[0]]):
                print('Wrong order of picks' + pair)
                wrong_pairs += pair[0] + ', '

    if len(missing_picks) > 0:
        output = '<td>%d</td>\n <td>%s</td>\n' % (ievent, event.name)
        output += '<td>%s</td>\n <td>%s</td>\n' % (event.mars_event_type_short, event.quality)
        output += '<td> '
        for pick in missing_picks:
            output += pick + ', '
        output += '</td>\n <td>'
        if len(wrong_pairs) > 0:
            output += + wrong_pairs
        output += '</td>\n'
        return output
    else:
        False

def create_event_row(dist_string, time_string, baz_string,
                     event, event_type_idx, formats,
                     ievent,
                     magnitude_version='Giardini2020',
                     path_images_local='/usr/share/nginx/html/InSight_plots',
                     path_images='../'):

    if event.origin_time == '':
        origin_time = '-'
    else:
        origin_time = event.origin_time.strftime('%Y-%m-%d<br>%H:%M:%S')

    utc_time = event.starttime.strftime('%Y-%m-%d<br>%H:%M:%S')
    lmst_time = solify(event.starttime).strftime('%H:%M:%S')
    duration = '%3d:%02d' % (float(event.duration) / 60, float(event.duration) % 60) #.strftime('%M:%S')
    event.fnam_report['name'] = event.name
    event.fnam_report['summary_local'] = pjoin(path_images_local,
                                               'summary',
                                               '%s_event_summary.png' %
                                               event.name)
    event.fnam_report['summary'] = pjoin(path_images,
                                         'summary',
                                         '%s_event_summary.png' %
                                         event.name)
    event.fnam_report['pol'] = pjoin(path_images,
                                     'polarization',
                                     '%s.png' %
                                     event.name)
    event.fnam_report['pol_zoom'] = pjoin(path_images,
                                          'polarization', 'zoom',
                                          '%s.png' %
                                          event.name)
    event.fnam_report['pol_polar'] = pjoin(path_images,
                                          'polarization', 'polar',
                                          '%s.png' %
                                          event.name)
    event.fnam_report['pol_local'] = pjoin(path_images_local,
                                           'polarization',
                                           '%s.png' %
                                           event.name)
    event.fnam_report['pol_zoom_local'] = pjoin(path_images_local,
                                                'polarization', 'zoom',
                                                '%s.png' %
                                                event.name)
    event.fnam_report['pol_polar_local'] = pjoin(path_images_local,
                                                'polarization', 'polar',
                                                '%s.png' %
                                                event.name)
    sp_files  = glob.glob( pjoin('filterbanks', event.name, f'*SampRate_SP_HF*.png') )
    lfhf_files  = glob.glob( pjoin('filterbanks', event.name, f'*SampRate_LF+HF*.png') )
    zrt_files  = glob.glob( pjoin('filterbanks', event.name, f'*Rotation_ZRT*.png') )
    deglitched_files = glob.glob( pjoin('filterbanks', event.name, f'*DEGLITCHED*.png') )
    denoised_files   = glob.glob( pjoin('filterbanks', event.name, f'*DENOISED*.png') )
    zoom_in_files   = glob.glob( pjoin('filterbanks', event.name, f'*Zoom_in*.png') )
    zoom_ph_files   = glob.glob( pjoin('filterbanks', event.name, f'*Zoom_phases*.png') )
    url_vars = ('eventId=%s&hasSP=%s&hasLFHF=%s&hasRT%s&hasDeglitched=%s&hasDenoised=%s&hasZoomIn=%s&hasZoomPh=%s' %
                (event.name,
                 'yes' if len(sp_files) != 0 else 'no',
                 'yes' if len(lfhf_files) != 0 else 'no',
                 'yes' if len(zrt_files) != 0 else 'no',
                 'yes' if len(deglitched_files) != 0 else 'no',
                 'yes' if len(denoised_files) != 0 else 'no',
                 'yes' if len(zoom_in_files) != 0 else 'no',
                 'yes' if len(zoom_ph_files) != 0 else 'no'
                 )
                )
    event.fnam_report['fb'] = ('filterbanks.html?%s' % url_vars)

    sp_files  = glob.glob( pjoin('spect', event.name, f'*SampRate_SP_HF*.png') )
    lfhf_files  = glob.glob( pjoin('spect', event.name, f'*SampRate_LF+HF*.png') )
    zrt_files  = glob.glob( pjoin('spect', event.name, f'*Component_T*.png') )
    deglitched_files = glob.glob( pjoin('spect', event.name, f'*DEGLITCHED*.png') )
    denoised_files   = glob.glob( pjoin('spect', event.name, f'*DENOISED*.png') )
    url_vars = ('eventId=%s&hasSP=%s&hasLFHF=%s&hasRT%s&hasDeglitched=%s&hasDenoised=%s' %
                (event.name,
                 'yes' if len(sp_files) != 0 else 'no',
                 'yes' if len(lfhf_files) != 0 else 'no',
                 'yes' if len(zrt_files) != 0 else 'no',
                 'yes' if len(deglitched_files) != 0 else 'no',
                 'yes' if len(denoised_files) != 0 else 'no'
                 )
                )
    event.fnam_report['spect'] = ('spect.html?%s' % url_vars)

    path_dailyspec = pjoin(path_images,
                           'spectrograms/by_sols/Sol%04d/0imageviewer.html'
                           % int(float(solify(event.starttime)) / 86400 + 1))
    # try:
    if event.mars_event_type_short in ('HF', 'VF', '24'):
        snr = calc_stalta(event, fmin=2.2, fmax=2.8)
        snr_string = '%.1f (2.4Hz)' % snr
    elif event.mars_event_type_short == ('SF'):
        snr, snr_win = calc_SNR(event, fmin=8.0, fmax=12.,
                                SP=True, hor=True)
        snr_string = '%.1f (%s, 8-12Hz)' % (snr, snr_win)
    else:
        snr, snr_win = calc_SNR(event, fmin=0.2, fmax=0.5)
        snr_string = '%.1f (%s, 2-5s)' % (snr, snr_win)

    sortkey = (ievent,
               None,
               event_type_idx[event.mars_event_type_short],
               None,
               float(utct(event.starttime)),
               float(utct(event.starttime)),
               float(solify(event.starttime)) % 86400,
               None,
               event.distance,
               event.baz,
               snr,
               event.pick_amplitude('Peak_MbP',
                                    comp='vertical',
                                    fmin=1. / 6.,
                                    fmax=1. / 2,
                                    unit='fm'),
               event.pick_amplitude('Peak_MbS',
                                    comp='horizontal',
                                    fmin=1. / 6.,
                                    fmax=1. / 2,
                                    unit='fm'),
               event.pick_amplitude('Peak_M2.4',
                                    comp='vertical',
                                    fmin=2.2, fmax=2.6,
                                    unit='fm'),
               event.amplitudes['A_24'],
               event.amplitudes['A0'],
               None,
               None,
               None,
               None,
               None,
               None,
               None,
               None,
               None,
               None
               )
    # They never exist locally, since Savas creates them manually
    if pexists(event.fnam_report['summary_local']):
        link_report = \
            ('<a href="{summary:s}" target="_blank">{name:s}</a><br>' +
             '<a href="{spect:s}" target="_blank">Spect</a>').format(
                **event.fnam_report)
    else:
        link_report = \
            ('{name:s}<br>' +
             '<a href="{spect:s}" target="_blank">Spect</a>').format(
                **event.fnam_report)
    if pexists(event.fnam_report['pol_local']):
        link_report += ' <a href="{pol:s}" target="_blank">Pol</a>'.format(
            **event.fnam_report)

    link_duration = '<a href="%s" target="_blank">%s</a>' % (
        event.fnam_report['fb'], duration)

    link_lmst = '<a href="%s" target="_blank">%s</a>' % (
        path_dailyspec, lmst_time)

    row = create_row(
        (ievent,
         link_report,
         event.mars_event_type_short,
         event.quality,
         time_string[event.distance_type] % origin_time,
         utc_time,
         link_lmst,
         link_duration,
         dist_string[event.distance_type].format(event),
         (" " if event.baz is None else "%.f"%event.baz), #baz_string[event.quality].format(event, **event.fnam_report),
         snr_string,
         event.pick_amplitude('Peak_MbP',
                              comp='vertical',
                              fmin=1. / 6.,
                              fmax=1. / 2),
         event.pick_amplitude('Peak_MbS',
                              comp='horizontal',
                              fmin=1. / 6.,
                              fmax=1. / 2),
         event.pick_amplitude('Peak_M2.4',
                              comp='vertical',
                              fmin=2.2, fmax=2.6),
         10 ** (event.amplitudes['A_24'] / 20.)
         if event.amplitudes['A_24'] is not None else None,
         (event.amplitudes['A0']
          if event.amplitudes['A0'] is not None else None,
          event.amplitudes['A0_err']
          if 'A0_err' in event.amplitudes and event.amplitudes['A0_err'] is not None else None),
         event.magnitude(mag_type='mb_P', version=magnitude_version)[0],
         event.magnitude(mag_type='mb_S', version=magnitude_version)[0],
         event.magnitude(mag_type='m2.4', version=magnitude_version)[0],
         event.magnitude(mag_type='MFB', version=magnitude_version),
         event.magnitude(mag_type='Mw', version=magnitude_version),
         event.amplitudes['f_c'],
         event.amplitudes['tstar'],
         (_replace_none(event.available_sampling_rates()['VBB_Z'])
             if event.available_sampling_rates()['VBB100_Z'] is None
             else event.available_sampling_rates()['VBB100_Z']),
         _replace_none(event.available_sampling_rates()['SP_Z']),
         _replace_none(event.available_sampling_rates()['SP_N']),
         ),
        event,
        extras=sortkey,
        fmts=formats)

    # except KeyError:  #ValueError: # KeyError: #, AttributeError) as e:
    #     link_lmst = '<a href="%s" target="_blank">%s</a>' % (
    #         path_dailyspec, lmst_time)
    #     sortkey = (ievent,
    #                None,
    #                event_type_idx[event.mars_event_type_short],
    #                None,
    #                float(utct(event.picks['start'])),
    #                float(solify(event.picks['start'])) % 86400,
    #                0.)
    #     row = create_row((  # ievent, event.name, 'PRELIMINARY LOCATION'
    #         ievent,
    #         event.name,
    #         event.mars_event_type_short,
    #         event.quality,
    #         utc_time,
    #         link_lmst,
    #         'PRELIM'
    #         ),
    #         extras=sortkey)
    return row


def create_footer():
    footer = 4 * ' ' + '</tbody>\n'
    footer += 2 * ' ' + '</table>\n</article>\n</body>\n</html>\n'
    return footer


def _replace_none(val):
    return ' ' if val is None else val

def create_html_header():
    return '''
<!DOCTYPE html>
<html lang="en-US">
<head>
  <script src="sorttable.js"></script>
  <title>MQS events</title>
  <meta charset="UTF-8">
  <meta name="description" content="InSight marsquakes">
  <meta name="author" content="Marsquake Service" >
  <link rel="stylesheet" type="text/css" href="./table.css">
</head>
<body>

<script>

eventFilteringToggle();

function eventFilteringToggle() {

  var evTypes=["SF","VF","WB","BB","LF","HF","24"];
  var evQs=["A","B","C","D"];

  for (var  t= 0; t < evTypes.length; t++) {
    for (var q = 0; q < evQs.length; q++) {

      var evType    = evTypes[t];
      var evQuality = evQs[q];

      var cbEvType = document.getElementById(`cb_event_type_${evType}`);
      var cbEvQuality = document.getElementById(`cb_event_quality_${evQuality}`);

      var enabled = ((cbEvType !== null ? cbEvType.checked : true) &&
                     (cbEvQuality !== null ? cbEvQuality.checked : true));

      var idPrefix=`ev_row_type_${evType}_quality_${evQuality}`;

      var evRows = document.querySelectorAll(`tr[id^='${idPrefix}']`);

      evRows.forEach((row) => {
          if (enabled) {
              row.style.display="table-row";
          } else {
              row.style.display="none";
          }
      });
    }
  }
}
</script>
'''


def create_table_head(column_names, table_head='Event table'):
    output = ''
    output += '<article>\n'
    output += '  <header>\n'
    output += '    <h1>' + table_head + '</h1>\n'
    output += '  </header>\n'
    table_head = '  <table class="sortable" id="events">\n' + \
                 '  <thead>\n' + \
                 create_row(column_names) + \
                 '  </thead>\n'
    output += table_head
    output += '  <tbody>\n'
    return output


def define_arguments():
    helptext = 'Create HTML overview table and individual event plots'
    parser = ArgumentParser(description=helptext)

    helptext = 'Input QuakeML BED file'
    parser.add_argument('input_quakeml', help=helptext)

    helptext = 'Input annotation file'
    parser.add_argument('input_csv', help=helptext)

    helptext = 'Input BAZ JSON file'
    parser.add_argument('input_baz', help=helptext)

    helptext = 'Input fitting parameter file (marsspectgui)'
    parser.add_argument('input_fitparams', help=helptext)

    helptext = 'Input default fitting parameter file (marsspectgui)'
    parser.add_argument('input_fitparams_default', help=helptext)

    helptext = 'Inventory file'
    parser.add_argument('inventory', help=helptext)

    helptext = 'Path to SC3DIR'
    parser.add_argument('sc3_dir', help=helptext)

    helptext = 'Location qualities (one or more)'
    parser.add_argument('-q', '--quality', help=helptext,
                        nargs='+', default=('A', 'B', 'C', 'D'))

    helptext = 'Distances to use: "all" (default), "aligned", "GUI"'
    parser.add_argument('-d', '--distances', help=helptext,
                        default='all')

    helptext = 'Magnitude version to use: "Giardini2020" (default), "Boese2021"'
    parser.add_argument('-m', '--mag_version', help=helptext,
                        default='Giardini2020')

    helptext = 'Event types'
    parser.add_argument('-t', '--types', help=helptext,
                        default='all')

    helptext = 'Single plot: all, filterbanks, spectral-fit, table'
    parser.add_argument('-p', '--plot', help=helptext,
                        default='all')

    helptext = 'Data type: RAW, DEGLITCHED, DENOISED'
    parser.add_argument('--data-type', help=helptext, default='RAW')
    
    helptext = 'Orientation (one or more, ZNE, ZRT)'
    parser.add_argument('-o', '--orientation', help=helptext, nargs='+', 
        default=('ZNE', 'ZRT'))
    
    helptext = 'Filterbank norm (one or more, none, single, all)'
    parser.add_argument('-n', '--norm', help=helptext, nargs='+', 
        default=('none', 'single', 'all'))
    
    helptext = 'force product re-creation'
    # parser.add_argument('--force-products', help=helptext, default=False)
    
    parser.add_argument('--force-products', action='store_true')
    parser.add_argument(
        '--no-force-products', dest='force-products', action='store_false')
    
    return parser.parse_args()


if __name__ == '__main__':

    args = define_arguments()

    # load GZ BAZ JSON file
    with open(args.input_baz, 'r') as baz_data:
        gz_baz_dict = json.load(baz_data)
        
    catalog = Catalog(
        fnam_event=args.input_quakeml, type_select=args.types, 
        quality=args.quality, baz=gz_baz_dict)
    
    if len(catalog) == 0:
        print("catalog is empty, exiting")
        sys.exit()

    ann = Annotations(fnam_csv=args.input_csv)

    # load manual (aligned) distances
    # OBSOLETE
#     if args.distances == 'all':
#         catalog.load_distances(fnam_csv=args.input_dist)
#         fnam_out='overview.html'
#     
#     elif args.distances == 'GUI':
#         fnam_out='overview_GUI.html'
#     
#     elif args.distances == 'aligned':
#         catalog.load_distances(fnam_csv=args.input_dist, overwrite=True)
#         fnam_out='overview_aligned.html'

    inv = obspy.read_inventory(args.inventory)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        print("Read {} waveforms".format(args.data_type))
        
        catalog.read_waveforms(
            inv=inv, wf_type=args.data_type, kind='DISP', sc3dir=args.sc3_dir)
    
    normtypes = []
    for n in args.norm:
        if n != 'none':
            normtypes.append("{}_components".format(n))
        else:
            normtypes.append(n)
    
    if 'all' in args.plot or 'filterbanks' in args.plot:
        for smprate in ('VBB_LF', 'SP_HF', 'LF+HF'):
            for rotate in (False, True):
                for normtype in ('none', 'single_components', 'all_components'):
                    
                    print("Plot filter banks for {} waveforms (smprate {}, "\
                        "norm {}, ZRT {})".format(
                            args.data_type, smprate, normtype, rotate))
                    
                    catalog.plot_filterbanks(
                        dir_out='filterbanks', annotations=ann, 
                        normtype=normtype, rotate=rotate, smprate=smprate,
                        orientation=args.orientation, norm=normtypes,
                        force_products=args.force_products)

    if 'all' in args.plot or 'spectral-fit' in args.plot:
        with open(args.input_fitparams) as json_data:
            fitting_parameters = json.load(json_data)
            
        with open(args.input_fitparams_default) as json_data:
            fitting_parameters_defaults = json.load(json_data)
        
        fitter = Fitter(
            catalog=catalog, inventory=inv, path_sc3dir=args.sc3_dir)
        
        for smprate in ('VBB_LF', 'SP_HF', 'LF+HF'):
            for rotate in (False, True):
                
                print("Plot spectra for {} waveforms (smprate {}, "\
                    "ZRT {})".format(args.data_type, smprate, rotate))
                
                # implicitly calls event.calc_spectra()
                plot_spectra(
                    fitter=fitter, fitting_parameters=fitting_parameters,
                    fitting_parameters_defaults=fitting_parameters_defaults,
                    dir_out='spect', winlen_sec=20.0, wf_type=args.data_type,
                    rotate=rotate, smprate=smprate, 
                    orientation=args.orientation,
                    force_products=args.force_products)

    if 'all' in args.plot or 'table' in args.plot:
        
        # HTML table creation is OBSOLETE
        # plot_spectra() calls event.calc_spectra()
        print('Calc spectra') # to be called only on RAW data
        catalog.calc_spectra(winlen_sec=20.0, detick_nfsamp=10)
        
        fnam_out = 'overview_GUI.html'
        
        print('Create table')
        write_html(catalog, fnam_out=fnam_out, magnitude_version=args.mag_version)
