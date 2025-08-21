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

import base64 
import bz2
import gzip
import io 
import json 

import numpy as np

from lxml import etree

from mqs_reports.event import Event
from mqs_reports.event import PICK_METHOD_ALIGNED

from marsprocessingtools import constants as cnt
from marsprocessingtools import dbqueries
from marsprocessingtools import utils as tools_utils

from singlestationlocator import geodesy


XMLNS_QUAKEML_BED = "http://quakeml.org/xmlns/bed/1.2"
XMLNS_QUAKEML_BED_MARS = "http://quakeml.org/xmlns/bed/1.2/mars"
XMLNS_QUAKEML_SST = "http://quakeml.org/xmlns/singlestation/1.0"
QML_EVENT_NAME_DESCRIPTION_TYPE = 'earthquake name'
QML_SINGLESTATION_PARAMETERS_ELEMENT_NAME = "singleStationParameters"
QML_MARSQUAKE_PARAMETERS_ELEMENT_NAME = "marsquakeParameters"

XMLNS_SINGLESTATION_ABBREV = "sst"
XMLNS_SINGLESTATION = "http://quakeml.org/xmlns/singlestation/1.0"

ALL_QUALITIES = ('A', 'B', 'C', 'D')

MARS_EVENT_TYPES_NON_SF = [
    "http://quakeml.org/vocab/marsquake/1.0/MarsEventType#LOW_FREQUENCY",
    "http://quakeml.org/vocab/marsquake/1.0/MarsEventType#BROADBAND",
    "http://quakeml.org/vocab/marsquake/1.0/MarsEventType#WIDEBAND",
    "http://quakeml.org/vocab/marsquake/1.0/MarsEventType#HIGH_FREQUENCY",
    "http://quakeml.org/vocab/marsquake/1.0/MarsEventType#2.4_HZ",
    "http://quakeml.org/vocab/marsquake/1.0/MarsEventType#VERY_HIGH_FREQUENCY",
]

MARS_EVENT_TYPES = MARS_EVENT_TYPES_NON_SF
MARS_EVENT_TYPES.append(
    "http://quakeml.org/vocab/marsquake/1.0/MarsEventType#SUPER_HIGH_FREQUENCY")

# PUBLICID_PREFIX = 'smi:insight.mqs/'


def lxml_prefix_with_namespace(elementname, namespace):
    """Prefix an XML element name with a namsepace in lxml syntax."""

    return "{{{}}}{}".format(namespace, elementname)


def lxml_text_or_none(element):
    """
    If an lxml element has a text node, return its value. Otherwise,
    return None.

    """

    txt = None
    try:
        txt = element.text

    except Exception:
        pass

    return txt


def qml_get_pick_time_for_phase(event_element, pref_ori_publicid, phase_name):
    # [contains(text(), '{}')]
    ph_el = event_element.findall(
        "./{}[@publicID='{}']/{}/{}".format(
            lxml_prefix_with_namespace("origin", XMLNS_QUAKEML_BED),
            pref_ori_publicid,
            lxml_prefix_with_namespace("arrival", XMLNS_QUAKEML_BED),
            lxml_prefix_with_namespace("phase", XMLNS_QUAKEML_BED)))

    pick_time_str = ''
    pick_unc_str = ''
    
    # TODO(fab): distance method (aligned) cannot be found in pick 
    # element
    pick_methodid_str = ''
    
    for ph in ph_el:
        if str(ph.text).strip() == phase_name:
            pickid = ph.find(
                "../{}".format(
                    lxml_prefix_with_namespace("pickID", XMLNS_QUAKEML_BED)))

            ev_pick_time = event_element.find(
                "./{}[@publicID='{}']/{}/{}".format(
                    lxml_prefix_with_namespace("pick", XMLNS_QUAKEML_BED),
                    pickid.text,
                    lxml_prefix_with_namespace("time", XMLNS_QUAKEML_BED),
                    lxml_prefix_with_namespace("value", XMLNS_QUAKEML_BED)))

            ev_pick_unc = event_element.find(
                "./{}[@publicID='{}']/{}/{}".format(
                    lxml_prefix_with_namespace("pick", XMLNS_QUAKEML_BED),
                    pickid.text,
                    lxml_prefix_with_namespace("time", XMLNS_QUAKEML_BED),
                    lxml_prefix_with_namespace("lowerUncertainty", XMLNS_QUAKEML_BED)))
            
            pick_time_str = str(ev_pick_time.text).strip()
            
            if ev_pick_unc is not None:
                pick_unc_str = str(ev_pick_unc.text).strip()
            else:
                pick_unc_str = ''
            
            
            break

    return pick_time_str, pick_unc_str, pick_methodid_str


def qml_get_event_info_for_event_waveform_files(
    xml_root, location_quality, event_type, phase_list):
    
    event_info = []

    for ev in xml_root.iter(
        "{}".format(lxml_prefix_with_namespace("event", XMLNS_QUAKEML_BED))):

        # publicID
        ev_publicid = ev.get("publicID")

        # preferred origin
        pref_ori = ev.find(
            "./{}".format(lxml_prefix_with_namespace(
                "preferredOriginID", XMLNS_QUAKEML_BED)))

        # location quality from mars extension
        lq = ev.find(
            "./{}[@publicID='{}']/{}".format(
                lxml_prefix_with_namespace("origin", XMLNS_QUAKEML_BED),
                pref_ori.text,
                lxml_prefix_with_namespace(
                    "locationQuality", XMLNS_QUAKEML_BED_MARS)))

        if not lq.text[-1] in tuple(location_quality):
            continue

        # event type from mars extension
        mars_event_type = ev.find("./{}".format(
            lxml_prefix_with_namespace("type", XMLNS_QUAKEML_BED_MARS)))
        mars_event_type_str = str(mars_event_type.text).split(sep='#')[-1]

        if mars_event_type_str not in event_type:
            continue

        # event name
        desc_texts = ev.findall("./{}/{}".format(
            lxml_prefix_with_namespace("description", XMLNS_QUAKEML_BED),
            lxml_prefix_with_namespace("text", XMLNS_QUAKEML_BED)))

        ev_name = ''
        for desc_text in desc_texts:

            desc_type = desc_text.find("../{}".format(
                lxml_prefix_with_namespace("type", XMLNS_QUAKEML_BED)))

            if str(desc_type.text).strip() == QML_EVENT_NAME_DESCRIPTION_TYPE:
                ev_name = str(desc_text.text).strip()
                break

        if not ev_name:
            continue
        
        print("read event: {}".format(ev_name))

        # Get single station origin (for PDF based distance and origin time)
        sso = qml_get_sso_info_for_event_element(xml_root=xml_root, ev=ev)
        
        if 'origin_time' in sso:
            sso_origin_time = sso['origin_time']
        else:
            sso_origin_time = None
        if 'distance' in sso:
            sso_distance = sso['distance']
        else:
            sso_distance = None
        if 'distance_pdf' in sso:
            sso_distance_pdf = sso['distance_pdf']
        else:
            sso_distance_pdf = None

        # Mars event type (from BED extension)
        mars_event_type_str = ''
        mars_event_type = ev.find("./{}".format(
            lxml_prefix_with_namespace("type", XMLNS_QUAKEML_BED_MARS)))

        if mars_event_type is not None:
            mars_event_type_str = str(mars_event_type.text).strip()

        picks = dict()
        picks_sigma = dict()
        picks_methodid = dict()
        
        # TODO(fab): find out if picks belong to aligned origin
        # (only in extended catalogue QuakeML)
        for phase in phase_list:
            picks[phase], picks_sigma[phase], picks_methodid[phase] = \
                qml_get_pick_time_for_phase(ev, pref_ori.text, phase)

        latitude = ev.find("./{}[@publicID='{}']/{}/{}".format(
            lxml_prefix_with_namespace("origin", XMLNS_QUAKEML_BED),
            pref_ori.text,
            lxml_prefix_with_namespace(
                "latitude", XMLNS_QUAKEML_BED),
            lxml_prefix_with_namespace(
                "value", XMLNS_QUAKEML_BED))).text
        longitude = ev.find("./{}[@publicID='{}']/{}/{}".format(
            lxml_prefix_with_namespace("origin", XMLNS_QUAKEML_BED),
            pref_ori.text,
            lxml_prefix_with_namespace(
                "longitude", XMLNS_QUAKEML_BED),
            lxml_prefix_with_namespace(
                "value", XMLNS_QUAKEML_BED))).text
        origin_time = ev.find("./{}[@publicID='{}']/{}/{}".format(
            lxml_prefix_with_namespace("origin", XMLNS_QUAKEML_BED),
            pref_ori.text,
            lxml_prefix_with_namespace(
                "time", XMLNS_QUAKEML_BED),
            lxml_prefix_with_namespace(
                "value", XMLNS_QUAKEML_BED))).text

        event_info.append(Event(
            name=ev_name,
            publicid=ev_publicid,
            origin_publicid=str(pref_ori.text).strip(),
            picks=picks,
            picks_sigma=picks_sigma,
            quality=str(lq.text).strip(),
            latitude=float(latitude),
            longitude=float(longitude),
            sso_distance=sso_distance,
            sso_distance_pdf=sso_distance_pdf,
            sso_origin_time=sso_origin_time,
            mars_event_type=mars_event_type_str,
            origin_time=origin_time,
            picks_methodid=picks_methodid))

    return event_info


def qml_get_sso_info_for_event_element(xml_root, ev):
    
    sso_info = {}

    # preferredOriginID
    preferred_ori_id = lxml_text_or_none(ev.find("./{}".format(
        lxml_prefix_with_namespace("preferredOriginID", XMLNS_QUAKEML_BED))))

    # find SingleStationOrigin that references preferredOrigin

    # this xpath expression is invalid. why?
    # bed_ori_ref = xml_root.xpath('./{}/{}/{}[text()="{}"]'.format(
    # lxml_prefix_with_namespace(
    # QML_SINGLESTATION_PARAMETERS_ELEMENT_NAME, XMLNS_SINGLESTATION),
    # lxml_prefix_with_namespace("singleStationOrigin", XMLNS_SINGLESTATION),
    # lxml_prefix_with_namespace("bedOriginReference", XMLNS_SINGLESTATION),
    # preferred_ori_id))

    bed_ori_ref = None

    for bed_ori_ref in xml_root.iterfind('./{}/{}/{}'.format(
            lxml_prefix_with_namespace(
                QML_SINGLESTATION_PARAMETERS_ELEMENT_NAME,
                XMLNS_SINGLESTATION),
            lxml_prefix_with_namespace("singleStationOrigin",
                                       XMLNS_SINGLESTATION),
            lxml_prefix_with_namespace("bedOriginReference",
                                       XMLNS_SINGLESTATION))):

        if bed_ori_ref.text.strip() == preferred_ori_id:
            break

    if bed_ori_ref is None:
        return sso_info

    sso = bed_ori_ref.getparent()

    # extract distance, origin time, depth, backazimuth
    pref_distance_id = lxml_text_or_none(sso.find("./{}".format(
        lxml_prefix_with_namespace(
            "preferredDistanceID", XMLNS_SINGLESTATION))))

    pref_ori_time_id = lxml_text_or_none(sso.find("./{}".format(
        lxml_prefix_with_namespace(
            "preferredOriginTimeID", XMLNS_SINGLESTATION))))

    pref_depth_id = lxml_text_or_none(sso.find("./{}".format(
        lxml_prefix_with_namespace("preferredDepthID",
                                   XMLNS_SINGLESTATION))))

    pref_azimuth_id = lxml_text_or_none(sso.find("./{}".format(
        lxml_prefix_with_namespace("preferredAzimuthID",
                                   XMLNS_SINGLESTATION))))

    if pref_distance_id is not None:
        distance = lxml_text_or_none(
            sso.find(
                "./{}[@publicID='{}']/{}/{}".format(
                    lxml_prefix_with_namespace("distance",
                                               XMLNS_SINGLESTATION),
                    pref_distance_id,
                    lxml_prefix_with_namespace("distance",
                                               XMLNS_SINGLESTATION),
                    lxml_prefix_with_namespace("value",
                                               XMLNS_SINGLESTATION)
                    )))

        if distance is not None:
            sso_info['distance'] = float(distance)

        distance_pdf_variable_text = lxml_text_or_none(
            sso.find(
                "./{}[@publicID='{}']/{}/{}/{}".format(
                    lxml_prefix_with_namespace("distance",
                                               XMLNS_SINGLESTATION),
                    pref_distance_id,
                    lxml_prefix_with_namespace("distance",
                                               XMLNS_SINGLESTATION),
                    lxml_prefix_with_namespace("pdf",
                                               XMLNS_SINGLESTATION),
                    lxml_prefix_with_namespace("variable",
                                               XMLNS_SINGLESTATION))))
        
        try:
            distance_pdf_variable = distance_pdf_variable_text.split()
        except Exception:
            distance_pdf_variable = None
            
        distance_pdf_prob_text = lxml_text_or_none(
            sso.find(
                "./{}[@publicID='{}']/{}/{}/{}".format(
                    lxml_prefix_with_namespace("distance",
                                               XMLNS_SINGLESTATION),
                    pref_distance_id,
                    lxml_prefix_with_namespace("distance",
                                               XMLNS_SINGLESTATION),
                    lxml_prefix_with_namespace("pdf",
                                               XMLNS_SINGLESTATION),
                    lxml_prefix_with_namespace("probability",
                                               XMLNS_SINGLESTATION))))
        
        try:
            distance_pdf_prob = distance_pdf_prob_text.split()
        except Exception:
            distance_pdf_prob = None
        
        if distance_pdf_variable is not None:
            sso_info['distance_pdf'] = np.asarray(
                (distance_pdf_variable, distance_pdf_prob), dtype=float)

    if pref_ori_time_id is not None:
        origin_time = lxml_text_or_none(
            sso.find(
                "./{}[@publicID='{}']/{}/{}".format(
                    lxml_prefix_with_namespace(
                        "originTime", XMLNS_SINGLESTATION),
                    pref_ori_time_id,
                    lxml_prefix_with_namespace(
                        "originTime", XMLNS_SINGLESTATION),
                    lxml_prefix_with_namespace("value", XMLNS_SINGLESTATION))))

        sso_info['origin_time'] = origin_time

    if pref_depth_id is not None:
        depth = lxml_text_or_none(
            sso.find(
                "./{}[@publicID='{}']/{}/{}".format(
                    lxml_prefix_with_namespace("depth", XMLNS_SINGLESTATION),
                    pref_depth_id,
                    lxml_prefix_with_namespace("depth", XMLNS_SINGLESTATION),
                    lxml_prefix_with_namespace("value", XMLNS_SINGLESTATION))))

        if depth is not None:
            sso_info['depth'] = float(depth)

    if pref_azimuth_id is not None:
        azimuth = lxml_text_or_none(
            sso.find(
                "./{}[@publicID='{}']/{}/{}".format(
                    lxml_prefix_with_namespace("azimuth", XMLNS_SINGLESTATION),
                    pref_azimuth_id,
                    lxml_prefix_with_namespace("azimuth", XMLNS_SINGLESTATION),
                    lxml_prefix_with_namespace("value", XMLNS_SINGLESTATION))))
        if azimuth is not None:
            sso_info['azimuth'] = float(azimuth)

    return sso_info


def read_QuakeML_BED(
    fnam, event_type, phase_list, quality=ALL_QUALITIES):
    
    with open(fnam) as fh:
        tree = etree.parse(fh)
        xml_root = tree.getroot()
        
        events = qml_get_event_info_for_event_waveform_files(
            xml_root, location_quality=quality,
            event_type=event_type,
            phase_list=phase_list)

    return events


def read_JSON_Events(
    fnam, event_type, phase_list, quality=ALL_QUALITIES, baz={}):
    
    # read location JSON file
    if fnam.endswith('.gz'):
        with gzip.open(fnam, 'r') as if_loc:
            events_dict = json.load(if_loc)
            
    elif fnam.endswith('.bz2'):
        with bz2.open(fnam, 'r') as if_loc:
            events_dict = json.load(if_loc)
            
    else:
        with io.open(fnam, 'r') as if_loc:
            events_dict = json.load(if_loc)
    
    print("before filtering: read {} events from catalog JSON".format(
        len(events_dict)))
    
    event_list = []
    event_names_no_good_distance_type = []
    event_names_no_expected_distance_type = []
    
    planet_mars_geo = geodesy.CelestialBody('mars')
    
    for ev_name, ev_info in events_dict.items():
        
        check_event_type = [
            "".join((cnt.MARS_EVENT_TYPE_SCHEMA, x)) for x in event_type]
        check_location_quality = [
            "".join((cnt.LOCATION_QUALITY_SCHEMA, x)) for x in quality]
        
        if ev_info['mars_event_type'] not in check_event_type or \
                ev_info['location_quality'] not in check_location_quality:
            continue 
        
        pref_dist_type = ev_info['preferred_distance_type']
        
        if pref_dist_type is None:
            event_names_no_expected_distance_type.append(
                (ev_name, pref_dist_type,
                    "Q{}".format(ev_info['location_quality'][-1])))
                    
        elif pref_dist_type not in ('GUI', 'DL', None):
            event_names_no_good_distance_type.append((
                ev_name, pref_dist_type, 
                "Q{}".format(ev_info['location_quality'][-1])))
        
        origin_time_iso = "{0}T{1}Z".format(*ev_info['origin_time'].split())
        
        the_lat = ev_info['latitude']
        the_lon = ev_info['longitude']
            
        picks = dict()
        picks_sigma = dict()
        picks_methodid = dict()
        
        sso_distance = None
        distance_pdf = None
        
        sso_origin_time = None
        
        # always read 'meta' pick info
        for phase_code, phase_data in \
            ev_info['pick_times_all']['meta'].items():
            
            if phase_code in phase_list:
                picks[phase_code] = phase_data[0]['pick_time']
                picks_sigma[phase_code] = phase_data[0]['pick_time_lu']
                picks_methodid[phase_code] = ""
        
        # distance, BAZ, picks only for valid pref_dist_type
        if pref_dist_type in ('GUI', 'DL'):
            sso_distance = \
                ev_info['location'][pref_dist_type]['distance_sum']
            
            sso_distance_pdf = ev_info['location'][pref_dist_type]\
                ['pdf_dist_sum']['probabilities']
            
            distance_pdf_var = []
            distance_pdf_prob = []
            
            for pdf_bin in sso_distance_pdf:
                distance_pdf_var.append(pdf_bin[0])
                distance_pdf_prob.append(pdf_bin[1])
            
            distance_pdf = np.asarray(
                (distance_pdf_var, distance_pdf_prob), dtype=float)
            
            sso_origin_time = \
                ev_info['location'][pref_dist_type]['origin_time_sum']
        
            for phase_code, phase_data in \
                ev_info['pick_times_all'][pref_dist_type].items():
                
                if phase_code in phase_list:
                    picks[phase_code] = phase_data[0]['pick_time']
                    
                    # Note: use only lower uncertainty, ignore upper uncertainty
                    picks_sigma[phase_code] = phase_data[0]['pick_time_lu']
                    picks_methodid[phase_code] = ""
    
            # if GZ BAZ
            if 'gz_baz' in ev_info:
                gz_baz = ev_info['gz_baz']['bazPolarisation']

                print("ev {}: GZ BAZ: {:.1f}, lon: {:.5f}, "\
                    "lat: {:.5f}".format(ev_name, gz_baz, 
                    ev_info['longitude'], ev_info['latitude']))
        
        curr_event = Event(
            name=ev_name,
            publicid=ev_info['bed_event_id'],
            origin_publicid=ev_info['preferred_origin_id'],
            mars_event_type=ev_info['mars_event_type'],
            quality=ev_info['location_quality'],
            origin_time=origin_time_iso,
            latitude=the_lat,
            longitude=the_lon,
            sso_distance=sso_distance,
            sso_distance_pdf=distance_pdf,
            sso_origin_time=sso_origin_time,
            picks=picks,
            picks_sigma=picks_sigma,
            picks_methodid=picks_methodid)

        event_list.append(curr_event)
    
    print("{} events w/o valid/expected preferred distance type: {}".format(
        len(event_names_no_expected_distance_type),
        str(event_names_no_expected_distance_type)))
    print("{} events w/o GUI/DL preferred distance type: {}".format(
        len(event_names_no_good_distance_type), 
        str(event_names_no_good_distance_type)))
    
    return event_list


def read_DB_Events(
    conn, event_type, phase_list, quality=ALL_QUALITIES, starttime=None, 
    endtime=None):
    

    event_info = []
    
    event_info_db = dbqueries.get_all_event_ids_and_origin_start_pick_time(
        conn, starttime, endtime, event_type)
    
    print("read DB with {} events, start {}, end {}".format(
        len(event_info_db['event_name']), starttime, endtime))
    
    for ev_idx, ev_name in enumerate(event_info_db['event_name']):
        
        # origin time, lat, lon, depth
        hypo_info = dbqueries.get_origin_time_coord_depth_for_origin_publicid(
            conn, event_info_db['origin_publicid'][ev_idx])
        
        latitude = hypo_info[1]
        longitude = hypo_info[2]
        depth = hypo_info[3]
        
        sso_info = dbqueries.get_sso_info_for_origin_publicid(
            conn, event_info_db['origin_publicid'][ev_idx], cnt.PUBLICID_PREFIX)
        
        distance_pdf_var = None
        distance_pdf_prob = None
        distance_pdf = None
        
        if sso_info['distance']['pdf_var'] is not None:
            
            print("event {}, ori ID {}, distance {:.2f}".format(
                ev_name, event_info_db['origin_publicid'][ev_idx], 
                sso_info['distance']['value']))
            
            distance_pdf_var = [float(x) for x in \
                sso_info['distance']['pdf_var'].split()]
            distance_pdf_prob = [float(x) for x in \
                sso_info['distance']['pdf_prob'].split()]
            
            distance_pdf = np.asarray(
                (distance_pdf_var, distance_pdf_prob), dtype=float)
        
        picks = dict()
        picks_sigma = dict()
        picks_methodid = dict()
        
        pick_info = dbqueries.get_pick_arrival_oids_phase_pickid_for_origin_oid(
            conn, event_info_db['origin_oid'][ev_idx])
        
        # (Pick._oid, Arrival._oid, Arrival.m_phase_code, Arrival.m_pickID)
    
        # loop through all picks for preferred origin
        for phase in phase_list:
            
            pt = ''
            puc = ''
            pmid = ''
                
            for pi in pick_info:
            
                pick_oid = pi[0]
                
                if pi[2] == phase:
                    
                    # get pick time and uncertainty
                    pt, puc, _ = \
                        dbqueries.get_pick_time_uncertainty_for_pick_oid(
                            conn, pick_oid)
                    
                    break
        
            # Note: use only lower uncertainty, ignore upper uncertainty
            picks[phase] = str(pt)
            picks_sigma[phase] = puc 
            picks_methodid[phase] = pmid
                
        curr_event = Event(
            name=ev_name,
            publicid=event_info_db['event_publicid'][ev_idx],
            origin_publicid=event_info_db['origin_publicid'][ev_idx],
            mars_event_type=event_info_db['mars_event_type'][ev_idx],
            quality=event_info_db['location_quality'][ev_idx],
            origin_time=event_info_db['origin_time'][ev_idx],
            latitude=latitude,
            longitude=longitude,
            sso_distance=sso_info['distance']['value'],
            sso_distance_pdf=distance_pdf,
            sso_origin_time=sso_info['origintime']['value'],
            picks=picks,
            picks_sigma=picks_sigma,
            picks_methodid=picks_methodid)
    
        event_info.append(curr_event)
    
    return event_info


if __name__ == '__main__':
    test = read_QuakeML_BED('./mqs_reports/data/catalog_20191002.xml',
                            event_type='BROADBAND',
                            phase_list=['P', 'S', 'noise_start', 'start',
                                        'PP', 'SS',
                                        'Pg', 'Sg', 'x1', 'x2', 'x3',
                                        'end'])
    print(test)
