__author__ = 'raphaelfettaya'
import datefinder
import datetime
from math import radians, cos, sin, asin, sqrt
import pandas as pd
import googlemaps
import urllib
import requests
# from urllib import quote #PY2
from urllib.parse import quote
import os
from xml.etree import ElementTree as ET

ACCESS_TOKEN = os.environ["GOOGLE_ACCES_TOKEN"]
GEOCODER = googlemaps.Client(key=ACCESS_TOKEN)
MAX_SHOP_RES = 5
MAX_DIST_SHOP = 50.0
LENGOW_TAB_ENTRY = '{http://www.w3.org/2005/Atom}entry'

def extract_day(sentence):
    date = datefinder.find_dates(sentence)
    if len(date) >= 1:
        return date[0].strftime('%A')
    return datetime.datetime.now().strftime('%A')

def compute_distance(loc_a, loc_b):
    if loc_a is None or loc_b is None:
        return 1000000
    lon1, lat1 = loc_a
    lon2, lat2 = loc_b
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    km = 6367 * c
    return km


def get_closest_shop_from_sentence(sentence, df_shops):
    try:
        loc = text_search(sentence)
        if loc is None:
            return []
        return get_closest_shop(loc, df_shops)
    except IndexError:
        return []

def get_closest_shop(loc, df_shops):
    shops = df_shops.apply(lambda x: compute_distance(loc, load_location(x['location'])), axis=1)
    sorted_res = shops.sort_values()
    res_shops = sorted_res[sorted_res <= MAX_DIST_SHOP].index[:MAX_SHOP_RES]
    return [df_shops.index.get_loc(res) for res in res_shops]


def text_search(sentence):
    base_url = 'https://maps.googleapis.com/maps/api/place/textsearch/json'
    key_string = '?key=' + ACCESS_TOKEN
    query_string = '&query='+quote(sentence)
    sensor_string = '&sensor=false'
    type_string = ''
    url = base_url+key_string+query_string+sensor_string+type_string
    res_data = requests.get(url).json()
    if len(res_data['results']) == 0:
        return
    loc = res_data['results'][0]['geometry']['location']
    return loc['lat'], loc['lng']


def load_location(loc_data):
    if loc_data == '':
        return
    return tuple([float(lt) for lt in loc_data.split('/')])

def plural(mot):

    def pluriel_ail(mot) :
        ail= 'bail corail émail soupirail travail ventail vitrail'.split()
        if mot in ail :
            return mot[0 : -2] + 'ux'

    def pluriel_ou(mot) :
        ou = 'hibou chou genou caillou pou bijou'.split()
        if mot in ou :
            return mot + 'x'

    def pluriel_eu(mot) :
        eu = 'pneu bleu'.split()
        if mot in eu : return mot + 's'
        elif mot[-2:] == 'eu' : return (mot+'x')

    def pluriel_al(mot) :
        al = 'banal fatal naval natal bancal bal festival chacal carnaval cal serval'.split()
        if mot in al : return mot + 's'

    def pluriel_au(mot):
        au = 'landau sarrau'.split()
        if mot in au : return mot + 's'
        elif mot[-2:] == 'au' : return (mot+'x')

    def pluriel_except(mot):
        if mot== 'oeil' : return ('yeux')
        elif mot == 'ail' : return ('aulx')
        elif mot[-1] == 'z' or mot[-1] == 'x' : return (mot)

    def pluriel_regular(mot):
        if mot[-1] == 's':
            return mot
        else:
            return mot + 's'

    functions = [pluriel_ail, pluriel_ou, pluriel_eu, pluriel_al, pluriel_au, pluriel_except, pluriel_regular]

    for f in functions:
        m = f(mot)
        if m is not None:
            return m


def lengow_parser(xml_file_path):
    """
    :param xml_file_path: path to lengow file
    :return: dataframe of parsed data
    """
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    entries = [child for child in root if child.tag == LENGOW_TAB_ENTRY]
    all_tags = set([tag.tag for att in entries for tag in att])
    all_data = {k: [] for k in all_tags}
    for entry in entries:
        for tag in all_tags:
            elem = entry.find(tag)
            if elem is None:
                elem = ''
            else:
                elem = elem.text
            all_data[tag].append(elem)
    return pd.DataFrame(all_data)

if __name__ == '__main__':
    # df_data = pd.read_csv('Data/Shops2.csv').fillna('')
    # print(get_clothest_shop('la boutique de Neuilly Sur Seine svp', df_data))
    # print(plural('manteau'))
    fp = '../Downloads/lengowFR.xml'
    df = lengow_parser(fp)
    df.columns = [col.replace('{http://www.w3.org/2005/Atom}', '') for col in df.columns]
    df.to_csv('Data/Lengow.csv', encoding='utf-8')
