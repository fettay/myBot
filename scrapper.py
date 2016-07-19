__author__ = 'raphaelfettaya'
import pandas as pd
import googlemaps


def load_shops_location(gmaps_client, df_shops):
    all_loc = []
    for ind, shop in df_shops.iterrows():
        address = shop['city'] + ' ' + shop['Adresse']
        try:
            location = gmaps_client.geocode(address)[0]['geometry']['location']
        except IndexError:
            all_loc.append('')
            continue
        lat_lg = '{lat}/{lng}'.format(lat=location['lat'], lng=location['lng'])
        all_loc.append(lat_lg)
        print(ind)
    return all_loc


if __name__ == '__main__':
    gmaps = googlemaps.Client(key='AIzaSyAUWBep-cs1UvD45Aiz5HSuhEjNkxJ2Vfs')
    df_data = pd.read_csv('Data/Shops.csv').fillna('')
    # all_loc = load_shops_location(gmaps, df_data)
    # df_data['location'] = all_loc
    # df_data.to_csv('Data/Shops2.csv', index=None, encoding='utf-8')