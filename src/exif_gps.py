import numpy as np
import pandas as pd
import exifread
from os import listdir
from os.path import isfile, join
import re
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import folium
from itertools import combinations

# from https://gist.github.com/snakeye/fdc372dbf11370fe29eb
# based on https://gist.github.com/erans/983821

def _get_if_exist(data, key):
    if key in data:
        return data[key]

    return None

def _convert_to_degrees(value):
    """
    Helper function to convert the GPS coordinates stored in the EXIF to degress in float format
    :param value:
    :type value: exifread.utils.Ratio
    :rtype: float
    """
    d = float(value.values[0].num) / value.values[0].den
    m = float(value.values[1].num) / value.values[1].den
    s = float(value.values[2].num) / value.values[2].den

    return d + (m / 60.0) + (s / 3600.0)

def get_exif_location(exif_data):
    """
    Returns the latitude and longitude, if available, from the provided exif_data (obtained through get_exif_data above)
    """
    lat = None
    lon = None
    gps_latitude = _get_if_exist(exif_data, 'GPS GPSLatitude')
    gps_latitude_ref = _get_if_exist(exif_data, 'GPS GPSLatitudeRef')
    gps_longitude = _get_if_exist(exif_data, 'GPS GPSLongitude')
    gps_longitude_ref = _get_if_exist(exif_data, 'GPS GPSLongitudeRef')

    if gps_latitude and gps_latitude_ref and gps_longitude and gps_longitude_ref:
        lat = _convert_to_degrees(gps_latitude)
        if gps_latitude_ref.values[0] != 'N':
            lat = 0 - lat

        lon = _convert_to_degrees(gps_longitude)
        if gps_longitude_ref.values[0] != 'E':
            lon = 0 - lon
    # if gps_latitude_ref.values[0] != 'N':
    #     gps_latitude.values[0].num *= -1
    # if gps_longitude_ref.values[0] != 'E':
    #     gps_longitude.values[0].num *= -1
    return lat, lon
    # return gps_latitude, gps_longitude

def gps_to_array_map(img_root):
    '''
    Returns array containing: image name, latitude, longitude values for images contained within directory provided.
    '''
    resultlist = []
    files = [f for f in listdir(img_root) if isfile(join(img_root, f))]
    for filename in files:
        if not (filename.startswith('.')):
            if not (filename.startswith('None')):
                path = '{}{}'.format(img_root, filename)
                with open(path, 'rb') as f:
                    tags = exifread.process_file(f)
                    lat, lon = get_exif_location(tags)
                    img_cat = re.sub("\d+", "", filename).rstrip('.jpg')
                    # img_cat = img_cat[:-3]
                    img_cat = img_cat.rstrip("_")
                    img_cat = img_cat.replace("_", " ")
                    resultlist.append((filename, lat, lon, img_cat))
            location_arr = np.asarray(resultlist)
    # plot_img_locations(location_arr)
    return location_arr

def plot_img_locations(location_arr):
    flower_map = folium.Map(location = [39.74675277777778, -105.2436], zoom_start = 10, tiles="Stamen Terrain")
    for i in range(len(location_arr)):
        lat = location_arr[i][1]
        lon = location_arr[i][2]
        category = location_arr[i][3]
        folium.Marker(location = [lat, lon], popup = category).add_to(flower_map)
    return flower_map.save("flower_map.html")

def make_plant_instances(location_arr):
    '''
    Groups individual images into plant instances (i.e., multiple images were taken of the same plant) based on GPS location.
    '''
    results = find_rows(location_arr))
    for pair in results:
        location_arr[location_arr['']]
    for i in range(len(location_arr)):
        coords = ((location_arr[i][1], location_arr[i][2]))

def find_rows(location_arr):
    iterable = zip(location_arr[:,0], location_arr[:,1])
    result_list = []
    for thing in combinations(iterable, 2):
        print(thing[0][1], thing[1][1])
        if float(thing[0][1]) == float(thing[1][1]):
            result_list.append(thing)
    return result_list

    # a = location_arr
    # b = np.copy(location_arr)
    # dt = np.dtype((np.void, a.dtype.itemsize * a.shape[1]))
    #
    # a_view = np.ascontiguousarray(a).view(dt).ravel()
    # b_view = np.ascontiguousarray(b).view(dt).ravel()
    #
    # sort_b = np.argsort(b_view)
    # where_in_b = np.searchsorted(b_view, a_view,
    #                              sorter=sort_b)
    # where_in_b = np.take(sort_b, where_in_b)
    # which_in_a = np.take(b_view, where_in_b) == a_view
    # where_in_b = where_in_b[which_in_a]
    # which_in_a = np.nonzero(which_in_a)[0]
    # return np.column_stack((which_in_a, where_in_b))

def unique_rows(location_arr):
    result_arr = find_rows(location_arr)
    final_list = []
    for i in range(len(result_arr)):
        if result_arr[i][0] != result_arr[i][1]:
            final_list.append(list(result_arr[i]))
            print(list(result_arr[i]))
    return np.array(final_list)
    # result_arr = np.array(result_arr)
    # return result_arr
    # result_arr = result_arr[result_arr[:,0] ]
    # for i in len(result_arr):
    #
    # return result_arr

# def get_exif_data(filename):
#     with open(filename, 'rb') as f:
#         tags = exifread.process_file(f)
#         lat, lon = get_exif_location(tags)
#     return lat, lon


if __name__ == '__main__':
    lat, lon = get_exif_location(exif_data)
