import numpy as np
import pandas as pd
from pymongo import MongoClient
import requests
from bs4 import BeautifulSoup
import os
import re
import urllib.request

headers = { 'User-Agent': 'Mozilla/5.0 (Windows NT 6.0; WOW64; rv:24.0) Gecko/20100101 Firefox/24.0' }
req = requests.get(url, headers=headers)

# from pymongo import MongoClient
# client = MongoClient()
# db = client.
# coll = db.articles
# for document in coll.find():

'''
In mongo terminal:
mongoimport --db veg --collection plants --type json --file "OSMPVegetation_2015.GeoJSON"
'''

'''
From brochure:
Aquilegia coerulea; Colorado Columbine
Mahonia repens; Oregon Grape
Pulsatilla patens ssp. multifida; Pasqueflower
Claytonia rosea; Spring Beauty
Leucocrinum montanum; Sand Lily
Adenolinum lewisii; Blue Flax
Thermopsis montana; Golden Banner
Mertensia lanceolata; Lance-leaved Chiming Bells
Geranium caespitosum ssp.caespitosum; Wild Geranium
Iris missouriensis; Wild Iris
Padus virginiana ssp. melanocarpa; Chokecherry
Arnica cordifolia; Heart-leaved Arnica
Achillea lanulosa; Yarrow
Delphinium nuttallianum; Larkspur
Lupinus argenteus; Lupine
Yucca glauca; Yucca
Penstemon secundiflorus; One-sided Penstemon
Toxicoscordion venenosum; Death Camas
Opuntia macrorhiza; Prickly Pear
Drymocallis fissa; Leafy Cinquefoil
Campanula rotundifolia; Harebell
Gaillardia aristata; Blanketflower
Calochortus gunnisonii; Mariposa Lily
Monarda fistulosa var. menthaefolia; Bergamot
Penstemon glaber; Alpine Penstemon
Frasera speciosa; Monument Plant
Amerosedum lanceolatum; Stonecrop
Erysimum asperum; Western Wallflower
Aster spp.; Aster
Liatris punctata; Blazing Star

TRY HIGH PASS FILTER FOR IMAGE CROPPING

'''

if __name__ == '__main__':
    # pathlist = read_html('http://www.ebay.com/sch/i.html?_from=R40&_trksid=p2050890.m570.l1313.TR10.TRC0.A0.H0.Xshoe.TRS0&_nkw=shoes&_sacat=6000', 'lvpic pic img left')
    # img_dir(pathlist)
    url = 'http://www.easterncoloradowildflowers.com/'
    payload = {'api-key': 'GOTCHA!', 'begin_date': '20170501', 'end_date': '20170502'}
    html_str = single_query(link, payload)
