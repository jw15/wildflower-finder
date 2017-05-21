import numpy as np
import pandas as pd

df = pd.read_csv('/Users/jenniferwaller/galvanize/capstone/data/OSMPVegetation_2015.csv')
df[]

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


'''

'''
A1540 = Soapweed Yucca Prairie Scrub Alliance AKA Yucca glauca
A1537 = Skunkbush Sumac / Little Bluestem - Threadleaf Sedge Shrub
A3561 = ? (not in spreadhseet)
A1488 = ? (not in spreadhseet)
A9012 = ? (not in spreadhseet)
A9017 = ? (not in spreadhseet)
A1234 = ? (not in spreadhseet)
A1192 = ? (not in spreadhseet)
A1374 = Baltic Rush - Mexican Rush Wet Meadow Alliance AKA Juncus balticus - Juncus mexicanus Wet Meadow Alliance
A1232 = ? (not in spreadhseet)
A2578 = ? (not in spreadhseet)
A1381 = Reed Canarygrass Ruderal Marsh Alliance AKA Phalaris arundinacea Eastern Ruderal Marsh Alliance
A1414 = ? (not in spreadhseet)
A1417 = ? (not in spreadhseet)
A9003 = ? (not in spreadhseet)
A1332 = Saltgrass Alkaline Wet Meadow Alliance AKA Distichlis spicata Alkaline Wet Meadow Alliance
A1433 = ? (not in spreadhseet)
A1444 = ? (not in spreadhseet)
A1436 = Narrowleaf Cattail - Broadleaf Cattail - Bulrush species Deep
A1252
'''

'''
***predict_proba to predict whether plant is in one category vs all other plants. returns tuple. can give you probability plant is in x category. then can return category with max probability.***
'''
