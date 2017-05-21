from pymongo import MongoClient
import requests
from bs4 import BeautifulSoup
import os
import re
import urllib.request

url = "http://www.easterncoloradowildflowers.com/_s_3frame.htm"
'''
In bash, start 'mongod'
In new bash tab, start 'mongo'.
Show dbs.
'use all_the_snacks' (if you call use and db doesnt exist yet, it creates new one for you).
db.createCollection('yum')

'''
client = MongoClient()
db = client['flowers']
tab = db['wf']


def read_html(url):
    headers = { 'User-Agent': 'Mozilla/5.0 (Windows NT 6.0; WOW64; rv:24.0) Gecko/20100101 Firefox/24.0' }
    req = requests.get(url, headers=headers)
    soup = BeautifulSoup(req.text, 'html.parser')
    list_items = soup.find_all("p")
    # css_tags = soup.find_all('div', attrs= {'class': css_class})
    links = []
    for li in list_items:
        links.append(li.a['href'])
    # pathlist = [tag.find('img')['src'] for tag in css_tags]
    for link in links:
        url_snack = url + link
        req_snack = requests.get(url_snack)
        soup_snack = BeautifulSoup(req_snack.text, html.parser)
        name_number = soup_snack.select('h4')[0].text.encode("ascii", "ignore")
        #encode("ascii", "ignore") Dumps out any character not encodable by asci
        snack_number = name_number.split()[0]
        #gets first item from string
        snack_name = ' '.join(name_number.split()[1:])
        #joins everything after the first element, which was snack_number
        tab = soup_snack.find_all('div', attrs={"class": "data clearfix"})
        #find all things inside div tag, select attributes where class is data clearfix (on the webpage, data clearfix was the entire table)
        tab_dd = tab[0].find_all('dd')
        # take first item from list returned from table, return all dd tags
        flavor = tab_dd[0].text.strip().split(',')
        #pull just the text out from first item in tab_dd, make a list of flavors
        cuisine = tab_dd[1].text.strip()
        series = tab_dd[2].text.strip().split(',')

        #for composition, there are little images of components. we want the names, not the images.
        comp_list = tab_dd[3].select('img')
        #the links we want are images. we just want the names. Pull that from the tags for the images.
        comps = []
        for comp in comp_list:
            comps.append(comp.get('alt', ''))
        #get alt tag (if doesnt exist, return '') for each component
        full_text = soup_snack.find_all('div', attrs={'id': 'rightstuff'})[0].text.split('\n')
        #get a new item for every carriage return. items = strings.
        date = ' '.join(full_text[6].split()[-3:])
        #year, day, month are always last 3 things in this sentence.
        descr = full_text[3]
        #description is always fourth paragraph
        taste = re.sub('Taste', '', full_text[4])
        #removes header from start of the paragraph

        coll.insert_one({'name': snack_name,
        etc.})
    # return link

'''
query the mongodb:

db.yum.find({composition: {$regex: 'cornmeal'}}, {_id: false, name: true})

For pinging, add certain number of seconds to how often they allow you to scrape, then add a random number of seconds to that.
add time.sleep to the for loop.
'''

if __name__ == '__main__':
    read_html("http://www.snackdata.com")


#boto s3 bucket amazon
