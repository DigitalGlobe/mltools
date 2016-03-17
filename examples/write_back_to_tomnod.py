# Writes the scores of the machine classifications back
# to Tomnod. Features classified as swimming pools are given the 
# highest priority (which is 0).
# This script was used in the adelaide_pools_2016 campaign.

import json
from mltools.json_tools as jt
from mltools.crowdsourcing import TomnodCommunicator

# contains classified features
filename = '1040010014BF5100_classified.geojson'

# get related data from file
property_names = ['class_name', 'score', 'feature_id']
data = jt.get_from_geojson(filename, property_names)  

# compute priorities
priorities = [abs(entry[1] - 0.5) for entry in data]

# add priorities to data
data = [list(x)+[y] for x,y in zip(data, priorities)]

# connect to tomnod
with open('credentials.json', 'r') as f:
    credentials = json.load(f)
tc = TomnodCommunicator(credentials)

# execute update query to feature table of adelaide_pools_2016
query = '''UPDATE adelaide_pools_2016.feature 
           SET type_id = (SELECT id FROM tag_type WHERE name = '{0}'),
               score = {1}, 
               priority = {3} 
               WHERE id = {2} 
               AND num_votes_total <= 0;'''
tc.execute(query, data, batch_size = 1000)