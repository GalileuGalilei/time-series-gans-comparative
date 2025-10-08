import csv
import geoip2.database
import json
import sys

DataFile = sys.argv[1]
IP=dict()
reader = geoip2.database.Reader('geolite/GeoLite2-City.mmdb')

with open(DataFile) as csv_file:
     csv_reader = csv.reader(csv_file, delimiter=',')
     lc=0
     for row in csv_reader:
         if lc==0:
            lc+=1
         elif row[1] not in IP:
             try:
                if reader.city(row[1]):
                   IP[row[1]] = reader.city(row[1]).city.name
                   lc+=1
                else:
                   IP[row[1]] = 'NA'
                   lc+=1
             except:
                continue

print(json.dumps(IP, indent=4))

    
