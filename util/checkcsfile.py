import copy
import csv
import functools
import glob
import os


mhd_list = glob.glob('data-unversioned/part2/luna/subset*/*.mhd')
presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}
    
#print(presentOnDisk_set)    

#candidates.csv
#annotations_with_malignancy
#annotations
filename = "candidates"
fileWriteName =  "data/part2/luna/" + filename + "_subset0.csv"
fileWriteHandle =  open(fileWriteName, "w")
fileReadName = "data/part2/luna/" + filename + ".csv"

fileHandle = open(fileReadName, "r")
fileReadLines = fileHandle.readlines()
    
ii = 0
for line in fileReadLines:
    
    if ii == 0:
        fileWriteHandle.write("%s" %line)
    else:
        series_uid = line.split(",")[0]
        if series_uid in presentOnDisk_set:
            print(line)
            fileWriteHandle.write("%s" %line)
    ii += 1
"""  
with open(fileReadName, "r") as f:
    row1 = csv.reader(f)[0:1]
    fileWriteHandle.write("%s\n" %row1)
    for row in list(csv.reader(f))[1:]:
        series_uid = row[0]
        if series_uid in presentOnDisk_set:
            print(row)
            fileWriteHandle.write("%s\n" %row)
"""     

       