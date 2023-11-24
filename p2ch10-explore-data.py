import numpy as np

#leo  need install the following 
# (py39-torch1.13.1) PS E:\> conda install  cassandra-driver
# (py39-torch1.13.1) PS E:\> conda install diskcache 

#from cassandra.cqltypes import BytesType
##from diskcache import FanoutCache, Disk,core
#from diskcache.core import io
#from io import BytesIO
#from diskcache.core import MODE_BINARY

from p2ch10.dsets import getCandidateInfoList, getCt, LunaDataset


candidateInfo_list = getCandidateInfoList(requireOnDisk_bool=False)
positiveInfo_list = [x for x in candidateInfo_list if x[0]]
diameter_list = [x[1] for x in positiveInfo_list]



