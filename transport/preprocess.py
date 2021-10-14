
"""
-------------------------------------------------------------------------------------------------
Author:         Patil Aakash (aakash.patil@mines-paristech.fr)    
From Library:   PyTransport                                
Year:           March, 2020                                                
-------------------------------------------------------------------------------------------------
"""



#case and results location
case_dir = '../case_sample'
resultats_dir = case_dir+'/resultats/2d/' 


#re-sampling grid nx and ny
Tnx, Tny = 360, 300


############## Import libs: SYS, LOCAL, USER   ##############
import os
import glob
import numpy as np
from tqdm import tqdm
from Lib_PyTransport import *
##############################################################

#get list of all result files
fileListVTU = sorted(glob.glob(resultats_dir+'bulles*.vtu'))
print("Found ",len(fileListVTU) ,".vtu files. \n First file: ",fileListVTU[0] ,"\n Last file: ", fileListVTU[-1] )


#get node info from original VTU / init VTU 
inputVTU = fileListVTU[0]
bounds = getVTUBounds(inputVTU) 


try:
    structDict, coordinates_structList = TransportVTU2Struct_2d(inputVTU,arrayName='ALL', nx=Tnx, ny=Tny, outputFile=None, returnDict=True)
    struct_Velocity = structDict['VitesseP1']
    print("struct_Velocity.shape ",struct_Velocity.shape)
except:
    raise NameError("** ERROR Some issue with ", inputVTU) 


#convert all vtu unstruct results to struct arrays 
bigArr = np.zeros((len(fileListVTU), Tnx, Tny, 5))
for t in tqdm(range(len(fileListVTU))):
    vtuFile = fileListVTU[t]
    print("Reading and Converting: ", vtuFile)
    structDict, coordinates_structList = TransportVTU2Struct_2d( vtuFile, arrayName='ALL', nx=Tnx, ny=Tny, outputFile=None, returnDict=True)
    bigArr[t, :, :, :2] = structDict['VitesseP1'][:,:,:2]
    bigArr[t, :, :, 2] = structDict['PressionP1']
    bigArr[t, :, :, 3] = structDict['NuTildeP1']
    bigArr[t, :, :, 4] = structDict['MuTurbP1']
    
    
#save in format [N, nx, ny, ch] 
np.save("dataset.npy", bigArr)

print("Done converting. Output saved to dataset.npy")






    



