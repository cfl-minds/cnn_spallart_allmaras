
"""
-------------------------------------------------------------------------------------------------
Author:         Patil Aakash (aakash.patil@mines-paristech.fr)    
From Library:   PyTransport                                
Year:           March, 2020                                                
-------------------------------------------------------------------------------------------------
"""


import os
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from scipy import interpolate 



def getVTUBounds(filenameVTU):
    """
    getVTUBounds gets the bounds in x,y,z for a given vtu file
    """
    print("Reading ",filenameVTU)
    reader = vtk.vtkXMLUnstructuredGridReader()  
    reader.SetFileName(filenameVTU) 
    reader.Update()
    data = reader.GetOutput()
    bounds = data.GetBounds()   
    print("Space bounds: ", bounds)
    return bounds;
#bounds = getVTUBounds(filenameVTU) 


def TransportVTU2Struct_2d(filenameVTU,arrayName='ALL', nx=36, ny=30, outputFile=None, returnDict=True, useCoors='No'):
    """
    TransportVTU2Struct_2d reads a given vtu file along with its content 
    and samples the content from unstructured to rectilinear grid
    """
    print("Reading ",filenameVTU)
    reader = vtk.vtkXMLUnstructuredGridReader()  
    reader.SetFileName(filenameVTU) 
    reader.Update()
    data = reader.GetOutput()
    points = data.GetPoints()
    coordinates = vtk_to_numpy(points.GetData())

    #struct mesh
    bounds = data.GetBounds()
    if useCoors == 'No':
        uxcor = np.linspace(bounds[0], bounds[1], nx)
        uycor = np.linspace(bounds[2], bounds[3], ny)
    else:
        uxcor = useCoors[0]
        uycor = useCoors[1]

    if bounds[4] == bounds[5]  :
        allUz = coordinates[:,2][:]  
        uzcor = allUz[0] 
        coordinates = coordinates[:,:2]
    else:
        raise NameError("*makeStructData_2d*  It appears to be a 3D mesh with z-Bounds: ", bounds[4], bounds[5] ) 

    coordinates_structList = [] 
    xv, yv = np.meshgrid(uxcor, uycor, sparse=False, indexing='ij') 
    for i in range(nx): 
        for j in range(ny): 
                coordinates_structList.append([ xv[i,j], yv[i,j] ])  
    coordinates_structList = np.array(coordinates_structList) 
    print("Shape of 2D coordinates_structList ", coordinates_structList.shape)
    print("Shape of 2D expected (", nx*ny, ",2)")

    #get arrays from VTU
    n_arrays = reader.GetNumberOfPointArrays()
    n_arrays_names = []
    for i in range(n_arrays):
        namearr = reader.GetPointArrayName(i)
        n_arrays_names.append(namearr)
  
    if arrayName in n_arrays_names: 
        print("Fetching only array ", arrayName) 
        arrayPosition = n_arrays_names.index(arrayName)
        jstart, jend = arrayPosition,arrayPosition+1

    elif arrayName == 'ALL':
        print("Fetching all arrays ", n_arrays_names) 
        jstart, jend = 0,n_arrays
        #continue
    else: 
        print("Not found array name ", arrayName) 
        print("use arrayName='ALL' or use one among available: ", n_arrays_names) 
        raise NameError('Error - Check the message above')  
  
    #sample from struct to unstruct
    structfileVTUData = [] 
    arrays_names = []
    for i in range(jstart,jend):
            print("Converting ",n_arrays_names[i]," from unstruct to struct")
            unstructArr = np.array(data.GetPointData().GetArray(i))
            structArr = interpolate.griddata(coordinates, unstructArr, coordinates_structList, method='linear', fill_value=0.0, rescale=False)

            if (len(unstructArr.shape)>1):
                subs = unstructArr.shape[-1]
                structfileVTUData.append( structArr.reshape(nx,ny,subs) )  
                namethis = n_arrays_names[i]
                arrays_names.append(namethis)
            else:
                structfileVTUData.append( structArr.reshape(nx,ny) )  
                namethis = n_arrays_names[i]
                arrays_names.append(namethis)
            print("_/ Converted ",n_arrays_names[i]," from unstruct to struct")

    if returnDict:
            zippedReturnArr = zip(n_arrays_names,structfileVTUData)
            print("Returning Dict ")
            return dict(zippedReturnArr) , coordinates_structList;
    else:
            print("NOT Returning Dict, returning array_names with array of shape ", np.array(structfileVTUData).shape)
            return np.array(structfileVTUData),arrays_names, coordinates_structList;

#structDict, coordinates_structList = TransportVTU2Struct_2d(filenameVTU,arrayName='ALL', nx=36, ny=30, outputFile=None, returnDict=True)
#structDict, coordinates_structList = TransportVTU2Struct_2d(filenameVTU,arrayName='Vitesse', nx=36, ny=30, outputFile=None, returnDict=True) 

