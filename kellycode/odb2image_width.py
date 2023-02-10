import sys
import os
from visualization import *
#Input = sys.argv[-1]

#The path to the simulation results folder to get images from. e.g. Para_2-56ft_PHI_23_THETA_-10
origin_file = "F:\\Jake\\test_folder"
origin_file = "F:\\Jake\\good_simies" 
#"C:\\Users\\Bjorn\\Desktop\\sfx\\sfx_matrix"
wireframe = False

#The path to the folder to store the images at
destination_file = "F:\\Jake\\test_images" #"C:\\Users\\Bjorn\\Desktop\\sfx\\sfx_matrix\\matrix_images"
destination_file = "C:\\Users\\u1056\\OneDrive\\Desktop\\bjorns pics"

file_type = '.odb'
file_name_list = []
dir_list = []
for root, dirs, files in os.walk(origin_file):
    # select file name
        for file in files:
            # check the extension of files
            if file.endswith(file_type):
                file_name_list.append (os.path.join(root, file))

from abaqus import *
from abaqusConstants import *

session.Viewport(name='Viewport: 1', origin=(0.0, 0.0), width=150.215621948242, 
    height=105.874992370605)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].maximize()
from caeModules import *
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
    referenceRepresentation=ON)
a = mdb.models['Model-1'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
session.journalOptions.setValues(replayGeometry=COORDINATE, recoverGeometry=COORDINATE)

for filepath in file_name_list:
    Input = os.path.basename(filepath)
    dir = os.path.dirname(filepath)
    new_dir = dir.split(origin_file,1)[1]
    model_name = Input.split(file_type)[0]

    mdb.ModelFromOdbFile(name=model_name, 
        odbFileName=filepath)
    model_name = model_name[0:38]

    session.viewports['Viewport: 1'].assemblyDisplay.setValues(
        optimizationTasks=OFF, geometricRestrictions=OFF, stopConditions=OFF)
    a = mdb.models[model_name].rootAssembly
    session.viewports['Viewport: 1'].setValues(displayedObject=a)


    #odb instead of cae
    myOdb = visualization.openOdb(path = filepath)
    session.viewports['Viewport: 1'].setValues(displayedObject=myOdb)
    session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=( CONTOURS_ON_DEF, ))
    
    #setting stress type to be max principal
    session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='S', outputPosition=INTEGRATION_POINT, refinement=(INVARIANT,
    'Max. Principal'))

    #set contour values
    session.viewports['Viewport: 1'].odbDisplay.contourOptions.setValues(numIntervals=10, 
    # maxAutoCompute=OFF, maxValue=0.10,
    maxAutoCompute=OFF, maxValue=23000, 
    minAutoCompute=OFF, minValue=0.0,)

    session.viewports['Viewport: 1'].odbDisplay.commonOptions.setValues(
        visibleEdges=FREE)
    # remove plate
    leaf = dgo.LeafFromOdbElementMaterials(elementMaterials=("PLATE", ))
    session.viewports['Viewport: 1'].odbDisplay.displayGroup.remove(leaf=leaf)

        # remove Brain
    leaf = dgo.LeafFromOdbElementMaterials(elementMaterials=("BR#BRAIN", ))
    session.viewports['Viewport: 1'].odbDisplay.displayGroup.remove(leaf=leaf)

    #     # remove Occipital
    # leaf = dgo.LeafFromOdbElementMaterials(elementMaterials=("SK-NORPA#OCCIPITAL", ))
    # session.viewports['Viewport: 1'].odbDisplay.displayGroup.remove(leaf=leaf)

    #     # remove Parietal
    # leaf = dgo.LeafFromOdbElementMaterials(elementMaterials=("SK-NORPA#PARIETAL", ))
    # session.viewports['Viewport: 1'].odbDisplay.displayGroup.remove(leaf=leaf)

    #     # remove Suture
    # leaf = dgo.LeafFromOdbElementMaterials(elementMaterials=("SU#SUTURE", ))
    # session.viewports['Viewport: 1'].odbDisplay.displayGroup.remove(leaf=leaf)


    #         # remove skull
    # leaf = dgo.LeafFromOdbElementMaterials(elementMaterials=("SK-NORPA#SKULL", ))
    # session.viewports['Viewport: 1'].odbDisplay.displayGroup.remove(leaf=leaf)
       
    
    

    #Make background white
    session.graphicsOptions.setValues(backgroundStyle=SOLID, 
        backgroundColor='#FFFFFF')

    session.viewports['Viewport: 1'].viewportAnnotationOptions.setValues(triad=OFF, 
        legend=OFF, title=OFF, state=OFF, annotations=OFF, compass=OFF)
    session.viewports['Viewport: 1'].assemblyDisplay.meshOptions.setValues(
        meshVisibleEdges=FREE)
    session.viewports['Viewport: 1'].assemblyDisplay.geometryOptions.setValues(
        datumPoints=OFF, datumAxes=OFF, datumPlanes=OFF, datumCoordSystems=OFF)
    session.viewports['Viewport: 1'].view.fitView()
    session.viewports['Viewport: 1'].view.setValues(nearPlane=404.993, 
        farPlane=735.403, width=518.806, height=179.784, cameraPosition=(289.861, 
        -151.741, 544.96), cameraUpVector=(0.175828, -0.672908, -0.718526), 
        cameraTarget=(123.103, 44.3063, 69.9004))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=357.981, 
        farPlane=633.524, width=219.16, height=105.751, viewOffsetX=14.7605, 
        viewOffsetY=14.3722)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=361.481, 
        farPlane=630.024, width=184.351, height=88.9544, viewOffsetX=0, 
        viewOffsetY=0)
    
    image_name = model_name + '.png'
    destination_dir = destination_file+new_dir
    if not os.path.exists(destination_dir):
        os.mkdir(destination_dir)
    session.printOptions.setValues(reduceColors=False)
    session.printToFile(fileName=destination_dir+'/'+image_name, format=PNG, canvasObjects=(
        session.viewports['Viewport: 1'], ))

