import sys
import os
import time

#Input = sys.argv[-1]
origin_file = "F:\\Jake\\good_simies"
# #origin_file = "F:\\Jake\\good_simies_coats"
# origin_file = "C:\\Users\\u1056\\sfx\\simulation_results"
# origin_file = "C:\\Users\\u1056\\sfx\\simulation_results2"
# origin_file = "C:\\Users\\u1056\\sfx\\simulations_new_crack_len"

# origin_file = "Z:\\bjornssimies\\delta_k"
# origin_file = "Z:\\Brian_simies\\k_diff_simmies"
# origin_file = "C:\\Users\\u1056\\sfx\\Jimmy_cases_validation\\Weber_1"
# origin_file = "C:\\Users\\u1056\\sfx\\Jimmy_cases_validation\\deltaK_simulation_results"
# origin_file = "C:\\Users\\u1056\\sfx\\try_kdiff_2"
#origin_file = "C:\\Users\\u1056\\sfx\\Jimmy_cases_validation\\main_simulation_results"
#origin_file = "C:\\Users\\u1056\\sfx\\bad_simies"

wireframe = False
#destination_file = "C:\\Users\\u1056\\sfx\\images_sfx\\Original\\OG"
#destination_file = "C:\\Users\\u1056\\sfx\\images_sfx\\Original_from_test_matrix\\OG"
# destination_file = "C:\\Users\\u1056\\OneDrive\\Desktop\\test_images"
destination_file = "C:\\Users\\u1056\\sfx\\images_sfx\\Visible_cracks_new_dataset_\\OG"
# destination_file = "C:\\Users\\u1056\\sfx\\simulations_new_crack_len"
# destination_file = "C:\\Users\\u1056\\sfx\\images_sfx\\Original_new_dataset\\OG"
#destination_file = "C:\\Users\\u1056\\OneDrive\\Desktop\\bad_simie_images"

# destination_file = "F:\\Jake\\brian_simies\\pics"
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
    print(filepath)
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

    #Remove plate
    a = mdb.models[model_name].rootAssembly
    if filepath.find("Dynamic")==-1:
        e1 = a.instances['PL-1'].elements
        elements1 = e1.getSequenceFromMask(mask=('[#ffffffff #1ffff ]', ), )
        leaf = dgm.LeafFromMeshElementLabels(elementSeq=elements1)
        session.viewports['Viewport: 1'].assemblyDisplay.displayGroup.remove(leaf=leaf)
    else:
        set1 = mdb.models[model_name].rootAssembly.sets['PL-1_ALL_ELEMENTS']
        leaf = dgm.LeafFromSets(sets=(set1, ))
        session.viewports['Viewport: 1'].assemblyDisplay.displayGroup.remove(leaf=leaf)
        
        #Make gray
        session.viewports['Viewport: 1'].enableMultipleColors()
        session.viewports['Viewport: 1'].setColor(initialColor='#BDBDBD')
        cmap=session.viewports['Viewport: 1'].colorMappings['Part']
        session.viewports['Viewport: 1'].setColor(colorMapping=cmap)
        session.viewports['Viewport: 1'].disableMultipleColors()
        session.viewports['Viewport: 1'].enableMultipleColors()
        session.viewports['Viewport: 1'].setColor(initialColor='#BDBDBD')
        cmap = session.viewports['Viewport: 1'].colorMappings['Part']
        cmap.updateOverrides(overrides={'PART-1-1':(True, '#CCCCCC', 'Default', 
            '#CCCCCC')})
        session.viewports['Viewport: 1'].setColor(colorMapping=cmap)
        session.viewports['Viewport: 1'].disableMultipleColors()

    #For wireframe images
    if wireframe == True:
        session.viewports['Viewport: 1'].assemblyDisplay.setValues(
        renderStyle=WIREFRAME)

    #Make background white
    session.graphicsOptions.setValues(backgroundStyle=SOLID, 
        backgroundColor='#FFFFFF')

    session.viewports['Viewport: 1'].viewportAnnotationOptions.setValues(triad=OFF, 
        legend=OFF, title=OFF, state=OFF, annotations=OFF, compass=OFF)
    session.viewports['Viewport: 1'].assemblyDisplay.meshOptions.setValues(
        meshVisibleEdges=FEATURE)#TODO changing this from "FREE" to "FEATURE" to show more of the crack potentially
    session.viewports['Viewport: 1'].odbDisplay.commonOptions.setValues(
        edgeLineThickness=VERY_THIN)#TODO changing the look of the thickness of the crack options: "VERY_THIN, THIN, MEDIUM, THICK"
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


