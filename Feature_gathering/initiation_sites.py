import json

""" gets the initiation site of a single simulation """
def get_initiation_cite(simulation_folder, simulation):
    filepath = simulation_folder + simulation + "\\" + simulation + ".odb_initiation_info.json"
    f = open(filepath, 'r')
    data = json.load(f)
    f.close()
    initiation_cite = data[1][0]
    return initiation_cite
