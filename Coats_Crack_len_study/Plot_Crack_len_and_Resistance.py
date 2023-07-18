import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas

def get_data(filename):
    new_crack_lens = {}
    old_crack_lens = {}
    Resistances = {}
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            print(row)  # Replace with your desired logic for each row
            step, uci, new_crack_length, old_crack_length, K1c_parallel  = int(row[0]), int(row[1]), float(row[2]), float(row[3]), float(row[4])
            Resistances[(step, uci)] = K1c_parallel
            new_crack_lens[(step, uci)] = new_crack_length
            old_crack_lens[(step, uci)] = old_crack_length
    return Resistances, new_crack_lens, old_crack_lens

def plot_data(plot1, plot2, ylabel):
    # Extract the keys and values from the dictionary
    x1 = list(plot1.keys())
    x1 = [str(t) for t in x1]
    y1 = list(plot1.values())
    # Extract the keys and values from the dictionary
    x2 = list(plot2.keys())
    x2 = [str(t) for t in x2]
    y2 = list(plot2.values())
    
    # Create the plot
    plt.plot(x1, y1, marker='o', c='r', label='Using New crack len to calculate R')
    # Create the plot
    plt.plot(x2, y2, marker='o', c='g', label='Using Old crack len to calculate R')

    # Set labels and title
    plt.xlabel('(Step, UCI)')
    plt.ylabel(ylabel)
    plt.title(ylabel + ' over time')
    plt.legend()

    # Display the plot
    plt.show()
    plt.close()


    
def populate_gaps_in_data(New_Resistances, Old_Resistances):
    New_keys = list(New_Resistances.keys())
    Old_keys = list(Old_Resistances.keys())
    New_R = 0; Old_R = 0
    j = 0
    old_j = -1
    for i in range(len(New_keys)):
        new_step = New_keys[i][0]; new_uci = New_keys[i][1]
        old_step = Old_keys[j][0]; old_uci = Old_keys[j][1] 
        if(New_R != 0 and Old_R != 0 and old_j != j):
            old_j = j
            Prev_New_R = New_R; Prev_Old_R = Old_R 
        New_R = New_Resistances[(new_step, new_uci)]; Old_R = Old_Resistances[(old_step, old_uci)]
        print('new = ' + str((new_step, new_uci)))
        print('old = ' + str((old_step, old_uci)))
        if(new_step == old_step and new_uci == old_uci):
            j += 1
            continue
        elif(new_step > old_step):
            New_Resistances[(old_step, old_uci)] = Prev_New_R
            print('here')
        elif(new_step < old_step):
            Old_Resistances[(new_step, new_uci)] = Prev_Old_R
            print('here')
        elif(new_uci > old_uci):
            New_Resistances[(old_step, old_uci)] = Prev_New_R
            print('here')
        elif(new_uci < old_uci):
            Old_Resistances[(new_step, new_uci)] = Prev_Old_R
            print('here')
        else:
            print('unknown case')
    
    return New_Resistances, Old_Resistances
            
# Custom key function to extract the step and uci values from the tuple keys
def sort_key(key):
    step, uci = key
    return (step, uci)

def sort_dict_by_step_then_uci(dict):
    sorted_keys = sorted(dict.keys(), key=sort_key)
    # Creating a new ordered dictionary using the sorted keys
    ordered_dict = {key: dict[key] for key in sorted_keys}
    return ordered_dict

New_crack_len_filename = '/Users/jakehirst/Desktop/sfx/sfx_ML_code/sfx_ML/Coats_Crack_len_study/new_crack_len_simmy_lengths_and_K1c.csv'
New_Resistances, New_new_crack_lens, New_old_crack_lens = get_data(New_crack_len_filename)
Old_crack_len_filename = '/Users/jakehirst/Desktop/sfx/sfx_ML_code/sfx_ML/Coats_Crack_len_study/old_crack_len_simmy_lengths_and_K1c.csv'
Old_Resistances, Old_new_crack_lens, Old_old_crack_lens = get_data(Old_crack_len_filename)

Old_Resistances, Old_new_crack_lens, Old_old_crack_lens = sort_dict_by_step_then_uci(Old_Resistances), sort_dict_by_step_then_uci(Old_new_crack_lens), sort_dict_by_step_then_uci(Old_old_crack_lens)
New_Resistances, New_new_crack_lens, New_old_crack_lens = sort_dict_by_step_then_uci(New_Resistances), sort_dict_by_step_then_uci(New_new_crack_lens), sort_dict_by_step_then_uci(New_old_crack_lens)
del New_Resistances[(31,0)]; del New_new_crack_lens[(31,0)]; del New_old_crack_lens[(31,0)]

New_Resistances, Old_Resistances = populate_gaps_in_data(New_Resistances, Old_Resistances)
New_new_crack_lens, Old_new_crack_lens = populate_gaps_in_data(New_new_crack_lens, Old_new_crack_lens)
New_old_crack_lens, Old_old_crack_lens = populate_gaps_in_data(New_old_crack_lens, Old_old_crack_lens)

#sort again
Old_Resistances, Old_new_crack_lens, Old_old_crack_lens = sort_dict_by_step_then_uci(Old_Resistances), sort_dict_by_step_then_uci(Old_new_crack_lens), sort_dict_by_step_then_uci(Old_old_crack_lens)
New_Resistances, New_new_crack_lens, New_old_crack_lens = sort_dict_by_step_then_uci(New_Resistances), sort_dict_by_step_then_uci(New_new_crack_lens), sort_dict_by_step_then_uci(New_old_crack_lens)







plot_data(New_Resistances, Old_Resistances, 'Resistances')
plot_data(New_new_crack_lens, New_old_crack_lens, 'New and Old Crack Lengths from New Crack Lengths simulation')
plot_data(New_old_crack_lens, Old_old_crack_lens, 'Old Crack Lengths')
