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
    plt.plot(x1, y1, marker='o', c='r')
    # Create the plot
    plt.plot(x2, y2, marker='o', c='g')

    # Set labels and title
    plt.xlabel('(Step, UCI)')
    plt.ylabel(ylabel)
    plt.title(ylabel + ' over time')

    # Display the plot
    plt.show()
    
def populate_gaps_in_data(New_Resistances, Old_Resistances):
    New_keys = list(New_Resistances.keys())
    Old_keys = list(Old_Resistances.keys())
    j = 0
    for i in range(len(New_keys)):
        new_step = New_keys[i][0]; new_uci = New_keys[i][1]
        old_step = Old_keys[j][0]; old_uci = Old_keys[j][1] 
        New_R = New_Resistances[(new_step, new_uci)]; Old_R = Old_Resistances[(old_step, old_uci)]
        print('new = ' + str((new_step, new_uci)))
        print('old = ' + str((new_step, new_uci)))
        if(new_step == old_step and new_uci == old_uci):
            j += 1
            continue
        elif( new_uci > old_uci or  new_step > old_step):
            print('here')
        elif( new_uci < old_uci or  new_step < old_step):
            print('here')
        else:
            print('unknown case')
            
            

New_crack_len_filename = '/Users/jakehirst/Desktop/sfx/sfx_ML_code/sfx_ML/Coats_Crack_len_study/new_crack_len_simmy_lengths_and_K1c.csv'
New_Resistances, New_new_crack_lens, New_old_crack_lens = get_data(New_crack_len_filename)
Old_crack_len_filename = '/Users/jakehirst/Desktop/sfx/sfx_ML_code/sfx_ML/Coats_Crack_len_study/old_crack_len_simmy_lengths_and_K1c.csv'
Old_Resistances, Old_new_crack_lens, Old_old_crack_lens = get_data(Old_crack_len_filename)

populate_gaps_in_data(New_Resistances, Old_Resistances)

# plot_data(New_Resistances, Old_Resistances, 'Resistances')

