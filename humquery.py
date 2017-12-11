
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
import rlcs as rlcs
import pandas as pd
import numpy as np
import math
import numpy as np
from scipy.spatial.distance import euclidean
import os
import DTW as DTW
import testEditDistance as ED



'''This function will remove -ve values and convert them to 0's'''
def pre_process(file):
    data = pd.read_excel(file)
    annotation = data['Annotation']
    new_array = []
    for i in range(len(annotation)):
        if(annotation[i]<=0):
            new_array.append(0)
        else:
            new_array.append(annotation[i])
    return new_array

#Convert the pre-proccessed data to MIDI
def convert_to_midi(input_array):
    array_to_midi = []
    for i in range(len(input_array)):
        if(input_array[i] == 0):
            array_to_midi.append(input_array[i])
        else:
            midi_value = 69 + 12*math.log(input_array[i]/440,2)
            array_to_midi.append(midi_value)
    return array_to_midi

'''This function was ideally used for downsampling. Not used anymore'''
def downsample_audio(input_array, sampling_rate):
    return input_array[::sampling_rate]

'''Convert the MIDI notes to a sequence of U,D and S notations'''
def midi_to_string(input_array):
    midi_string = []
    i = 0
    while(i<len(input_array)-1):
        if(input_array[i]==input_array[i+1]):
            midi_string.append('S')
        elif(input_array[i] < input_array[i+1]):
            midi_string.append('D')
        elif(input_array[i] > input_array[i+1]):
            midi_string.append('U')
        i = i+1
    return midi_string

#Store the midi values as a relative sequence of numbers'''
'''This was done by using the diff betweeen consecutive MIDI modes, however, this approach did not produce any 
results as the MIDI numbers were mostly same and the diff was turning out to be 0's in most cases'''
def midi_to_numbers(input_array):
    midi_relative_numbers = [0]
    return np.diff(input_array)


'''Process all the files in the given directory and return a dictionary of file:MIDI values'''
def process_all_files(directory_name):
    dictionary = {}
    for file in os.listdir(directory_name):
        if file.endswith(".xlsx"):
            dictionary[file] = process_file(os.path.join(directory_name, file))
            print(os.path.join(directory_name, file))
    return dictionary

''''Remove all silence frames from the computed string'''
def remove_silence(input_array):
    query_sequence = []
    for i in range(0,len(input_array)):
        if(input_array[i]!=0):
            query_sequence.append(input_array[i])
    return query_sequence

'''This function was being tried for getting segment information. However, this was discarded because it was hard to 
get segment information on the basis of silence''' 
def get_segments(input_array):
    #import pdb;pdb.set_trace()
    data = {}
    seg_number = 1
    for i in range(len(input_array)):
        if(input_array[i] == 0):
            continue;
        if(input_array[i]>0):
            data[seg_number] = []
            while(input_array[i]>0):
                data[seg_number].append(input_array[i])
                i = i+1
        seg_number = seg_number+1
    return data

'''Function not in use anymore'''
# def get_largest_segments(input_dic):
#     for key in input_dic:
#         segment = input_dic[key]
#         print(len(segment))

def get_continuous_numbers(input_array):
    return input_array[np.insert(np.diff(input_array).astype(np.bool),0,True)]

''''Given an input file, perform all the steps i.e remove negative numbers, conversion to MIDI, rounding the numbers,
removing of silence frames and then converting the obtained MIDI numbers to string'''
def process_file(file_name):
    #import pdb;pdb.set_trace()
    pre_processed_query = pre_process(file_name)
    numpy_array = np.asarray(pre_processed_query)
    midi_values_query = convert_to_midi(numpy_array)
    midi_values_query = np.round((midi_values_query))
    #return midi_values_query
    query_without_silence = remove_silence(midi_values_query)    
    return midi_to_string(query_without_silence)




'''This is the main function calling other functions.
It takes hummed query and the audio files in the excel format as input and produces the normalized costs, path_lengths
as output'''
'''The other way to call it is by changing the hummed query and it will automatically produce the song with minimum cost as output'''

if __name__ == '__main__':
    #file_audio = r'/Users/jyotinarang/Desktop/querybyhumming/Final QBH Repository/ExcelFormatAudio/1_audio.xlsx'
    file_query = r'/Users/jyotinarang/Desktop/querybyhumming/Final QBH Repository/ExcelFormatHummed/10_hummed.xlsx'
    file_audio = r'/Users/jyotinarang/Desktop/querybyhumming/Final QBH Repository/ExcelFormatAudio/3_audio.xlsx'
    all_costs = []
    file_number = []
    actual_all_costs = []
    
    '''Pre-process all the hummed queries and audio files in the given directory'''
    audio_data = process_file(file_audio)
    query_data = process_file(file_query)

    all_data = process_all_files('/Users/jyotinarang/Desktop/querybyhumming/Final QBH Repository/ExcelFormatAudio')
    #import pdb;pdb.set_trace()
    
    for key,value in all_data.iteritems():
        audio_data = all_data[key]
        distance_matrix = DTW.computeDistanceMatrix(query_data,audio_data)
        cost_matrix,trace_index = DTW.dtw(distance_matrix)
        costs, path_lengths, cost_array = DTW.traceback(cost_matrix, trace_index)
        print 'Key is',key        
        minimum_cost = min(costs)
        print 'Cost is',minimum_cost
        max_path = max(path_lengths)
        min_cost_array = min(cost_array)
        print 'Actual cost is', min_cost_array
        all_costs.append(minimum_cost)
        actual_all_costs.append(min_cost_array)
        file_number.append(key)

    final_minimum = min(all_costs)
    index = all_costs.index(min(all_costs))
    print(file_number[index])
    print(final_minimum)



