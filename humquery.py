
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

from fastdtw import fastdtw


print('Import successful')

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

def downsample_audio(input_array, sampling_rate):
    return input_array[::sampling_rate]

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

#Store the midi values as a relative sequence of numbers
def midi_to_numbers(input_array):
    midi_relative_numbers = [0]
    return np.diff(input_array)
    #import pdb;pdb.set_trace()
    # for i in range(1,len(input_array)):
    #     if(input_array[i]==0):
    #         midi_relative_numbers.append(input_array[i])
    #     elif(input_array[i]>input_array[i-1]):
    #         if(abs(input_array[i]-input_array[i-1])>1):
    #             #value = int(abs(input_array[i]-input_array[i-1]))
    #             sequence = midi_relative_numbers[i-1] + 1
    #             midi_relative_numbers.append(sequence)
    #         else:
    #             midi_relative_numbers.append(midi_relative_numbers[i-1])
    #     elif(input_array[i]<input_array[i-1]):
    #         if(abs(input_array[i]-input_array[i-1])>1):
    #             #value = int(abs(input_array[i]-input_array[i-1]))
    #             sequence = midi_relative_numbers[i-1]-1
    #             midi_relative_numbers.append(sequence)
    #         else:
    #             midi_relative_numbers.append(midi_relative_numbers[i-1])
    #     else:
    #         midi_relative_numbers.append(midi_relative_numbers[i-1])
    # return midi_relative_numbers

#Remove silence frames as well as negative numbers from the frames
def process_all_files(directory_name):
    d = {}
    for file in os.listdir(directory_name):
        if file.endswith(".xlsx"):
            d[file] = process_file(os.path.join(directory_name, file))
            print(os.path.join(directory_name, file))
    return d

def remove_silence(input_array):
    query_sequence = []
    for i in range(0,len(input_array)):
        if(input_array[i]!=0):
            query_sequence.append(input_array[i])
    return query_sequence

def get_continuous_numbers(input_array):
    return input_array[np.insert(np.diff(input_array).astype(np.bool),0,True)]

def process_file(file_name):
    #import pdb;pdb.set_trace()
    pre_processed_query = pre_process(file_name)
    numpy_array = np.asarray(pre_processed_query)
    midi_values_query = convert_to_midi(numpy_array)
    midi_values_query = np.round((midi_values_query))
    query_without_silence = remove_silence(midi_values_query)
    return midi_to_string(query_without_silence)
    #downsampled = downsample_audio(query_without_silence,10)
    #midi_relative_query = midi_to_numbers(query_without_silence)
    #query_sequence = np.round(midi_relative_query)
    #non_repeated = get_continuous_numbers(midi_relative_query)
    return midi_relative_query
    #return midi_relative_query



#def segment_audio(input_array):
#    segment_to_pattern = {}
    
''' This module is not intended to run from interpreter.
        Instead, call the functions from your main script.
        from lcs import rlcs as rlcs

        score, diag, cost = rlcs.rlcs(X, Y, tau_dist,  delta)
        segment = rlcs.backtrack(X, Y, score, diag, cost)'''


if __name__ == '__main__':
    file_audio = r'/Users/jyotinarang/Desktop/querybyhumming/QueryByHumming/ExcelFormatAudio/1.xlsx'
    file_query = r'/Users/jyotinarang/Desktop/querybyhumming/QueryByHumming/ExcelFormatHummed/1_hummed.xlsx'
    #Process the audio file
    #file_audio = r'q7_freq.xlsx'
    #another_test_file = r'/Users/jyotinarang/Desktop/querybyhumming/QueryByHumming/ExcelFormatAudio/1_vamp_mtg-melodia_melodia_melody.csv.xlsx'
    # #Process the query file
    #file_query = r'q7_query.xlsx'
    #another_file = r'file1.xlsx'
    #file = r'query_q1.csv'
    #audio_data = process_file(file_audio)
    #query_data = process_file(file_query)
    #import pdb;pdb.set_trace()
    audio_data = np.asarray(['S','U','S','D','S','D','S','S','D','S','S','U','U','D','S','S','U'])
    query_data = np.asarray(['D','S','S','U','U','D','S'])
    print(len(audio_data))
    print(len(query_data))
    print(query_data)
    distance_matrix = DTW.computeDistanceMatrix(query_data,audio_data)
    cost_matrix,trace_index = DTW.dtw(distance_matrix)
    p, q = DTW.traceback(cost_matrix, trace_index)
    #np.set_printoptions(threshold=np.nan)
    print(distance_matrix)
    print(p)
    print(q)
    print(len(p))
    #print(cost_matrix)
    #print(trace_index)
    #print(p)
    #print(q)
    #import pdb;pdb.set_trace()
    #audio_another_file = process_file(another_file)
    #audio_more = process_file(another_test_file)
    #check = pre_process(file)
    #np.set_printoptions(threshold=np.nan)
    #directory_name = '/Users/jyotinarang/Desktop/querybyhumming/QueryByHumming/ExcelFormatAudio/'
    #result = process_all_files(directory_name)
    # print(audio_data)
    # print(query_data)
    # distance, path = fastdtw(audio_data, query_data, dist=euclidean)
    # print(distance)
    # print(path)
    # distance, path = fastdtw(audio_another_file, query_data, dist=euclidean)
    # print(distance)
    # print(path)

    #score, diag, cost = rlcs.rlcs(query_data, audio_data)
    #segment = rlcs.backtrack(query_data, audio_data, score, diag, cost)
    #print(cost)
    #print(diag)
    #print(cost)
    #print(segment)
    # print('Another file')
    # score1, diag1, cost1 = rlcs.rlcs(query_data, audio_another_file)
    # segment1 = rlcs.backtrack(query_data, audio_another_file, score, diag, cost)
    # print(segment1)
    # print('------')
    # score2, diag2, cost2 = rlcs.rlcs(query_data, audio_another_file)
    # segment2 = rlcs.backtrack(query_data, audio_another_file, score, diag, cost)


