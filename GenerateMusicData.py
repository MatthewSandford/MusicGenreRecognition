import os
import json_handler
import song_class
import numpy

if __name__ == '__main__':

    print "Generating Music Data:"

    test_data = json_handler.load_json("C:\Users\Tobe\Documents\Programming\Python\Python2\Music Analysis\Training\TestData.json")
    training_data = json_handler.load_json("C:\Users\Tobe\Documents\Programming\Python\Python2\Music Analysis\Training\TrainingData.json")

    i = 0

    for file_name in os.listdir('C:\Users\Tobe\Documents\Programming\Python\Python2\Music Analysis\Music'):

        i += 1

        if file_name.endswith(".wav"):

            if i%2 == 0:
            
                if file_name not in test_data:

                    print "New Song:", file_name

                    song = song_class.song(file_name)
                    song.generate_descriptor()
       
                    test_data[file_name] = song.get_descriptor()

            else:

                if file_name not in training_data:

                    print "New Song:", file_name


                    song = song_class.song(file_name)
                    song.generate_descriptor()
       
                    training_data[file_name] = song.get_descriptor()
                    
        if i%50 == 0:
                                
            json_handler.save_to_json(test_data,"C:\Users\Tobe\Documents\Programming\Python\Python2\Music Analysis\Training\TestData.json")
            json_handler.save_to_json(training_data,"C:\Users\Tobe\Documents\Programming\Python\Python2\Music Analysis\Training\TrainingData.json")

    data =      {"BPM":[],
                "tempo":[],
                "speed":[],
                "BPM_uncertainty":[],
                "MFCC_mean":[],
                "MFCC_coef":[],
                "SSC":[],
                "complexity":[],
                "avg_freq_power":[],
                "layers":[],
                "centroid":[],
                "spectral_flux":[],
                "rolloff":[],
                "zeros":[],
                "max_freq":[],
                "freq_power":[],
                "volume":[]}

    for song in training_data:
        for feature in training_data[song]["descriptor"]: 
                data[feature] = [[] for i in range(0,len(training_data[song]["descriptor"][feature]))]
                
    for song in training_data:
        for feature in training_data[song]["descriptor"]:
            for i in range(0,len(training_data[song]["descriptor"][feature])):         
                data[feature][i].append(training_data[song]["descriptor"][feature][i])
    
    for feature in data:
        for i in range(0,len(data[feature])):

            minimum = min(data[feature][i])
            normal = max(data[feature][i])
            
            for song in test_data:
           
                value = test_data[song]["descriptor"][feature][i]
                test_data[song]["descriptor"][feature][i] = (value-minimum)/(normal-minimum) 

            for song in training_data:

                value = training_data[song]["descriptor"][feature][i]             
                training_data[song]["descriptor"][feature][i] = (value-minimum)/(normal-minimum) 
        
    json_handler.save_to_json(test_data,"C:\Users\Tobe\Documents\Programming\Python\Python2\Music Analysis\Training\TestData.json")
    json_handler.save_to_json(training_data,"C:\Users\Tobe\Documents\Programming\Python\Python2\Music Analysis\Training\TrainingData.json")
   
    print "Program Exiting"
