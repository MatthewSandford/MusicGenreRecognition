from sklearn import datasets
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import json_handler
import pylab
import copy
import numpy

def get_data(json):

    features = ["BPM",
                "tempo",
                "speed",
                "BPM_uncertainty",
                "MFCC_mean",
                "MFCC_coef",
                "SSC",
                "complexity",
                "avg_freq_power",
                "layers",
                "centroid",
                "spectral_flux",
                "rolloff",
                "zeros",
                "max_freq",
                "freq_power",
                "volume"]
    
    features = ["MFCC_mean","MFCC_coef","SSC","avg_freq_power","centroid","spectral_flux","zeros","freq_power","volume"]

    x = []
    y = []
    
    for song in json:

        x.append(json[song]["genre"])

        temp = []

        for feature in features:
            for i in range(0,len(json[song]["descriptor"][feature])):
        
                temp.append(json[song]["descriptor"][feature][i])

        y.append(temp)
        
    return x, y

def PCA(training_data, test_data):

    dimensions = len(training_data[0])

    print dimensions

    training_data = numpy.array(training_data).transpose()
    test_data = numpy.array(test_data).transpose()
    
    C = numpy.cov(training_data)

    values, vectors = numpy.linalg.eig(C)
 
    eig_pairs = [(numpy.abs(values[i]), vectors[:,i]) for i in range(len(values))]
    
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    k = 100
    
    lines = []
    
    for i in range(0,k):
    
        lines.append(eig_pairs[i][1].reshape(dimensions,1))

    lines = tuple(lines)

    matrix_w = numpy.concatenate(lines, axis=1)

    training_data = matrix_w.T.dot(training_data).transpose()
    test_data = matrix_w.T.dot(test_data).transpose()

    return training_data, test_data
            
def test():
    
    training_target, training_data = get_data(training_data_json)
    test_target, test_data = get_data(test_data_json)

    training_data, test_data = PCA(training_data,test_data)

    #model = GaussianNB()
    model = MLPClassifier()
    #model = DecisionTreeClassifier()
    #model = LinearSVC(C=20)    
    #model = KNeighborsClassifier(n_neighbors = 10)
    
    model.fit(training_data,training_target)

    guess_list = []
    
    expected = test_target
    predicted = model.predict(test_data)
    
    correct = len([1 for k in range(0,len(predicted)) if predicted[k] == expected[k]])

    for i in range(0,len(expected)):
        guess_list.append((expected[i],predicted[i]))

    print correct*100/len(test_target), "%"
    print correct, "/", len(test_target)
    
    data = {}
    personal_data = {}
    personal_data["correct"] = correct
    personal_data["incorrect"] = len(expected)-correct
    personal_data["guesses"] = guess_list

    data["AI_Test"] = personal_data
    json_handler.save_to_json(data,"C:\Users\Tobe\Documents\Programming\Python\Python2\Music Analysis\Test Results\AI Test.json")

if __name__ == '__main__':
    
    test_data_json = json_handler.load_json("C:\Users\Tobe\Documents\Programming\Python\Python2\Music Analysis\Training\TestData.json")
    training_data_json = json_handler.load_json("C:\Users\Tobe\Documents\Programming\Python\Python2\Music Analysis\Training\TrainingData.json")
    
    test()
