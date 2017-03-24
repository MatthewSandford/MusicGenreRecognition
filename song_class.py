import numpy
import pylab
import scipy
from scipy import signal
import scipy.io.wavfile
import python_speech_features
from operator import itemgetter
import json_handler
import array
import math
import time
import copy
import os

class song:

    def __init__(self, title):

        if title.endswith(".wav"):

            self.title = title
            self.genre = self.title[:self.title.index(".")]
            self.sample_rate,self.data = scipy.io.wavfile.read("C:\Users\Tobe\Documents\Programming\Python\Python2\Music Analysis\Music\\" + self.title)
            self.wave = self.data[:,0]
            self.windows = 40
            self.calculate_fft()
           
        else:

            print "Incorrect File Type"

    def generate_descriptor(self):
        
        self.descriptor = {}
        self.calculate_mfcc()
        self.calculate_ssc()
        self.calculate_bpm()
        self.calculate_frequency_power()
        self.calculate_centriod()
        self.calculate_volume()
        self.calculate_spectral_flux()
        self.calculate_zeros()
        self.calculate_complexity()
        self.calculate_layers()     
        self.calculate_rolloff()
            
    def calculate_fft(self):
        
        start = time.time()
        
        self.sample_rate = 44100
        self.samples = len(self.data)
        self.song_time = self.samples/self.sample_rate
        
        self.frequency_range = scipy.arange(self.samples)/self.song_time      
        self.fourier = numpy.fft.fft(self.wave)
        self.fourier = self.fourier[range(self.samples / 2)] / max(self.fourier[range(self.samples / 2)])
        self.abs_fourier = [abs(item) for item in self.fourier]
        self.power, freqs, bins, im = pylab.specgram(self.wave, NFFT=512, Fs=self.sample_rate, noverlap=10)
        
        end = time.time()

        #print "FFT: ", (end - start)
        
    def picket_fence(self,interval,length):

        signal = numpy.zeros(length)

        for i in range(0,length/interval):
            signal[i*interval] = 1
            
        return signal

    def plot_signal(self,signal):

        timp=len(signal)/44100.
        time=numpy.linspace(0,timp,len(signal))

        pylab.plot(time,signal)
        pylab.show()

    def moving_average(self,data,window_size):

        window = numpy.ones(window_size)/float(window_size)
        return numpy.convolve(data, window, 'same')

    def low_pass(self,data,scale):

        data = numpy.array(data)
        B, A = signal.butter(scale*2, 0.5, output = 'ba')
        filtered = numpy.convolve(data, B, 'same')
        
        return filtered

    def expand_function(self,function,scale):
        
        new_function = []

        for i in range(0,scale*len(function)):

            if i%scale == 0:

                new_function.append(function[i/scale])

            else:

                new_function.append(0)

        new_function = self.low_pass(new_function,scale)
        
        return new_function

    def reduce_function(self,function,scale):

        function = self.low_pass(function,scale)
        
        new_function = []

        for i in range(0,len(function)):

            if i%scale == 0:

                new_function.append(function[i/scale])

        new_function = self.low_pass(new_function,scale)
        
        return new_function

    def extract_signal_info(self,signal):
        
        mean = numpy.mean(signal)
        variance = numpy.var(signal)
        maximum = max(signal)
        minimum = min(signal)

        return [mean, variance, maximum, minimum]

    def find_peaks(self,sig,sensativity,window_size):

        length = len(sig)

        average = self.moving_average(sig,length/5)
        average = [item*sensativity for item in average]

        x = []
        y = []
        
        for i in range(0,length):

            amplitude = sig[i]

            if amplitude > average[i]:

                before = i-window_size
                after = i+window_size

                before = max(0, min(before, length))
                after = max(0, min(after, length))

                peak = 1
                
                for j in range(before,after):

                    if sig[j] > amplitude and j != i:

                        peak = 0
                        break

                if peak:

                    x.append(i)
                    y.append(sig[i])
                
        return x, y
    
    def calculate_bpm(self):

        start = time.time()
        
        diff = [ numpy.diff(item) for item in self.power]
        diff = numpy.array(diff)       
        data = diff.sum(axis=0)
        data = [item.clip(min=0)  for item in data]
        data /= max(data)
        average = self.moving_average(data,len(data))
        data = [(data[i] - average[i]).clip(min=0) for i in range(0,len(data))]
        data = list(data)
        
        interval = len(data)/(self.song_time*4)
        x, y = self.find_peaks(data,1,interval)

        #Set Tempo
        self.tempo = len(x)*60/self.song_time
        
        big_data = self.expand_function(data,4)

        data_length = len(big_data)

        bpms = []

        for i in range(45,215):

            interval = (data_length*60)/(self.song_time*i)           
            picket = self.picket_fence(interval,(interval*5)+1)           
            sig = numpy.convolve(big_data,picket)
            sig = list(sig)           
            sig.sort()           
            bpms.append(numpy.mean(sig[len(sig)*95/100:]))
            
        x, y = self.find_peaks(bpms,1.02,15)

        #Set Uncertanty
        self.bpm_uncertainty = len(x)
        
        #amp = [i for i in range(0,len(bpms))]
        #pylab.show()
        #pylab.scatter(x,y)
        #pylab.plot(amp,bpms)
        #pylab.show() 

        points  = [(x[i],y[i]) for i in range(0,len(x))]        
        points = [points[i] for i in range(0,len(points)) if points[i][0] > 45 and points[i][0] < 200]
        
        #Set Speed
        self.speed = max(points,key=itemgetter(1))[0]

        best_score = 100
        best_bpm = 120 

        for point in points:

            if abs(self.tempo-point[0]) < best_score:
                best_score = abs(self.tempo-point[0])
                best_bpm = point[0]
                
        self.bpm = best_bpm

        self.descriptor["BPM"] = [self.bpm]
        self.descriptor["tempo"] = [self.tempo]
        self.descriptor["speed"] = [self.speed]
        
        self.descriptor["BPM_uncertainty"] = [self.bpm_uncertainty]
        
        end = time.time()

        #print "BPM: ", (end - start)
        
    def calculate_mfcc(self):

        start = time.time()

        mfcc = numpy.array(python_speech_features.base.mfcc(self.wave,self.sample_rate,numcep=20))
        self.mfcc = [item for item in list(mfcc.mean(axis=0))]
        
        self.mfcc_coef = []
        
        for i in range(0,5):
            self.mfcc_coef.extend(mfcc[i]) 

        self.descriptor["MFCC_mean"] = self.mfcc
        self.descriptor["MFCC_coef"] = self.mfcc_coef

        end = time.time()

        #print "MFCC: ", (end - start)

    def calculate_ssc(self):
        
        start = time.time()
        
        ssc = python_speech_features.base.ssc(self.wave,self.sample_rate,nfilt=10)
        self.ssc = [item for item in list(ssc[0])]

        self.descriptor["SSC"] = self.ssc

        end = time.time()

        #print "SSC: ", (end - start)
        
    def calculate_complexity(self):

        start = time.time()

        freq_power_avg = self.reduce_function(self.abs_fourier,2)
        power_avg = numpy.mean(freq_power_avg)
        
        self.complexity = 1

        for avg in freq_power_avg:
            if avg > power_avg:
             self.complexity += 1

        self.complexity = self.complexity*100/len(freq_power_avg)
        self.average_frequency_power = power_avg

        self.descriptor["complexity"] = [self.complexity]
        self.descriptor["avg_freq_power"] = [self.average_frequency_power]
        
        end = time.time()

        #print "Complexity: ", (end - start)
        
    def calculate_layers(self):
        
        start = time.time()
        
        x, y = self.find_peaks(self.reduce_function(self.abs_fourier,10),3,10)
        
        self.layers = len(y)
        self.descriptor["layers"] = [self.layers]

        end = time.time()

        #print "Layers: ", (end - start)
            
    def calculate_centriod(self):

        start = time.time()

        nom = 0
        dom = 0

        for i in range(0,len(self.abs_fourier)):

            nom += self.abs_fourier[i]*i
            dom += self.abs_fourier[i]

        self.centroid = nom/dom
        self.descriptor["centroid"] = [self.centroid]

        end = time.time()

        #print "Centriod: ", (end - start)
            
    def calculate_spectral_flux(self):

        start = time.time()
        
        diff = numpy.diff(self.abs_fourier)
        windows = 40
        window_size = len(diff)/windows
        spectral_flux = []
        
        for j in range(0,windows):

            spectral_flux.append(sum([abs(diff[i+j*window_size]) for i in range(0,window_size)])/window_size)

        self.spectral_flux = self.extract_signal_info(spectral_flux)

        self.descriptor["spectral_flux"] = self.spectral_flux

        end = time.time()

        #print "Spectral Flux: ", (end - start)
        
    def calculate_rolloff(self):
        
        start = time.time()
        
        iteration = 0
        total = 0
        cap = sum(self.abs_fourier)*0.7

        while total < cap:

            iteration+=1
            total+=self.abs_fourier[iteration]

        self.rolloff = iteration
        self.descriptor["rolloff"] = [self.rolloff]

        end = time.time()

        #print "Rolloff: ", (end - start)
        
    def calculate_zeros(self):

        start = time.time()
            
        data = self.wave
        window_size = len(self.wave)/40
        zeros = []

        for j in range(0,40):

            zeros.append(len(numpy.where(numpy.diff(numpy.sign(data[j*window_size:(j+1)*window_size])))[0]))

        self.zeros = self.extract_signal_info(zeros)

        self.descriptor["zeros"] = self.zeros

        end = time.time()

        #print "Zeros: ", (end - start)
        
    def calculate_frequency_power(self):

        start = time.time()

        bins = 10
        bin_size = len(self.power)/bins
        
        self.frequency_power = []

        for i in range(0,bins):

                self.frequency_power.append(self.extract_signal_info([item for item in list(self.power[bin_size*i:bin_size*(i+1)].mean(axis=0))]))
        
        self.max_frequency = self.frequency_range[self.abs_fourier.index(max(self.abs_fourier))]
            
        self.descriptor["max_freq"] = [self.max_frequency]

        self.descriptor["freq_power"] = []

        for item in self.frequency_power:
            for value in item:
                self.descriptor["freq_power"].append(value)
        
        end = time.time()

        #print "Frequency Power: ", (end - start)

    def calculate_volume(self):

        start = time.time()
        
        data = self.wave
        data = [abs(item) for item in data]
        window_size = len(self.wave)/40

        volumes = []

        for j in range(0,40):
        
            volumes.append(numpy.mean(self.wave[j*window_size:(j+1)*window_size]))

        self.volumes = self.extract_signal_info(volumes)

        self.descriptor["volume"] = self.volumes

        end = time.time()

        #print "Volume: ", (end - start)

    def get_genre(self):

        return self.genre

    def get_descriptor(self):

        return {"genre":self.genre,"descriptor":self.descriptor}
    
#asd = song("blues.00000.wav")
#asd.generate_descriptor()
#print asd.get_descriptor()


