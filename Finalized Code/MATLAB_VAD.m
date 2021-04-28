clear;
clc;
%%Analyse the audio
%This program takes source code directly from MATLAB's Voice Activity
%Detection Toolbox 

%Read audio file from desired location 
rootdirectory = 'C:\Users\nguid\Documents\EECE 5666\Project\VAD Stuff';
fileReader = dsp.AudioFileReader('harvard_11_1_FIR.wav');

%Determine Sampling Frequency 
fs = fileReader.SampleRate;

fileReader.SamplesPerFrame = ceil(10e-3*fs);

%Implement MATLAB's VAD
VAD = voiceActivityDetector;

%Output results on a timescope with desired parameters 
scope = timescope( ...
    'NumInputPorts',2, ...
    'SampleRate',fs, ...
    'TimeSpanSource','Property','TimeSpan',4.5, ...
    'BufferLength',3*fs, ...
    'YLimits',[-.3 1.5], ...
    'TimeSpanOverrunAction','Scroll', ...
    'ShowLegend',true, ...
    'ChannelNames',{'Audio','Probability of speech presence'});
deviceWriter = audioDeviceWriter('SampleRate',fs)


%Output instances where the probability is above a desired threshold (.998)
while ~isDone(fileReader)
    audioIn = fileReader();
    probability = VAD(audioIn);
    
   
   for i = 1:length(audioIn)
    if probability >.998
        
       probcount(i:1) = probability

    end
   end 
    scope(audioIn,probability*ones(fileReader.SamplesPerFrame,1))
    deviceWriter(audioIn);
    
end

