clear;
clc;
%%Analyse the audio
rootdirectory = 'C:\Users\nguid\Documents\EECE 5666\Project\Audio Files';
fileReader = dsp.AudioFileReader('Audio_007.wav');

fs = fileReader.SampleRate;

fileReader.SamplesPerFrame = ceil(10e-3*fs);

VAD = voiceActivityDetector;

scope = timescope( ...
    'NumInputPorts',2, ...
    'SampleRate',fs, ...
    'TimeSpanSource','Property','TimeSpan',13, ...
    'BufferLength',3*fs, ...
    'YLimits',[-1.5 1.5], ...
    'TimeSpanOverrunAction','Scroll', ...
    'ShowLegend',true, ...
    'ChannelNames',{'Audio','Probability of speech presence'});
deviceWriter = audioDeviceWriter('SampleRate',fs)



while ~isDone(fileReader)
    audioIn = fileReader();
    probability = VAD(audioIn);
   
    if probability >.998
        probcount = probability
    end
    
    scope(audioIn,probability*ones(fileReader.SamplesPerFrame,1))
    deviceWriter(audioIn);
    
end

