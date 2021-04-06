clear;
clc;
%%Analyse the audio
rootdirectory = 'C:\Users\nguid\Documents\EECE 5666\Project\Audio Files';
fileReader = dsp.AudioFileReader('Audio_007.wav');

VAD = voiceActivityDetector;

f0 = [];
while ~isDone(fileReader)
    x = fileReader();
    
    if VAD(x) > 0.99
        decision = pitch(x,fileReader.SampleRate)
%             "WindowLength",size(x,1), ...
%             "OverlapLength",0, ...
%             "Range",[100,800]);
    else
        decision = NaN;
    end
    f0 = [f0;decision];
end

t = linspace(0,(length(f0)*fileReader.SamplesPerFrame)/fileReader.SampleRate,length(f0));
plot(t,f0)
ylim([0,500])
ylabel('Fundamental Frequency (Hz)')
xlabel('Time (s)')
grid on