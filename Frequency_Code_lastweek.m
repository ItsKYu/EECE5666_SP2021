close all; clear all; clc;
%%Analyse the audio
rootdirectory = 'C:\Users\nguid\Documents\EECE 5666\Project\Audio Files'
Audio = 'Audio_005.wav';
[y,Fs] = audioread(fullfile(rootdirectory, Audio));
spectrogram(y) % plot the spectrogram of the signal
figure()

% Plot the Time Domain Representation
t = (0:1/Fs:(length(y)-1)/Fs);
subplot(2,1,1)
plot(t,y)
title('Time Domain - Unfiltered Audio')
xlabel('Time (seconds)')
ylabel('Amplitude')
xlim([0 t(end)])

% Plot the Frequency Domain Representation
m = length(y);
x = fft(y, m);
f = (0:m-1)*(Fs/m);
amplitude = abs(x)/m;
subplot(2,1,2)
amp = amplitude(1:(m/2));
freq = f(1:(m/2));
plot(freq,amp)
title('Frequency Domain - Unfiltered Audio')
xlabel('Frequency')
ylabel('Amplitude')
xlim([0 2000])

% idx=[]; %get the index of all values classified in category 1.
% for i = 1: length(amp)
%   if amp(i)<.0001, idx = [idx,i];
%   end
% end


