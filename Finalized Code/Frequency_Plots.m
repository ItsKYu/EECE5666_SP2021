clear;
clc;

rootdirectory = 'C:\Users\nguid\Documents\EECE 5666\Project\VAD Stuff'; %Specify the location of the audio file

[y,fs] = audioread('harvard_11_1.wav'); %Read the file from the directory 
y = y';

ts = 1/fs; %Calculate the time 
total_time = (size(y,2)-1)/fs;
t = 0:ts:total_time; %Convert to total time 

m = length(y);
x1 = fft(y, m); %Perform fft on input signal 
%Adjust parameters 
f = (0:m-1)*(fs/m); 
amplitude = abs(x1)/m;

amp = amplitude(1:(m/2));
freq = f(1:(m/2));

%Plot frequency domain of original signal 
subplot(3,1,1)
plot(freq,amp)
title('Original Signal - Harvard_11_1','Interpreter','none')
xlabel('Frequency')
ylabel('Amplitude')
xlim([0 2000])

%
%
%
%
%
%

%Repeat process outlined about for FIR signal 
rootdirectory = 'C:\Users\nguid\Documents\EECE 5666\Project\VAD Stuff';

[y,fs] = audioread('harvard_11_1_FIR.wav');
y = y';

ts = 1/fs;
total_time = (size(y,2)-1)/fs;
t = 0:ts:total_time;

m = length(y);
x1 = fft(y, m);
f = (0:m-1)*(fs/m);
amplitude = abs(x1)/m;

amp = amplitude(1:(m/2));
freq = f(1:(m/2));

%Plot frequency domain of FIR signal 
subplot(3,1,2)
plot(freq,amp)
title('Denoised FIR Signal - Harvard_11_1','Interpreter','none')
xlabel('Frequency')
ylabel('Amplitude')
xlim([0 2000])

%
%
%
%
%
%

%Repeat process outlined above for IIR Signal 
rootdirectory = 'C:\Users\nguid\Documents\EECE 5666\Project\VAD Stuff';

[y,fs] = audioread('harvard_11_1_IIR.wav');
y = y';

ts = 1/fs;
total_time = (size(y,2)-1)/fs;
t = 0:ts:total_time;

m = length(y);
x1 = fft(y, m);
f = (0:m-1)*(fs/m);
amplitude = abs(x1)/m;

amp = amplitude(1:(m/2));
freq = f(1:(m/2));

%Plot frequency domain of IIR signal 
subplot(3,1,3)
plot(freq,amp)
title('Denoised IIR Signal - Harvard_11_1','Interpreter','none')
xlabel('Frequency')
ylabel('Amplitude')
xlim([0 2000])








