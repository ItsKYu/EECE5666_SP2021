clear;
clc;

%Design and IIR filter for processing input audio signal 

%Specify directory of audio file 
rootdirectory = 'C:\Users\nguid\Documents\EECE 5666\Project\VAD Stuff';

%Read input file 
[y,fs] = audioread('harvard_62_9.wav');
y = y';

%Play audio file out loud over computer speakers 
player = audioplayer(y,fs);
% play(player);

%Define time parameters 
ts = 1/fs;
total_time = (size(y,2)-1)/fs;
t = 0:ts:total_time;

m = length(y);
x1 = fft(y, m); %Implement the fast fourier transform to view frequency domain 
f = (0:m-1)*(fs/m);
amplitude = abs(x1)/m;

%Specify plot parameters 
amp = amplitude(1:(m/2));
freq = f(1:(m/2));

%Plot original, unfiltered audio signal in frequency domain to extract
%desired features 
subplot(2,1,1)
plot(freq,amp); grid on;
title('Frequency Domain - Unfiltered Audio')
xlabel('Frequency')
ylabel('Amplitude')
xlim([0 2000])


%Filter parameters defined by visually inspecting the original signal
%These can be changed based on the user's desired spectrum 
Fp = 300/fs;
wp = Fp; 
Fs = 1000/fs;
ws = Fs;
As = 50; 
Ap = 0.5;

%Evaluate the cutoff frequency 
wc = (wp+ws)/2

%Establish parameters for a butterworth filter 
[N,wn] = buttord(wp,ws,1,50,'s')

%Calculate transfer function coefficients for a lowpass butterworth filter 
[b,a] = butter(N,wc,'low')
Sys = tf(b,a)

%Implement filter 
Y = filter(b,a,y);

m = length(Y);
x2 = fft(Y, m);
f = (0:m-1)*(fs/m);
amplitude = abs(x2)/m;

amp = amplitude(1:(m/2));
freq = f(1:(m/2));

%Plot the filtered audio in the frequency domain 
subplot(2,1,2)
plot(freq,amp); grid on;
title('Frequency Domain - filtered Audio')
xlabel('Frequency')
ylabel('Amplitude')
xlim([0 2000])

U = ifft(x2,m); %Perform inverse fourier transform 

freq = f(1:m);
time = linspace(0,16,length(t));

%Plot the unfiltered signal in the time domain 
figure()
subplot(2,1,1)
plot(time,y); grid on;
title('Time Domain - Unfiltered Signal')
xlabel('Time (s)')
ylabel('Amplitude')

%Plot the filtered signal in the time domain 
subplot(2,1,2)
plot(time,Y); grid on;
title('Time Domain - Filtered Signal')
xlabel('Time (s)')
ylabel('Amplitude')
figure()

%Uncomment this section to play filtered audio out loud 
% player1 = audioplayer(Y,fs);
% play(player1);


%Filter Specificaiton Plots 

%Determine filter parameters based on information above 
Fmax = 1000;
F = linspace(0,Fmax,200);
Om = 2*pi*F;
H = freqs(b,a);
Hmag = 10*abs(H);
Hdb = 10*20*log10(Hmag/max(Hmag));

%Calculate and plot the magnitude response of the filter 
plot(F,Hdb); grid on;
xlabel('Frequency (Hz)')
ylabel('Magnitude (dB)')
title('Log-Magnitude Response of Butterworth Filter')

%Write the filtered signal and save it as a new audio file 
filename = 'harvard_62_9_IIR.wav'
audiowrite(filename,Y,fs);









