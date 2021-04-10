clear;
clc;
%%Analyse the audio
rootdirectory = 'C:\Users\nguid\Documents\EECE 5666\Project\Audio Files';

[y,fs] = audioread('sentences_static.wav');
y = y';
player = audioplayer(y,fs);
% play(player);


ts = 1/fs;
total_time = (size(y,2)-1)/fs;
t = 0:ts:total_time;
% s = spectrogram(y);
% xlim([0 1500])
% spectrogram(y,100,80,100,fs,'yaxis')

plot(t,y)
xlabel('time')
ylabel('Amplitude')
grid minor

m = length(y);
x1 = fft(y, m);
f = (0:m-1)*(fs/m);
amplitude = abs(x1)/m;

amp = amplitude(1:(m/2));
freq = f(1:(m/2));

subplot(2,1,1)
plot(freq,amp)
title('Frequency Domain - Unfiltered Audio')
xlabel('Frequency')
ylabel('Amplitude')
xlim([0 2000])


[b,a] = butter(9,0.1)
% freqz(b,a)


 Y = filter(b,a,y);

m = length(Y);
x2 = fft(Y, m);
f = (0:m-1)*(fs/m);
amplitude = abs(x2)/m;

amp = amplitude(1:(m/2));
freq = f(1:(m/2));

subplot(2,1,2)
plot(freq,amp)
title('Frequency Domain - filtered Audio')
xlabel('Frequency')
ylabel('Amplitude')
xlim([0 2000])

U = ifft(x2,m);

freq = f(1:m);
time = linspace(0,16,249856);

figure()
subplot(2,1,1)
plot(time,y)
title('Unfiltered Signal')
xlabel('time')
ylabel('Amplitude')
grid minor
subplot(2,1,2)
plot(time,Y)
title('Filtered Signal')
xlabel('time')
ylabel('Amplitude')



audiowrite('result.wav',Y,fs);

[final,Fs] = audioread('sound_voice.wav');
sound(final,Fs)