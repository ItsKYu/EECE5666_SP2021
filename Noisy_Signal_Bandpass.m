clear;
clc;
%%Analyse the audio
%Note: This code only works if the person speaking has the dominant
%frequency in the audio file. It also assumes the value of human speech
%cannot exceed 200 Hz
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

for i = 1:length(freq)
    if freq(i) > 300;
        freq(i) = 0;
        amp(i) = 0;
    end 
end


[max_num, max_idx]=max(amp(:))
[P,Q]=ind2sub(size(amp),max_idx)

maxfreqoccurs = freq(Q)

fc1 = maxfreqoccurs - maxfreqoccurs/2;
fc2 = maxfreqoccurs + maxfreqoccurs/2;
[b,a] = butter(4,[fc1/(fs/2), fc2/(fs/2)],'bandpass')
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
time = linspace(0,16,length(t));

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



% audiowrite('result.wav',Y,fs);
% 
% [final,Fs] = audioread('sound_voice.wav');
% sound(final,Fs)