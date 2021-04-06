[y,fs] = audioread('Audio_005.wav');
y = y';
player = audioplayer(y,fs);
%play(player);

freq = linspace(0,fs,size(y,2));



ts = 1/fs
total_time = (size(y,2)-1)/fs;
t = 0:ts:total_time;
s = spectrogram(y);
xlim([0 1500])
spectrogram(y,100,80,100,fs,'yaxis')
figure
plot(t,y)
grid minor
figure
plot(abs(fft(y)))
grid minor
xlim([0 5000])