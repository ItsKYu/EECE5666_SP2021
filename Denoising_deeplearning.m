%% Denoise Speech Using Deep Learning Networks
%  Mathworks article
%  link: https://www.mathworks.com/help/audio/ug/denoise-speech-using-deep-learning-networks.html
%
%  Adaptations made by Ethan Marcello
%  
%  UPDATE LOG:
%  4/11/2021
%  - Copied over and built initial script (parts Problem summary through
%       XX)
%  
%
%


%% Problem Summary

% Consider the clean speech signal
[cleanAudio,fs] = audioread("Audio_003.wav"); %Audio 3 is spoken word "Happy"
sound(cleanAudio,fs)
pause

% Add washing machine noise to speech signal. Set noise power such
% that SNR is zero dB
noise = audioread("WashingMachine-16-8-mono-1000secs.mp3");

% Extract a noise segment from a random location in the noise file
ind = randi(numel(noise) - numel(cleanAudio) + 1, 1, 1);
noiseSegment = noise(ind:ind + numel(cleanAudio) - 1);

speechPower = sum(cleanAudio.^2);
noisePower = sum(noiseSegment.^2);
noisyAudio = cleanAudio + sqrt(speechPower/noisePower) * noiseSegment;

sound(noisyAudio,fs); %listen to noisy audio

% Visualize the original and noisy signals
t = (1/fs) * (0:numel(cleanAudio)-1);

subplot(2,1,1)
plot(t,cleanAudio)
title("Clean Audio")
grid on

subplot(2,1,2)
plot(t,noisyAudio)
title("Noisy Audio")
xlabel("Time (s)")
grid on

%%

