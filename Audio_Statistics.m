%% Audio statistics
%
% Author: Ethan Marcello
%

% Read in custom file
[cleanAudio,Fs] = audioread("harvard_62_9.wav"); % Harvard Sentences list 11 # 1

fs = 8e3; % output frequency sampling
decimationFactor = Fs/fs;

L = floor(numel(cleanAudio)/decimationFactor);
cleanAudio = cleanAudio(1:decimationFactor*L);

% Create new obj to convert sample rate from 16k to 8k
src16 = dsp.SampleRateConverter("InputSampleRate",Fs, ...
                              "OutputSampleRate",fs, ...
                              "Bandwidth",7920);
% Convert the audio signal to 8kHz
cleanAudio = src16(cleanAudio);
reset(src16)


% Create randomnoise segment
randind = randi(numel(noise) - numel(cleanAudio), [1 1]);
noiseSegment = noise(randind : randind + numel(cleanAudio) - 1);
sound(noiseSegment)

%% Plot time domain
t = (1/fs) * (0:numel(noiseSegment)-1);

figure

subplot(2,1,1)
plot(t,noiseSegment(1:numel(noiseSegment)))
ylabel("Amplitude"); xlabel("Time (s)");
title("Noise")
grid on

subplot(2,1,2)
hist(noiseSegment,21)
title("Noise Amplitude Distribution")
ylabel("Number of Occurences"); xlabel("Amplitude");
grid on

% Fits the data to the closest approximate Gaussian
PD = fitdist(noiseSegment,'normal')

figure()
% Plot the Frequency Domain Representation
m = length(noiseSegment);
x = stft(noiseSegment, m);
f = (0:m-1)*(fs/m);
amplitude = abs(x)/m;
amp = amplitude(1:(m/2));
freq = f(1:(m/2));
plot(freq,amp)
title('Frequency Domain - Noise Signal')
xlabel('Frequency')
ylabel('Amplitude')
xlim([0 2000])

% STFT
windowLength = 256;
win = hamming(windowLength,"periodic");
overlap = round(0.75 * windowLength);
ffTLength = windowLength;
inputFs = 48e3; % fs is actually 16e3 from the audio sampling
fs = 8e3;
numFeatures = ffTLength/2 + 1;
numSegments = 8;

noiseSTFT = stft(noiseSegment,'Window',win,'OverlapLength',overlap,'FFTLength',ffTLength);
noiseSTFT = abs(noiseSTFT(numFeatures-1:end,:));

figure
plot(noiseSTFT)

%% Statistical comparison of clean audio to denoised audio

% The Denoising_deeplearning.m script must be run first to get the data
% output.

% Calculate the RMSE from noisy audio to clean (benchmark)
% get the residuals
residuals = noisyAudio - cleanAudio;
value_range = max(noisyAudio)-min(noisyAudio)
% square and take the mean
sq_mean = mean(residuals.^2);
noisy_RMSE = sqrt(sq_mean)

noisy_stderr = std(residuals);


% Calculate the RMSE for Fully Connected
% get the residuals
residuals = denoisedAudioFullyConnected - cleanAudio;
% square and take the mean
sq_mean = mean(residuals.^2);
FC_RMSE = sqrt(sq_mean)

FC_stderr = std(residuals);

% Calculate the RMSE for Convolutional
% get the residuals
residuals = denoisedAudioFullyConvolutional - cleanAudio;
% square and take the mean
sq_mean = mean(residuals.^2);
CL_RMSE = sqrt(sq_mean)

CL_stderr = std(residuals);



