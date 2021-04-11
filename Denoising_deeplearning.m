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

%% Examine Training Dataset
% Download training files
url = 'http://ssd.mathworks.com/supportfiles/audio/commonvoice.zip';
downloadFolder = tempdir;
dataFolder = fullfile(downloadFolder,'commonvoice');

if ~exist(dataFolder,'dir')
    disp('Downloading data set (956 MB) ...')
    unzip(url,downloadFolder)
end

adsTrain = audioDatastore(fullfile(dataFolder,'train'),'IncludeSubfolders',true);

reduceDataset = true;
if reduceDataset
    adsTrain = shuffle(adsTrain);
    adsTrain = subset(adsTrain,1:1000);
end

[audio,adsTrainInfo] = read(adsTrain);

% sound(audio,adsTrainInfo.SampleRate); % to listen to speech singal

%Plot the speech signal
figure
t = (1/adsTrainInfo.SampleRate) * (0:numel(audio)-1);
plot(t,audio)
title("Example Speech Signal")
xlabel("Time (s)")
grid on

%% Deep Learning Notes Overview
% Need to downsample signals to 8kHz (4kHz is usually the max frequency of
% human speech). Reduces computational load of the network.
% Use Short Time Fourier Transform (STFT) to convert signal to frequency
% spectrum. Window length 256, 75% overalap, Hamming window.

%% STFT Targets and Predictors

windowLength = 256;
win = hamming(windowLength,"periodic");
overlap = round(0.75 * windowLength);
ffTLength = windowLength;
inputFs = 48e3; % fs is actually 16e3 from the audio sampling
fs = 8e3;
numFeatures = ffTLength/2 + 1;
numSegments = 8;

% Converts the 48kHz audio to 8kHz
src = dsp.SampleRateConverter("InputSampleRate",inputFs, ...
                              "OutputSampleRate",fs, ...
                              "Bandwidth",7920);
% Get audio file from the datastore
audio = read(adsTrain);
% Make sure audio length is a multiple of the sample rate converter
% decimation factor.
decimationFactor = inputFs/fs;
L = floor(numel(audio)/decimationFactor);
audio = audio(1:decimationFactor*L);
% Make conversion to 8kHz
audio = src(audio);
reset(src)

% Create random noise segment from washing machine noise vector
randind = randi(numel(noise) - numel(audio),[1 1]);
noiseSegment = noise(randind : randind + numel(audio) - 1);
% Add noise such that SNR is 0dB
noisePower = sum(noiseSegment.^2);
cleanPower = sum(audio.^2);
noiseSegment = noiseSegment .* sqrt(cleanPower/noisePower);
noisyAudio = audio + noiseSegment;

% generate mangitude STFT vectors from original and noisy audio signals.
cleanSTFT = stft(audio,'Window',win,'OverlapLength',overlap,'FFTLength',ffTLength);
cleanSTFT = abs(cleanSTFT(numFeatures-1:end,:));
noisySTFT = stft(noisyAudio,'Window',win,'OverlapLength',overlap,'FFTLength',ffTLength);
noisySTFT = abs(noisySTFT(numFeatures-1:end,:));

% generate 8-segment training predictor signals from the noisy STFT.
% overlap btwn consecutive preditors is 7 segments.
noisySTFT = [noisySTFT(:,1:numSegments - 1), noisySTFT];
stftSegments = zeros(numFeatures, numSegments , size(noisySTFT,2) - numSegments + 1);
for index = 1:size(noisySTFT,2) - numSegments + 1
    stftSegments(:,:,index) = (noisySTFT(:,index:index + numSegments - 1)); 
end
% Set targets and predictors
targets = cleanSTFT;
size(targets)
predictors = stftSegments;
size(predictors)

%% Extract Features Using Tall Arrays
% Advantage: Faster processing

reset(adsTrain)
T = tall(adsTrain)


