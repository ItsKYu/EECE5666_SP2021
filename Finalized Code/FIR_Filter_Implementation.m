clear;
clc;

%Design and FIR filter for processing input audio signal 

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

%Plot the orginal signal in the time domain 
subplot(2,1,1)
plot(t,y)
xlabel('Time (s)')
ylabel('Amplitude')
title('Time Domain - Unfiltered Audio')
grid minor;

m = length(y);
x1 = fft(y, m); %Implement the fast fourier transform to view frequency domain 
f = (0:m-1)*(fs/m);
amplitude = abs(x1)/m;

%Specify plot parameters 
amp = amplitude(1:(m/2));
freq = f(1:(m/2));

%Plot original, unfiltered audio signal in frequency domain to extract
%desired features 
subplot(2,1,2)
plot(freq,amp)
title('Frequency Domain - Unfiltered Audio')
xlabel('Frequency (Hz)')
ylabel('Amplitude')
xlim([0 2000])
grid minor;

%Filter parameters defined by visually inspecting the original signal
%These can be changed based on the user's desired spectrum 
Fp = 300/fs;
wp = Fp*pi; 
Fs = 1000/fs;
ws = Fs*pi;
As = 50; 
Ap = 0.5;

%Define the cutoff frequency and frequency difference 
wc = (ws+wp)/2;
dw = ws-wp;

%Calculate the length of the filter based on the hamming window definition 
L = ceil(6.6*pi/dw); 
M = L-1;
%Ensure it is type 1
if mode(M,2) == 1
    M = M+1;
    L = M+1;
end 

%Calculate the windowing function of the hamming window 
w_ham = (hamming(L))'; 

%Use function ideal_lp to find the response of a lowpass FIR filter 
hd = ideal_lp(wc,L); 
w_ham = (hamming(L))'; 
h = hd .* w_ham; %Implement the windowing function to the response 
om = linspace(0,1,1001)*pi;
H = freqz(h,1,om); %Calculate the parameters of the magnitude response 
%establish the magnitude and log magnitude response parameters 
Hmag = abs(H);
Hdb = 20*log10(Hmag./max(Hmag));

n = 0:M;


%Plotting log-magnitude response, ideal impulse response, actual impulse 
%response, and window visualization  
figure()
plot(om/pi,Hdb); grid on;
title('Log-Magnitude Reponse - Hamming')
xlabel('w/\pi');
ylabel('Magnitude (dB)')
figure() 
subplot(2,1,1)
stem(n,hd,'fill')
xlabel('n')
ylabel('h_d[n]')
title('Ideal Impulse Response - Hamming')
subplot(2,1,2)
stem(n,h,'fill')
xlabel('n')
ylabel('h[n]')
title('Actual Impulse Response - Hamming')
figure()
stem(n,w_ham,'fill')
xlabel('n')
ylabel('w[n]')
title('Hamming Window')
ylim([0,1.1])
figure()


%Make more plots for comparison
%Compare time domain unfiltered and filtered audio 
subplot(2,1,1)
plot(t,y); 
xlabel('time')
ylabel('Amplitude')
title('Time Domain - Unfiltered Audio')
grid minor

b = fir1(M,.1*pi/pi);
out = filter(b,1,y)
subplot(2,1,2)
plot(t,out)
ylabel('time (s)')
xlabel('Amplitude')
title('Time Domain - Filtered Audio')
grid minor


figure()

m = length(y);
x1 = fft(y, m);
f = (0:m-1)*(fs/m);
amplitude = abs(x1)/m;

amp = amplitude(1:(m/2));
freq = f(1:(m/2));

%Compare frequency domain unfiltered and filtered audio 
subplot(2,1,1)
plot(freq,amp); grid on;
title('Frequency Domain - Unfiltered Audio')
xlabel('Frequency')
ylabel('Amplitude')
xlim([0 500])

m = length(out);
x2 = fft(out, m);
f = (0:m-1)*(fs/m);
amplitude1 = abs(x2)/m;

amp2 = amplitude1(1:(m/2));
freq2 = f(1:(m/2));

subplot(2,1,2)
plot(freq2,amp2); grid on;
title('Frequency Domain - Filtered Audio')
xlabel('Frequency')
ylabel('Amplitude')
xlim([0 500])


%Uncomment this section to play filtered audio out loud 
% player1 = audioplayer(Y,fs);
% play(player1);

%Write the filtered signal and save it as a new audio file 
filename = 'harvard_62_9_FIR.wav'
audiowrite(filename,out,fs);


