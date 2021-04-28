Denoised Audio Folder:
The CL, FC, FIR, and IIR prefixed files for the harvard sentences are denoised audio clips.

-> CL stands for the Convolutional Layered Deep Learning Network that was used to detect
    features and denoise the audio.
-> FC stands for the Fully Connected Deep Learning Network that was used to detect
    features and denoise the audio.
-> FIR is the finite impulse response filter.
-> IIR is the infinite impulse response filter using the hamming window.

The sample rate was 8000, and the noise file used was a random sample from the 
the matlab-included "WashingMachine-16-8-mono-200secs.mp3"

The denoised audio files using a random sample from the "relaxing_piano.mp3" as the noise
were not uploaded. They can be easily obtained by running the Denoising_deeplearning.m 
script file as instructed in the original readme, and then save the CL and FC denoised files
using the following commands in the terminal:

audiowrite("desired_output_filename1.wav",denoisedAudioFullyConnected,fs)
audiowrite("desired_output_filename2.wav",denoisedAudioFullyConvolutional,fs)