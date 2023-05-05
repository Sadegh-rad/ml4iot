import sounddevice as sd
from scipy.io.wavfile import write
from time import time
import os
import argparse as ap
import tensorflow as tf
import tensorflow_io as tfio

def callback(indata,frames,callback_time,status):
    """
    indata data
    frames number of sample inside indata
    callback-time,status """
    global best_param
    if not is_silence(indata,best_param['downsampling_rate'], best_param['frame_length_in_s'], best_param['dbFSthres'], best_param['duration_thres']):
        timestamp = int(time() * 1000) 
        write(f'{timestamp}.wav',rate=16000,data=indata)
        size_in_bytes = os.path.getsize(f'./{timestamp}.wav')
        size_in_kb = size_in_bytes / 1024.0
        print(f'Sound found Size {size_in_kb} KB')

    
def is_silence(indata, downsampling_rate, frame_length_in_s, dbFSthres, duration_thres):
    spectrogram, sampling_rate = get_spectrogram(
        indata,
        downsampling_rate,
        frame_length_in_s,
        frame_length_in_s
    )
    dbFS = 20 * tf.math.log(spectrogram + 1.e-6)
    energy = tf.math.reduce_mean(dbFS, axis=1)
    non_silence = energy > dbFSthres
    non_silence_frames = tf.math.reduce_sum(tf.cast(non_silence, tf.float32))
    non_silence_duration = (non_silence_frames + 1) * frame_length_in_s

    if non_silence_duration > duration_thres:
        return 0
    else:
        return 1 

def get_audio_from_numpy(indata):
    indata = tf.convert_to_tensor(indata, dtype=tf.float32)
    indata = 2 * ((indata + 32768) / (32767 + 32768)) - 1  # CORRECT normalization between -1 and 1
    indata = tf.squeeze(indata)
    return indata, tf.convert_to_tensor(16000, dtype=tf.float32)

def get_spectrogram(indata, downsampling_rate, frame_length_in_s, frame_step_in_s):
    audio_padded, sampling_rate = get_audio_from_numpy(indata)
    if downsampling_rate != sampling_rate:
        sampling_rate_int64 = tf.cast(sampling_rate, tf.int64)
        audio_padded = tfio.audio.resample(audio_padded, sampling_rate_int64, downsampling_rate)
    sampling_rate_float32 = tf.cast(downsampling_rate, tf.float32)
    frame_length = int(frame_length_in_s * sampling_rate_float32)
    frame_step = int(frame_step_in_s * sampling_rate_float32)
    spectrogram = stft = tf.signal.stft(
        audio_padded, 
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=frame_length
    )
    spectrogram = tf.abs(stft)
    return spectrogram, sampling_rate

# device as argument, need to be converted to int
parser = ap.ArgumentParser()
parser.add_argument('--device', type=int, default=0)
args = parser.parse_args()

best_param={
    'downsampling_rate': [16000],
    'frame_length_in_s': [0.0016],
    'dbFSthres': [-120], # param per accu
    'duration_thres': [0.025] # param per accu
    }

store_audio = True

with sd.InputStream(device=args.device,
               channels=1,
               samplerate=16000,
               dtype="int16",
               callback=callback,
               blocksize=16000):
    #with open a stream of data, here we put some code that is executed until stream is open

    while True:
        key = input()
        if key in ['q', 'Q']:
            break
        if key in ['p', 'P']:
            store_audio = not(store_audio)
