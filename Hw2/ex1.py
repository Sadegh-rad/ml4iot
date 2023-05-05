import numpy as np
from time import time
import uuid
import redis
import psutil
import sounddevice as sd
import os
import tensorflow as tf
import argparse as ap

# Input Parameters
parser = ap.ArgumentParser()
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--host', type=str, default='redis-19466.c293.eu-central-1-1.ec2.cloud.redislabs.com')
parser.add_argument('--port', type=int, default=19466)
parser.add_argument('--user', type=str, default='default')
parser.add_argument('--password', type=str, default='iIQjppRdJ7f3kKU4Ng5IDkgv6TvaPnrN')
args = parser.parse_args()

REDIS_HOST = args.host
REDIS_PORT = args.port
REDIS_USERNAME = args.user
REDIS_PASSWORD = args.password

# Possible words that can be recognized
LABELS = ['go', 'stop']
pred_label = ''

# Model parameters
PREPROCESSING_ARGS = {
    'downsampling_rate': 16000,
    'frame_length_in_s': 0.032,
    'frame_step_in_s': 0.016,
    'num_mel_bins': 10,
    'lower_frequency': 20,
    'upper_frequency': 8000,
    'num_coefficients': 10
}

IS_SILENT_ARGS = {
    'downsampling_rate': [16000],
    'frame_length_in_s': [0.016],
    'dbFSthres': [-120],
    'duration_thres': [0.025]
}

downsampling_rate = PREPROCESSING_ARGS['downsampling_rate']
sampling_rate_int64 = tf.cast(downsampling_rate, tf.int64)
frame_length = int(downsampling_rate * PREPROCESSING_ARGS['frame_length_in_s'])
frame_step = int(downsampling_rate * PREPROCESSING_ARGS['frame_step_in_s'])
spectrogram_width = (16000 - frame_length) // frame_step + 1
num_spectrogram_bins = frame_length // 2 + 1

linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
    num_mel_bins=PREPROCESSING_ARGS['num_mel_bins'],
    num_spectrogram_bins=num_spectrogram_bins,
    sample_rate=PREPROCESSING_ARGS['downsampling_rate'],
    lower_edge_hertz=PREPROCESSING_ARGS['lower_frequency'],
    upper_edge_hertz=PREPROCESSING_ARGS['upper_frequency']
)

interpreter = tf.lite.Interpreter(model_path='model18.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Check if redis is connected (1st time)
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, username=REDIS_USERNAME, password=REDIS_PASSWORD)
is_connected = redis_client.ping()
print('Redis Connected:', is_connected)

mac_address = hex(uuid.getnode())
mac_battery = mac_address + ':battery'
mac_power = mac_address + ':power'

try:
    redis_client.ts().create(mac_battery)
    redis_client.ts().create(mac_power)
except redis.ResponseError:
    pass


def callback(indata, frames, callback_time, status):
    timestamp = time()  # Save time as soon as it enter the callback
    global IS_SILENT_ARGS
    global pred_label

    if not is_silence(indata=indata,
                      downsampling_rate=IS_SILENT_ARGS['downsampling_rate'],
                      frame_length_in_s=IS_SILENT_ARGS['frame_length_in_s'],
                      dbFSthres=IS_SILENT_ARGS['dbFSthres'],
                      duration_thres=IS_SILENT_ARGS['duration_thres']):
        # If not silent call MFCCS and extend dimension to do the inference
        mfccs = get_mfccs(filename=indata,
                          downsampling_rate=PREPROCESSING_ARGS['downsampling_rate'],
                          frame_length_in_s=PREPROCESSING_ARGS['frame_length_in_s'],
                          frame_step_in_s=PREPROCESSING_ARGS['frame_step_in_s'],
                          num_coefficients=PREPROCESSING_ARGS['num_coefficients'])
        mfccs = tf.expand_dims(mfccs, -1)
        mfccs = tf.expand_dims(mfccs, 0)
        # Prediction
        interpreter.set_tensor(input_details[0]['index'], mfccs)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        mag_index = np.max(output[0])  # Probability in output
        if mag_index > 0.95:
            top_index = np.argmax(output[0])
            pred_label = LABELS[top_index]
            print(f'The predicted label is {pred_label}', output[0])  # Keeps memory of last valid word

    if pred_label == 'go':
        battery_level = psutil.sensors_battery().percent
        power_plugged = int(psutil.sensors_battery().power_plugged)
        timestamp_ms = int(timestamp * 1000)  # Redis TS requires the timestamp in ms
        try:
            redis_client.ts().add(key=mac_battery, timestamp=timestamp_ms, value=battery_level)
            redis_client.ts().add(mac_power, timestamp_ms, power_plugged)
        except:
            print('Error on adding value to redis TS.')
        print(f'Power-plugged status is {power_plugged}')
        print(f'Battery level status is {battery_level}')


# Checked
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


# Checked
def get_audio_from_numpy(indata):
    indata = tf.convert_to_tensor(indata, dtype=tf.float32)
    indata = 2 * ((indata + 32768) / (32767 + 32768)) - 1  # CORRECT normalization between -1 and 1
    indata = tf.squeeze(indata)
    return indata, tf.convert_to_tensor(16000, dtype=tf.float32)


# Checked
def get_spectrogram(indata, downsampling_rate, frame_length_in_s, frame_step_in_s):
    # Down sampling is not used, should be renamed to sample_rate
    audio_padded, sampling_rate = get_audio_from_numpy(indata)
    sampling_rate_float32 = sampling_rate
    frame_length = int(frame_length_in_s * sampling_rate_float32)
    frame_step = int(frame_step_in_s * sampling_rate_float32)
    stft = tf.signal.stft(
        audio_padded,
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=frame_length
    )
    spectrogram = tf.abs(stft)
    return spectrogram, sampling_rate


# Checked
def get_log_mel_spectrogram(filename, downsampling_rate, frame_length_in_s, frame_step_in_s):
    # linear_to_mel_weight_matrix is global. Spectrogram can be given as param to avoid double calculation
    global linear_to_mel_weight_matrix
    spectrogram, sampling_rate = get_spectrogram(filename, downsampling_rate, frame_length_in_s, frame_step_in_s)
    mel_spectrogram = tf.matmul(spectrogram, linear_to_mel_weight_matrix)
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
    return log_mel_spectrogram


# Checked
def get_mfccs(filename, downsampling_rate, frame_length_in_s, frame_step_in_s, num_coefficients):
    log_mel_spectrogram = get_log_mel_spectrogram(filename,
                                                  downsampling_rate,
                                                  frame_length_in_s,
                                                  frame_step_in_s)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
    mfccs = mfccs[..., :num_coefficients]
    return mfccs


with sd.InputStream(device=args.device,
                    channels=1,
                    samplerate=16000,
                    dtype="int16",
                    callback=callback,
                    blocksize=16000):
    while True:
        key = input()
        if key in ['q', 'Q']:
            break
