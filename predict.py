import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from model.model import predict_emotion


# def power_spectrum(signal, fft_points=512):
#     spec, sr = librosa.core.spectrum._spectrogram(
#         y=signal, n_fft=fft_points, power=2)
#     return spec


# def frequency_to_mel(f):
#     return 2595.0 * np.log10(1 + f / 700.0)


# def mel_to_frequency(mel):
#     return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


# def triangle(x, left, middle, right):
#     out = np.zeros(x.shape)
#     out[x <= left] = 0
#     out[x >= right] = 0
#     first_half = np.logical_and(left < x, x <= middle)
#     out[first_half] = (x[first_half] - left) / (middle - left)
#     second_half = np.logical_and(middle <= x, x < right)
#     out[second_half] = (right - x[second_half]) / (right - middle)
#     return out


# def filterbanks(
#         num_filter,
#         n_fft,
#         sampling_freq,
#         low_freq=None,
#         high_freq=None):

#     high_freq = high_freq or sampling_freq / 2
#     low_freq = low_freq or 300
#     mels = np.linspace(
#         frequency_to_mel(low_freq),
#         frequency_to_mel(high_freq),
#         num_filter + 2)
#     hertz = mel_to_frequency(mels)
#     freq_index = (
#         np.floor(
#             (n_fft +
#              1) *
#             hertz /
#             sampling_freq)).astype(int)
#     filterbank = np.zeros([num_filter, (int(n_fft/2 + 1))])
#     for i in range(0, num_filter):
#         left = int(freq_index[i])
#         middle = int(freq_index[i + 1])
#         right = int(freq_index[i + 2])
#         z = np.linspace(left, right, num=right - left + 1)
#         filterbank[i,
#                    left:right + 1] = triangle(z,
#                                               left=left,
#                                               middle=middle,
#                                               right=right)

#     return filterbank


# def mfcc(
#         file,
#         sampling_rate,
#         fft_length,
#         num_cepstral=16,
#         num_filters=128,
#         low_frequency=0,
#         high_frequency=None):

#     signal, sampling_frequency = librosa.load(file, sr=sampling_rate)

#     # STFT
#     fr, t, spec = stft(signal, fs=sampling_rate,
#                        nperseg=1024, axis=0, nfft=fft_length)

#     # SQUARING
#     spec = np.square(spec)

#     # CREATING FILTER BANKS
#     fb = filterbanks(128, fft_length, sampling_rate)
#     features = np.dot(fb, spec)

#     # LOG_SPECTRUM
#     log_power_spec = librosa.spectrum.power_to_db(abs(features))

#     # DCT
#     mfccs = dct(log_power_spec, axis=0, norm='ortho')
#     mfcc_final = np.mean(mfccs, axis=1)

#     return mfcc_final[:num_cepstral]


# def raw_prediction(signal_stft, sampling_rate, fft_length, num_cepstral=16, num_filters=128):

#     spec = np.square(signal_stft)

#     # CREATING FILTER BANKS
#     fb = filterbanks(128, fft_length, sampling_rate)
#     features = np.dot(fb, spec)

#     # LOG_SPECTRUM
#     log_power_spec = librosa.spectrum.power_to_db(abs(features))

#     # DCT
#     mfccs = dct(log_power_spec, axis=0, norm='ortho')
#     mfcc_final = np.mean(mfccs, axis=1)

#     return mfcc_final[:num_cepstral]


# def f_high(y, sr):
#     b, a = signal.butter(10, 2000/(sr/2), btype='highpass')
#     yf = signal.lfilter(b, a, y)
#     return yf


# def pre_process(file, NFFT=16384):

#     signal, sr = librosa.load(file, sr=NFFT)
#     s = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=64)
#     d = librosa.power_to_db(s, ref=np.max)

#     #windowing and framing
#     pre_emphasis = 0.97
#     emphasized_signal = np.append(
#         signal[0], signal[1:] - pre_emphasis * signal[:-1])
#     frame_size = 0.025
#     frame_stride = 0.01
#     frame_length, frame_step = frame_size * sr, frame_stride * sr
#     signal_length = len(emphasized_signal)
#     frame_length = int(round(frame_length))
#     frame_step = int(round(frame_step))
#     num_frames = int(
#         np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))

#     pad_signal_length = num_frames * frame_step + frame_length
#     z = np.zeros((pad_signal_length - signal_length))
#     pad_signal = np.append(emphasized_signal, z)

#     indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(
#         np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
#     frames = pad_signal[indices.astype(np.int32, copy=False)]
#     # Apply hamming window
#     frames *= np.hanning(frame_length)

#     mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
#     pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))

#     yf = f_high(mag_frames, NFFT)

#     return yf.T, sr


def predict_emotion(aud):

    with open("/home/sidharth/Desktop/S8project/code/site/Emotion_Voice_Detection_Model_dataset.pkl", 'rb') as file:
        Emotion_Voice_Detection_Model = pickle.load(file)

    with open("/home/sidharth/Desktop/S8project/code/site/x_values_dataset.pkl", 'rb') as file:
        x_values = pickle.load(file)

    scaler = MinMaxScaler()
    scaler.fit(x_values)
    features = predict_emotion(aud, 16384, 16384, num_cepstral=19)
    feature = [features]
    feature = np.array(feature)
    print(feature)
    feature = scaler.transform(feature)
    print(feature)

    emotion = Emotion_Voice_Detection_Model.predict(feature)

    return emotion[0]
