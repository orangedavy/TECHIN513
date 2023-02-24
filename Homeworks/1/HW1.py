import numpy as np
import matplotlib.pyplot as plt
import simpleaudio as sa
import scipy.io.wavfile as wav
import scipy.signal as sig


def play(file):

    wav_obj = sa.WaveObject.from_wave_file(path + file + '.wav')
    play_obj = wav_obj.play()
    play_obj.wait_done()


def timescale(x, fs, a):
    # x: input signal vector
    # fs: sampling rate (in Hz)
    # a: scaling parameter. This has to be a decimal value for as_integer_ratio to work. 
    # So, explicitly casting it into a float or a double or any fractional data type will help.
    # returns t: time samples vector corresponding to y: scaled signal

    # n, d = decimal.Decimal(a).as_integer_ratio()
    [n, d] = (np.double(a)).as_integer_ratio()
    y = sig.resample_poly(x.astype('float'), d, n)
    t = np.arange(0, len(y), 1) * (1.0 / fs)
    return y, t


def timeshift(x, fs, t0):
    # x: input signal vector
    # fs: sampling rate (in Hz)
    # t0: shifting parameter. Positive value indicates a delay, and negative an advance. 
    # n0: integer shift, corresponding to the index of samples given t0 and fs.
    # returns t: time samples vector corresponding to y: shifted signal

    n0 = int(abs(t0) * fs)
    if t0 > 0:
        y = np.concatenate((np.zeros(n0), x))
    elif t0 < 0:
        if n0 <= len(x):
            y = np.concatenate((x[n0 - 1], np.zeros(n0)))
        else:
            y = np.ones(n0)
    else:
        y = x
    t = np.arange(0, len(y), 1) * (1.0 / fs)
    return y, t


path = 'Homeworks/1/sounds/'

# fs1, x1 = wav.read(path + 'train32.wav')
# len1 = len(x1)
# try:
#     ch1 = x1.shape[1]
# except IndexError:
#     ch1 = 1

# print(
#     f"The train.wav file has {len1} samples with a sampling rate of {fs1}.\n"
#     f"It has {ch1} channel{'s' if ch1 != 1 else ''} and the type {x1.dtype}."
# )

# fs2 = int(fs1 / 2)
# fs3 = int(1.5 * fs1)

# wav.write(path + 'train16.wav', fs2, x1.astype('int16'))
# wav.write(path + 'train48.wav', fs3, x1.astype('int16'))

# play('train32')
# play('train16')
# play('train48')

# n0 = int(fs1 * 0.5)
# s1 = np.concatenate((np.ones(n0), 0.2 * np.ones(len1 - n0)))
# v1 = np.multiply(x1, s1)

# wav.write(path + 'train_dampen.wav', fs1, v1.astype('int16'))

# r1 = np.arange(1, 0, -1 / len1)
# # t1 = np.arange(len1)
# # r1 = 1 - 1 / t1  # how to make calculations?
# v2 = np.multiply(x1, r1)

# wav.write(path + 'train_dampen_lnr.wav', fs1, v2.astype('int16'))

# play('train32')
# play('train_dampen')
# play('train_dampen_lnr')

# w, t_w = timescale(x1, fs1, 2)
# v, t_v = timescale(x1, fs1, 0.5)
# x1, t_x1 = timescale(x1, fs1, 1)
# z, t_z = np.ascontiguousarray(list(reversed(x1))), t_x1

# wav.write(path + 'train_0.5x.wav', fs1, w.astype('int16'))
# wav.write(path + 'train_2x.wav', fs1, v.astype('int16'))
# wav.write(path + 'train_reverse.wav', fs1, z.astype('int16'))

# fig, axs = plt.subplots(4, sharex=True, sharey=True)
# plt.xlim(0.0, 4.0)
# plt.ylim(-25000, 25000)
# plt.xlabel('time')
# plt.ylabel('amplitude')

# axs[0].plot(t_x1, x1)
# axs[0].set_title('train')
# axs[1].plot(t_w, w)
# axs[1].set_title('w(t)')
# axs[2].plot(t_v, v)
# axs[2].set_title('v(t)')
# axs[3].plot(t_z, z)
# axs[3].set_title('z(t)')

# plt.tight_layout()
# plt.show()

# play('train32')
# play('train_0.5x')
# play('train_2x')
# play('train_reverse')

# dl, t_dl = timeshift(x1, fs1, 0.5)
# ad, t_ad = timeshift(x1, fs1, -2)
# x1, t_x1 = timeshift(x1, fs1, 0)

# wav.write(path + 'train_+0.5.wav', fs1, dl.astype('int16'))
# wav.write(path + 'train_-2.wav', fs1, ad.astype('int16'))

# fig, axs = plt.subplots(3, sharex=True, sharey=True)
# plt.xlim(0.0, 4.0)
# plt.ylim(-25000, 25000)
# plt.xlabel('time')
# plt.ylabel('amplitude')

# axs[0].plot(t_x1, x1)
# axs[0].set_title('x1(t)')
# axs[1].plot(t_dl, dl)
# axs[1].set_title('x1(t+0.5)')
# axs[2].plot(t_ad, ad)
# axs[2].set_title('x1(t-2)')

# plt.tight_layout()
# plt.show()

# play('train32')
# play('train_+0.5')
# play('train_-2')


fs, cat = wav.read(path + 'cat.wav')
fs, cow = wav.read(path + 'cow.wav')

cat_amp = np.multiply(cat, 10 * np.ones(len(cat)))
cow_fast = timescale(cow, fs, 3)
cat_dl = timeshift(cat_amp, fs, 1)
cow_ad = timeshift(cow_fast, fs, -1)

linear = np.concatenate((cat_amp, cow_fast[0]))
delay = np.concatenate((cat_dl[0], cow))

decay = np.arange(1, 0, -1 / len(delay))
delay_dmp = np.multiply(delay, decay)

synth = np.concatenate((linear, delay_dmp))
wav.write(path + 'synth.wav', fs, synth.astype('int16'))
