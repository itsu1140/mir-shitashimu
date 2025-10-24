# %% [markdown]
"""
# 音声の再合成

音声ファイルからオンセット，ピッチ推定を行う．
得られたピッチから正弦波合成を行う．
"""

# %%
import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

# %% [markdown]
"""
## 波形データの読み込みと表示
"""

# %%
x, sr = librosa.load("./data/piano_sample.wav")
print(x.shape[0], sr)  # data size, sampling rate

librosa.display.waveshow(x)
plt.show()

# %% [markdown]
"""
## 時間周波数解析
"""

# %%
X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
librosa.display.specshow(Xdb, sr=sr, x_axis="time", y_axis="hz")
plt.show()

# %% [markdown]
"""
## オンセットエンベロープを求める
エンベロープ: 音が鳴り始めてから消えるまでの音量，音色の時間的変化を表す
"""

# %%
hop_length = 100
onset_env = librosa.onset.onset_strength(y=x, sr=sr, hop_length=hop_length)
plt.plot(onset_env)
plt.xlim(0, len(onset_env))
plt.show()

# %% [markdown]
"""
## オンセット時刻を求める
"""

# %%
onset_samples = librosa.onset.onset_detect(
    y=x,
    sr=sr,
    units="samples",
    backtrack=False,
    pre_max=20,
    post_max=20,
    pre_avg=100,
    post_avg=100,
    delta=0.2,
    wait=0,
)
print(onset_samples)

# %%
onset_boundaries = np.concatenate([[0], onset_samples, [len(x)]])
onset_times = librosa.samples_to_time(onset_boundaries, sr=sr)
print(onset_times)

# %% [markdown]
"""
## 音声波形にオンセット位置を表示
"""

# %%
librosa.display.waveshow(x)
plt.vlines(onset_times, -1, 1, color="r")
plt.show()

# %% [markdown]
"""
## ピッチ推定と合成
"""


# %%
def estimate_pitch(segment, sr, fmin=50.0, fmax=1000.0):
    r = librosa.autocorrelate(segment)
    i_min = sr / fmax
    i_max = sr / fmin
    r[: int(i_min)] = 0
    r[int(i_max) :] = 0
    i = r.argmax()
    f0 = float(sr) / i
    return f0


def generate_sine(f0, sr, n_duration):
    n = np.arange(n_duration)
    return 0.2 * np.sin(2 * np.pi * f0 * n / float(sr))


def estimate_pitch_and_generate_sine(x, onset_samples, i, sr):
    n0 = onset_samples[i]
    n1 = onset_samples[i + 1]
    f0 = estimate_pitch(x[n0:n1], sr)
    return generate_sine(f0, sr, n1 - n0)


y = np.concatenate(
    [
        estimate_pitch_and_generate_sine(x, onset_boundaries, i, sr)
        for i in range(len(onset_boundaries) - 1)
    ],
)
sf.write("piano_sample_gen.wav", y, sr, "PCM_24", endian="LITTLE")
Y = librosa.stft(y)
Ydb = librosa.amplitude_to_db(abs(Y))
librosa.display.specshow(Ydb, sr=sr, x_axis="time", y_axis="hz", cmap="coolwarm")
plt.show()
