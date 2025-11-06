"""
音声の再合成

音声ファイルからオンセット，ピッチ推定を行う．
得られたピッチから正弦波合成を行う．
"""

from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


def load_wav(file_path: Path) -> tuple[np.ndarray, int]:
    """波形データの読み込みと表示"""
    wav, sr = librosa.load(file_path)
    librosa.display.waveshow(wav)
    plt.savefig("waveform.png")
    return wav, sr


def time_frequency(wav: np.ndarray, sr: int) -> None:
    """時間周波数解析"""
    stft = librosa.stft(wav)
    stft_db = librosa.amplitude_to_db(abs(stft))
    librosa.display.specshow(stft_db, sr=sr, x_axis="time", y_axis="hz")
    plt.savefig("time_frequency.png")


def onset_envelope(wav: np.ndarray, sr: int, hop_length: int = 100) -> np.ndarray:
    """
    オンセットエンベロープを求める

    エンベロープ: 音が鳴り始めてから消えるまでの音量，音色の時間的変化を表す
    """
    onset_env = librosa.onset.onset_strength(y=wav, sr=sr, hop_length=hop_length)
    plt.plot(onset_env)
    plt.xlim(0, len(onset_env))
    plt.savefig("onset_envelope.png")
    return onset_env


def get_onset_times(wav: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """オンセット時刻を求める"""
    onset_samples = librosa.onset.onset_detect(
        y=wav,
        sr=sr,
        units="samples",
        pre_max=20,
        post_max=20,
        pre_avg=100,
        post_avg=100,
        delta=0.2,
        wait=0,
    )
    onset_boundaries = np.concatinate([[0], onset_samples, [len(wav)]])
    # 先頭と末尾のサンプル位置を追加
    onset_times = librosa.samples_to_time(onset_boundaries, sr=sr)
    return onset_boundaries, onset_times


def show_onset_times(wav: np.ndarray, onset_times: np.ndarary) -> None:
    """オンセット時刻の表示"""
    librosa.display.waveshow(wav)
    plt.vlines(onset_times, -1, 1, colors="r")
    plt.savefig("onset_times.png")


def estimate_pitch(segment, sr: int, fmin: float = 50.0, fmax: float = 1000.0) -> float:
    """ピッチ推定"""
    r = librosa.autocorrelate(segment)
    i_min = sr / fmax
    i_max = sr / fmin
    r[: int(i_min)] = 0
    r[int(i_max) :] = 0
    i = r.argmax()
    f0 = float(sr) / i
    return f0


def generate_sine_wave(f0: float, sr: int, n_duration: int) -> np.ndarray:
    n = np.arange(n_duration)
    return 0.2 * np.sin(2 * np.pi * f0 * n / sr)


def estimate_and_synthesize(
    wav: np.ndarray,
    onset_samples: np.ndarray,
    i: int,
    sr: int,
) -> np.ndarray:
    n0 = onset_samples[i]
    n1 = onset_samples[i + 1]
    f0 = estimate_pitch(wav[n0:n1], sr)
    return generate_sine_wave(f0, sr, n1 - n0)


def resynthesize(wav: np.ndarray, onset_boundaries: np.ndarray, sr: int) -> None:
    y = np.concatenate(
        [
            estimate_and_synthesize(wav, onset_boundaries, i, sr)
            for i in range(len(onset_boundaries) - 1)
        ],
    )
    sf.write("resynthesized.wav", y, sr, "PCM_24", endian="LITTLE")
    stft = librosa.stft(y)
    stft_db = librosa.amplitude_to_db(abs(stft))
    librosa.display.specshow(stft_db, sr=sr, x_axis="time", y_axis="hz")
    plt.savefig("resynthesized_time_frequency.png")


def main():
    wav_path = Path("data/piano_sample.wav")
    wav, sr = load_wav(wav_path)
    time_frequency(wav, sr)
    onset_envelope(wav, sr)
    onset_boundaries, onset_times = get_onset_times(wav, sr)
    show_onset_times(wav, onset_times)
    resynthesize(wav, onset_boundaries, sr)


if __name__ == "__main__":
    main()
