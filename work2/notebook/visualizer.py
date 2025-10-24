# %%
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
from IPython.display import Audio

# %% [markdown]
"""
# スペクトログラムを用いた音楽ジャンル分類

参考
[Music classification and generation with spectrograms](https://deeplearning.neuromatch.io/projects/ComputerVision/spectrogram_analysis.html)

[GTZAN Genre Classification Preprocessing](https://www.kaggle.com/code/eonuonga/gtzan-genre-classification-preprocessing-1-2)

[GTZAN](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) データセットを用いて，音楽ジャンルの分類を行います．
GTZANデータセットにはオーディオファイルとメルスペクトログラムの画像が含まれています (元のデータセットは一部データが破損しているため実際に使用しているのは[こちら](https://www.kaggle.com/datasets/murataktan/gtzan-fixed))．
各種特徴量を用いて CNN モデルを学習し，ジャンル分類を行います．
"""

# %% [markdown]
"""
## セットアップ
`nvidia-smi` で GPU が使用できることを確認
データセットのシンボリックリンクを作成
"""

# %%
data_path = Path("GTZAN")
if not data_path.exists():
    data_path.symlink_to(Path("/work2/itsuki/shitashimu/GTZAN"))

# %% [markdown]
"""
## 各種表示

### オーディオの読み込み
"""

# %%
example_genre, example_file = "jazz", "jazz.00054.wav"
example_path = data_path / "genres_original" / example_genre / example_file
example_audio, sample_rate = librosa.load(str(example_path))
Audio(example_audio, rate=sample_rate)

# %% [markdown]
"""
### MFCC: Mel Frequency Cepstral Coefficients
資料参照
"""

# %%
example_mfcc = librosa.feature.mfcc(y=example_audio, n_mfcc=13)[1:]
fig, ax = plt.subplots(figsize=(7, 3))
img = librosa.display.specshow(example_mfcc, ax=ax, x_axis="s")
ax.set(title=f"MFCCs – {example_file}", ylabel="MFCC")
plt.tight_layout()
plt.show()

# %% [markdown]
"""
### クロマグラム
資料参照
"""

# %%
example_chroma = librosa.feature.chroma_stft(y=example_audio)
fig, ax = plt.subplots(figsize=(7, 3))
librosa.display.specshow(example_chroma, ax=ax, x_axis="s", y_axis="chroma")
ax.set(title=f"Chromagram – {example_file}")
plt.tight_layout()
plt.show()

# %% [markdown]
r"""
# 動的特徴 - デルタ ($\Delta$) およびダブルデルタ ($\Delta\Delta$) 特徴

MFCC やクロマ特徴のようなオーディオ表現の1次および2次微分の近似 ([参考](https://www.kaggle.com/code/eonuonga/gtzan-genre-classification-preprocessing-1-2#ref-3))．
フレーム間の特徴値の変化がいかに動的であるかを捉えるために使用されます．

$\Delta$ および $\Delta\Delta$ 特徴は，元の特徴とかなり高い相関を持ちますが，より良い収束と性能をもたらす場合があります．

MFCC とクロマグラムの両方に対して $\Delta$ および $\Delta\Delta$ の特徴を生成し，各オーディオサンプルに使用できる合計6種類の特徴が得られます．
以下に，MFCC に対する$\Delta$および$\Delta\Delta$特徴の例を示します．
"""
# %%
example_mfcc_delta = librosa.feature.delta(example_mfcc)
example_mfcc_delta_delta = librosa.feature.delta(example_mfcc, order=2)

fig, axs = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

# Plot MFCC ∆
librosa.display.specshow(example_mfcc_delta, ax=axs[0], x_axis="s")
axs[0].set(title=f"∆ (MFCCs) – {example_file}", xlabel=None, ylabel="MFCC")

# Plot MFCC ∆∆
librosa.display.specshow(example_mfcc_delta_delta, ax=axs[1], x_axis="s")
axs[1].set(title=f"∆∆ (MFCCs) – {example_file}", ylabel="MFCC")

plt.tight_layout()
plt.show()
