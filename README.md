# MIR に親しむ

MIR に親しむの作業資料です．
コードは秘伝のタレなので私が書いたものではありません．
アップデートは Fork なりしてやってください．

[notion ページ](https://www.notion.so/283d5612f793800e98b3fe6542b52dc8?pvs=25)

## 環境の作成
仮想環境には uv を使用しています．
uv のセットアップの後 `uv sync` により環境の作成が始まります．

## 課題1

- `work1/` に移動

### `.abc` ファイルを他の音楽形式に変換

- `build.sh` で必要なライブラリをインストール
- `uv run python3 convert_abc.py` で MIDI, WAV, PDF の作成

> [!NOTE]
> 現在のバージョンでは wav の合成はできない

### 波形データからオンセット，ピッチ推定，正弦波で再合成

- `uv run python3 synth.py` で正弦波の再合成まで


## 課題2

[GTZAN](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) データセットを用いて，音楽ジャンルの分類を行います．
GTZANデータセットにはオーディオファイルとメルスペクトログラムの画像が含まれています (元のデータセットは一部データが破損しているため実際に使用しているのは[こちら](https://www.kaggle.com/datasets/murataktan/gtzan-fixed))．
各種特徴量を用いて CNN モデルを学習し，ジャンル分類を行います．
`notebook/` 以下のファイルを動かしたい場合は拡張機能 [Jupytext](https://marketplace.visualstudio.com/items?itemName=congyiwu.vscode-jupytext) の利用をお勧めします．

### 学習1

波形データのメルスペクトログラムからジャンル分類を行うモデルの学習

- `work2/training-1/` に移動
- `uv run python3 setup.py` でデータセットのリンクとデータ用のディレクトリを用意
- `uv run python3 main.py` により学習開始

### 学習2

波形データの MFCC ，クロマグラムとその $\Delta$ 及び $\Delta\Delta$ 特徴量からジャンル分類を行うモデルの学習

学習には [WandB](https://wandb.ai/) の API キーが必要です．
`training-2/.env` に

```
WANDB_API_KEY=your_api_key
```

を記述し，エディタの `setting.json` に

```
"python.envFile":"${workspaceFolder}/.env"
```

を記述してください．

- `work2/training-2/` に移動
- `uv run python3 setup.py` で特徴量を抽出，結合して保存する処理を行う
- `uv run python3 main.py` により学習開始

### 参考

[Music classification and generation with spectrograms](https://deeplearning.neuromatch.io/projects/ComputerVision/spectrogram_analysis.html)

[GTZAN Genre Classification Preprocessing](https://www.kaggle.com/code/eonuonga/gtzan-genre-classification-preprocessing-1-2)