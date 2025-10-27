# キャラクター行動検出AI

物体検出AIを使用し、ゲームにおけるプレイヤーの行動を把握するシステムです。<br>
これにより、プレイヤーの行動ログを出力し、上級者等が取る行動にはどのようなものが多いかを<br>
解析することができます。<br>
このプロジェクトでは、プレイヤーの行動の検出までを行っています。<br>

---

## プロジェクト概要
Python 3.11.9 の仮想環境でYOLOv11使用し、<br>
学習 → 検証 → 推論（動画）までを一通り実行する手順をまとめたものです。<br>
（Windows11を想定しています。）

GPU を使う場合は CUDA 12.2 ドライバ環境でも PyTorch cu121（12.1）ビルドが一般的に動作します。<br>
GPU が無い/使わない場合は CPU 版でもそのまま動作します。<br>

---

## 1. 事前準備
### 1-1. Python 3.11.9 の導入
既にインストール済みならスキップしてください。<br>
まだなら下記にアクセスし、公式インストーラで導入し、**「Add Python to PATH」**にチェックしてインストールしてください。<br>
🔗 https://www.python.org/downloads/windows/

### 1-2. NVIDIA GPU を使う場合
nvidia-smi が動作し、GPU が認識されていることを確認してください。
```bash
nvidia-smi
```
ドライバ更新は各自の環境に合わせて実施してください。

---

## 2. 仮想環境の作成と有効化（Python 3.11.9）
上から順番にコマンドを実行して仮想環境の作成、実行してください。
```bash
# 仮想環境を作成（フォルダ名: DetectGame）
py -3.11 -m venv DetectGame

# 有効化
.\DetectGame\Scripts\activate

# PowerShell 実行ポリシーが厳しい環境で、activate時アクセスエラーがあれば、下記を実行してください。
# （ターミナル毎の一時設定なので安全）
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# バージョン確認（3.11.9 系であればOK）
python -V
```

---

## 3.PyTorch のインストール
### 3-A. GPU を使う場合（CUDA 12.2 ドライバ環境想定）
cu121（12.1）ビルドを入れます。（12.2 ドライバでも一般に可動）
```bash
pip install -U pip
pip uninstall -y torch torchvision torchaudio
pip cache purge
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 動作確認（True/1 以上ならOK）
python - << PY
import torch
print("cuda_available:", torch.cuda.is_available())
print("cuda_in_torch:", torch.version.cuda)
print("device_count:", torch.cuda.device_count())
PY
```

### 3-B CPU で使う場合（または GPU が無い場合）
```bash
pip install -U pip
pip uninstall -y torch torchvision torchaudio
pip cache purge
pip install torch torchvision torchaudio  # (+cpu) が入ります

# 確認（cuda_available は False で正常）
python - << PY
import torch
print("cuda_available:", torch.cuda.is_available())
PY
```

## 4. Ultralytics（YOLOv11）のインストール
```bash
pip install ultralytics
yolo --version       # 例: Ultralytics 8.3.221
```
以降のコマンドは、GPUなら device=0、CPUなら device=cpu を付けてください。

---

### 4. データセットの準備（YOLO 形式）
### 4-1 作成した仮想環境直下に、以下の構成でフォルダを作成してください。
```bash
DetectGame/
└─ data/
   ├─ train　# 学習画像
   └─ val　　# 推論画像
```
### 4-2 trainとvalにアノテーションした学習データを格納します。
アノテーションはlabelingというツールを使用しました。使い方は下記URL等をご参照ください。
https://note.com/npaka/n/nf74e32b47712

### 4-3 data.yamlのラベル名を編集
作成した学習データのclasses.txtのラベル名とdata.yamlのラベルを揃えます。<br>
```bash
# namesを編集します。
# 例:classes.txt     data.yaml
#                     names:
#      dog              0: dog
#      person           1: person
#      cat              2: cat
#        .....              .....

path: datasets/mydata
train: images/train
val: images/val
names:
  0: dog
  1: person
  2: cat
      .....
```

## 5. 学習・検証・推論
### 5-1. 学習
train.batを実行してください。もしくは下記コマンドを実行してください。<br>
精度やお好みに合わせてエポック数、バッチサイズを変更してください。
```bash
# 例: 軽量モデル yolo11s、解像度640、50epoch
# GPU:
yolo detect train model=yolo11s.pt data=data.yaml imgsz=640 epochs=50 batch=16 device=0
# CPU:
yolo detect train model=yolo11s.pt data=data.yaml imgsz=640 epochs=50 batch=16 device=cpu
```

###　5-2. 検証（mAP）
下記コマンドを実行してください。
```bash
# GPU:
yolo detect val model=runs/detect/train/weights/best.pt data=data.yaml device=0
# CPU:
yolo detect val model=runs/detect/train/weights/best.pt data=data.yaml device=cpu
```

### 5-3. 動画で推論（可視化ファイル保存）
下記ファイルを実行してください。
適宜、重みファイル、動画ファイル名、confを変更してください。
```bash
python predict.py
```

実行前に、描画するラベル名を変更することができます。<br>
文字化けした場合、predict.pyの下記を編集してください。
```bash
# RENAME_TABLE = {"classes.txt記載のラベル名": "描画するラベル名"}

# 例
RENAME_TABLE = {
    "メインウェポン": "Main_Weapon",
    "潜伏": "hide",
    "人移動": "move",
    "スペシャルウェポン": "special_weapon",
    "デス": "death",
    "マップ": "map",
}
```
