# ===============================================================================
# YOLO推論用コード
# ===============================================================================
import cv2
from ultralytics import YOLO

# ファイルパス等設定
WEIGHTS   = r"runs/detect/train/weights/best.pt"  # 学習した重みファイル"
VIDEO_IN  = r"test_video.mp4"                     # 入力動画パス
VIDEO_OUT = r"predict.mp4"                        # 出力動画パス
DEVICE    = "0"                                   # cpuなら"cpu"、GPUなら0

# ラベル名の置換テーブル
# 未定義はunknownで描画
RENAME_TABLE = {
    "メインウェポン": "Main_Weapon",
    "潜伏": "hide",
    "人移動": "move",
    "スペシャルウェポン": "special_weapon",
    "デス": "death",
    "マップ": "map",
}

def decorate_name(label_name, conf):
    """
    ラベル名の最終表記を返す。：
      - 日本語を英語へ（文字化け対策）
      - 信頼度の取得
    """
    base = RENAME_TABLE.get(label_name, 'unknown')  # テーブルに無ければ元名
    return f"{base} ({conf*100:.0f}%)"             # 例: 「Main_Weapon (87%)」

def main():
    """
    入力動画をフレーム単位で読み込み、YOLOで推論した結果を描画して動画を書き出す。
    - 動画の入出力はを利用
    - 推論は動画で実行
    """
        
    # 動画の取得
    cap = cv2.VideoCapture(VIDEO_IN)
    if not cap.isOpened():
        raise RuntimeError(f"動画が読み込めませんでした。: {VIDEO_IN}")

    # fps, w, h, 動画作成オブジェクト初期化 
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (w, h))

    # モデル読み込み
    model = YOLO(WEIGHTS)

    # 元クラス名取得
    base_names = model.names if hasattr(model, "names") else {}

    # =========
    # 推論の実行
    # =========
    gen = model.predict(
        source=VIDEO_IN,
        device=DEVICE,
        imgsz=640,
        conf=0.25,
        stream=True,
        save=False,        
        vid_stride=1,      
        verbose=False
    )

    # ===========================
    # 推論結果を1フレームごとに描画
    # ===========================
    frame_idx = 0
    for r in gen:
        # BGRで画像データ受け取り
        frame = r.orig_img.copy()

        # 検出されなければ描画しない
        if r.boxes is None or len(r.boxes) == 0:
            writer.write(frame)
            frame_idx += 1
            continue

        # バウンディングボックス、信頼度、クラス取得
        boxes = r.boxes.xyxy.cpu().numpy()   # (N,4)
        confs = r.boxes.conf.cpu().numpy()   # (N,)
        clses = r.boxes.cls.cpu().numpy()    # (N,)

        # ==========================================
        # 検出ごとにバウンディングボックスとラベルを描画
        # ==========================================
        for (x1, y1, x2, y2), conf, cls_id in zip(boxes, confs, clses):
            cls_id = int(cls_id)
            orig_name = base_names.get(cls_id, str(cls_id))
            label_txt = decorate_name(orig_name, float(conf))

            # 1) バウンディングボックス描画（黄色）
            x1i, y1i, x2i, y2i = map(lambda v: int(max(0, min(v, 10**7))), (x1, y1, x2, y2))
            cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), (0, 255, 255), 2) 

            # 2) ラベル背景、クラス名描画
            font      = cv2.FONT_HERSHEY_SIMPLEX
            font_scale= 0.6
            thickness = 2
            (tw, th), baseline = cv2.getTextSize(label_txt, font, font_scale, thickness)
            # 背景
            cv2.rectangle(frame, (x1i, max(0, y1i - th - 6)), (x1i + tw + 6, y1i), (0, 255, 255), -1)
            # ラベル名（黒文字）
            cv2.putText(frame, label_txt, (x1i + 3, y1i - 4), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

        # フレーム書き込み
        writer.write(frame)
        frame_idx += 1

    # ======================
    # 終了処理（リソース解放）
    # ======================
    cap.release()
    writer.release()
    print(f"検出結果を反映した動画を作成しました。: {VIDEO_OUT}")

if __name__ == "__main__":
    main()
