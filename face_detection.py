import cv2
import numpy as np
import glob
import sys

# 設定ファイル
prototxt = 'deploy.prototxt'
model = 'res10_300x300_ssd_iter_140000_fp16.caffemodel'
# 検出した部位の信頼度の下限値
confidence_limit = 0.5
# 設定ファイルからモデルを読み込み
net = cv2.dnn.readNetFromCaffe(prototxt, model)
files_path = glob.glob('./outputs/' + sys.argv[1] + '/*.jpg')

confidence_ave = 0
for file_path in files_path:
    # 解析対象の画像を読み込み
    image = cv2.imread(file_path)
    # 300x300に画像をリサイズ、画素値を調整
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))
    # 顔検出の実行
    net.setInput(blob)
    detections = net.forward()
    # 検出部位をチェック
    for i in range(0, detections.shape[2]):
        # 信頼度
        confidence = detections[0, 0, i, 2]
        # 信頼度が下限値以下なら無視
        if confidence < confidence_limit:
            continue
        confidence_ave += confidence

confidence_ave = confidence_ave / len(files_path)
print(confidence_ave)