## 評価方法
### eval
```
<train時のコマンド> --mode eval --KMMD --FID --resume ./experiments/<モデルのパス>/checkpoint_iter30000.pth.tar
```

### FID
https://github.com/mseitzer/pytorch-fid
```
python -m pytorch_fid /home/yanai-lab/takeda-m/space/dataset/FFHQ/all/ ./outputs/ファイル名
```

### face_detection
https://symfoware.blog.fc2.com/blog-entry-2413.html
```
python face_detection.py <outputsのフォルダ名>
```

## コマンドリスト
スプレッドシート<br>
https://docs.google.com/spreadsheets/d/1lDxAl5atrzrP7_QL9skjtxNZSQX7pI1gnf6ywrhKNco/edit#gid=0                                                           

## モデルの説明
| モデル名                | 説明                      |
| ------------------- | ----------------------- |
| biggan128-ada       | オリジナルのやつ（FiLM）          |
| biggan128-conv1x1   | 1x1convのパラメータをそのまま学習    |
| biggan128-conv1x1-2 | 1x1convのパラメータをzベクトルから生成 |
| biggan128-conv1x1-3 | groupsの方法変更，zベクトルのサイズ変更 |
|                     |                         |


