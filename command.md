## 評価方法
### eval
```
<train時のコマンド> --mode eval --KMMD --FID --resume ./experiments//checkpoint_iter30000.pth.tar
```

### FID
https://github.com/mseitzer/pytorch-fid
face
```
python -m pytorch_fid /home/yanai-lab/takeda-m/space/dataset/FFHQ/all/ ./outputs/ファイル名
```
flower
```
python -m pytorch_fid /home/yanai-lab/takeda-m/space/dataset/102flowers/passiflora_resize/ ./outputs/ファイル名
```
anime
```
python -m pytorch_fid /home/yanai-lab/takeda-m/space/dataset/Danbooru2019/anime500_resize/ ./outputs/ファイル名
```
bird
```
python -m pytorch_fid /home/yanai-lab/takeda-m/space/dataset/bird/african_firefinch_resize/ ./outputs/ファイル名
```
car
```
python -m pytorch_fid /home/yanai-lab/takeda-m/space/dataset/car/bmw_resize/ ./outputs/ファイル名
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
<!-- | biggan128-conv1x1   | 1x1convのパラメータをそのまま学習    | -->
| biggan128-conv1x1-2 | 学習率低め |
| biggan128-conv1x1-3 | メインモデル |
|                     |                         |


