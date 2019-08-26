# YANS ハッカソン 2019 (Kaggleハッカソンコース)
Kaggle ハッカソン
- 2つの質問が同じ質問か 否かを分類するタスク
- 2値分類問題 (0: ペアでない. 1: ペアである)

- Quora Question Pairs
https://www.kaggle.com/c/quora-question-pairs

## ベースラインモデル
- 1. SVM  + TF-IDF feature
- 2. MLP + Glove vectors (sum of word vectors)

https://colab.research.google.com/drive/1sCOOtH9PRtCBK6Z2LgXxeOhuNpED1m9H

質問は YANS Slackで @Motoki Sato まで。


## データセット
使った訓練データ | サイズ | ペナルティー付きスコア | データ
--------------| :---: | :---: | :---:
訓練データ　全体 | 222,074ペア | Accuracy - 5.0 % | [train_222070.csv](http://sato-motoki.com/yans/hackathon2019/dataset/train_222070.csv)
訓練データ　中 | 100,000ペア | Accuracy - 2.0 % | [train_100000.csv](http://sato-motoki.com/yans/hackathon2019/dataset/train_100000.csv)
訓練データ　小 | 10,000ペア | Accuracy  | [train_10000.csv](http://sato-motoki.com/yans/hackathon2019/dataset/train_10000.csv)

```
$ head train.csv
"question1","question2","is_duplicate"
"How do the people of Delhi feel about their chief minister?","How do people in Delhi feel about the AAP?","0"
"How long does chloroform knock you out for?","How long is a person knocked out for?","0"
```
※ train_10000.csv の中のデータはtrain_100000.csvは包含しています。（ソート順がランダムになっています）

## 開発データ (dev)
- 5,000ペア
- [dev.csv](http://sato-motoki.com/yans/hackathon2019/dataset/dev.csv)

## テストデータ
2日目の18:00に配布予定
http://sato-motoki.com/yans/hackathon2019/dataset/test.csv

test.csvの形式 (5000ペア)
```
$ head test.csv
"question1","question2"
"Which are hogehoge?","How strong is hogehoge?"
```

提出していただくファイル : predict.csv
5000行に 0, 1が予測されているファイルを提出してください。
 (@Motoki Satoまで DMで送ってください)
締め切り : 3日目 9:30まで
```
$ head predict.csv
0
1
0
1
```
