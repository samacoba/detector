# Any detector

いろいろなものを検出します。
<img src="https://raw.githubusercontent.com/samacoba/detector/master/sample.png">


## Requirement

- Chainer（GPU計算は必要）
- Jupyter
- numpy,skimage,bokeh

## Usage
- 教師データの作成：get_position.ipynb  
「imgA.png」を読み込んで手動で丸を付け、「posA.pkl」に位置を保存  
(click_pos.html内のjavascriptを使っています）

- 学習とカウントを実行：main.ipynb  
「posA.pkl」の位置と「imgA.png」を読み込んで学習  
「imgB.png」を読み込んでカウント  
