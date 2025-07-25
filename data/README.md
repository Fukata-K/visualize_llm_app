# データファイル

## prompt_sample.csv

このファイルには，アプリケーションで使用するサンプルプロンプトと期待される回答のペアが含まれています．

### データの出典

`prompt_sample.csv` は [evandez/relations](https://github.com/evandez/relations/tree/main/data) リポジトリのデータを使用して作成されたものです．

### ファイル形式

CSVファイルには以下のカラムが含まれています：

- `prompt`: モデルに入力するプロンプト文字列
- `subject`: プロンプトの主語（未使用）
- `object`: 期待される回答（プロンプトの次に来るべき単語）
- `keywords`: プロンプトのキーワード（未使用）

### 使用方法

このファイルは `prompt.py` の `get_random_prompt()` 関数で読み込まれ，`Random Sample` がクリックされたときにランダムなプロンプトと回答のペアを提供します．

### サンプルデータの例

```csv
prompt,subject,object,keywords
Rio de Janeiro is located in the country of,Rio de Janeiro, Brazil, country
Sendai is located in the country of,Sendai, Japan, country
MacBook Pro is a product of,MacBook Pro, Apple, product
```