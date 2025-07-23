# Visualize LLM App

このアプリケーションは, 大規模言語モデル (LLM) の内部動作を可視化する Streamlit アプリです.
特定の入力に対してモデルがどのように動作しているかを, Attention Pattern や各構成要素の出力などの観点から視覚的に理解できます.

本アプリケーションは, オープンキャンパスの研究室展示用デモの一部として作成したものです.

## 環境セットアップ (Conda 推奨)

### 1. 本リポジトリの clone

```bash
git clone https://github.com/Fukata-K/visualize_llm_app.git
cd visualize_llm_app
```

### 2. Conda 環境の作成

```bash
conda env create -f environment.yml
conda activate visualize_llm
```

### 3. アプリの起動

```bash
streamlit run app.py
```
