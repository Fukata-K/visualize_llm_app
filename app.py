import streamlit as st

from display_utils import visualize_svg

# 初期設定
st.set_page_config(page_title="Visualize LLM Demo", layout="wide")

# タイトル
st.title("Visualize LLM Demo")

# 横並びレイアウト
col1, col2 = st.columns([4, 1])  # テキスト入力 : ボタン = 4:1 の比率

with col1:
    prompt = st.text_input(
        label="プロンプト入力",
        placeholder="プロンプトを入力してください（例：A teacher typically works at a）",
        label_visibility="collapsed",
    )

with col2:
    run = st.button("Go")

# ボタンが押されたときだけ処理を実行
if run and prompt:
    st.success(f"プロンプト: {prompt}")

    visualize_svg("figures/model_visualization.svg", max_height=800)
