import streamlit as st
import torch
from transformer_lens import HookedTransformer

from attention_pattern import generate_attention_heatmaps
from display_utils import create_svg_html_content
from model import get_cache, get_output, visualize_model

# 初期設定
st.set_page_config(page_title="Visualize LLM Demo", layout="wide")
st.title("Visualize LLM Demo")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HookedTransformer.from_pretrained("gpt2-small", device=device)

# 横並びレイアウト
col1, col2 = st.columns([4, 1])  # テキスト入力 : ボタン = 4:1 の比率

with col1:
    prompt = st.text_input(
        label="プロンプト入力",
        placeholder="プロンプトを入力してください（例：Sendai is located in the country of）",
        label_visibility="collapsed",
    )

with col2:
    run = st.button("Go")

# ボタンが押されたときだけ処理を実行
if run and prompt:
    # モデルのキャッシュと logits を取得
    logits, cache = get_cache(model, prompt, device=device)

    # モデルの出力を取得
    output = get_output(model, logits)

    # Attention Pattern の生成
    generate_attention_heatmaps(
        model=model,
        cache=cache,
        prompt=prompt,
        output_dir="figures/attention_patterns",
    )

    # モデルの可視化
    output_path = "figures/model_visualization.svg"
    visualize_model(model, filename=output_path, use_urls=True)

    # HTML コンテンツを生成して表示
    max_height = 800
    margin = 20
    html_content = create_svg_html_content(
        output_path, max_height=max_height, input=prompt, output=output
    )
    st.components.v1.html(html_content, height=max_height + margin, scrolling=False)
