import time

import streamlit as st
from transformer_lens import HookedTransformer

from attention_pattern import generate_attention_heatmaps
from display_utils import create_svg_html_content
from logits import save_all_logits_figures
from model import get_cache, get_output, visualize_model
from prompt import check_answer_correctness, get_random_prompt


@st.cache_resource
def load_model():
    return HookedTransformer.from_pretrained("gpt2-small", device="cpu")


# 初期設定
st.set_page_config(page_title="Visualize LLM Demo", layout="wide")
st.title("Visualize LLM Demo")
model = load_model()

# 横並びレイアウト
col1, col2, col3 = st.columns([4, 1, 1])  # テキスト入力 : Go : Random = 4:1:1 の比率

# Random ボタンが押された時の処理
if "random_prompt" not in st.session_state:
    st.session_state.random_prompt = ""
if "random_answer" not in st.session_state:
    st.session_state.random_answer = ""

with col3:
    if st.button("Random"):
        prompt_text, answer_text = get_random_prompt()
        st.session_state.random_prompt = prompt_text
        st.session_state.random_answer = answer_text
        st.rerun()

with col1:
    prompt = st.text_input(
        label="プロンプト入力",
        placeholder="プロンプトを入力してください（例：Sendai is located in the country of）",
        label_visibility="collapsed",
        value=st.session_state.random_prompt,
    )

with col2:
    run = st.button("Go")

# 答え入力エリア
expected_answer = st.text_input(
    label="答え",
    placeholder="プロンプトに対応する答えを入力してください（例：Japan）",
    label_visibility="collapsed",
    help="このプロンプトに対してモデルが出力すると期待される答えを入力してください",
    value=st.session_state.random_answer,
)

# ボタンが押されたときだけ処理を実行
if run and prompt:
    # モデルのキャッシュと logits を取得
    logits, cache = get_cache(model, prompt)

    # モデルの出力を取得
    output = get_output(model, logits)

    # 一致チェック
    is_correct = check_answer_correctness(model, prompt, logits, expected_answer)

    # 表示用のプレースホルダーを作成
    visualization_placeholder = st.empty()

    # 簡易版の HTML コンテンツを生成して表示 (後続の処理を待つ間に表示)
    init_path = "figures/graph_init.svg"
    visualize_model(model, filename=init_path, use_urls=False)
    max_height = 800
    margin = 20
    html_content_init = create_svg_html_content(
        init_path,
        max_height=max_height,
        input=prompt,
        output=output,
        is_correct=is_correct,
    )

    # プレースホルダーに簡易版を表示
    with visualization_placeholder.container():
        st.components.v1.html(
            html_content_init, height=max_height + margin, scrolling=False
        )

    # Attention Pattern の生成
    start_time = time.time()
    generate_attention_heatmaps(
        model=model,
        cache=cache,
        prompt=prompt,
        output_dir="figures/attention_patterns",
    )
    elapsed_time = time.time() - start_time
    print(f"Attention Pattern の生成にかかった時間: {elapsed_time:.2f}秒")

    # logits の可視化
    start_time = time.time()
    save_all_logits_figures(model, cache)
    elapsed_time = time.time() - start_time
    print(f"Logits の可視化にかかった時間: {elapsed_time:.2f}秒")

    # モデルの可視化
    start_time = time.time()
    output_path = "figures/model_visualization.svg"
    visualize_model(model, filename=output_path, use_urls=True)
    elapsed_time = time.time() - start_time
    print(f"モデルの可視化にかかった時間: {elapsed_time:.2f}秒")

    # 完全版の HTML コンテンツを生成してプレースホルダーを更新
    html_content_final = create_svg_html_content(
        output_path,
        max_height=max_height,
        input=prompt,
        output=output,
        is_correct=is_correct,
    )

    # プレースホルダーに完全版を表示 (簡易版を上書き)
    with visualization_placeholder.container():
        st.components.v1.html(
            html_content_final, height=max_height + margin, scrolling=False
        )
