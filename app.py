import streamlit as st
from transformer_lens import HookedTransformer

from attention_pattern import generate_attention_heatmaps
from display_utils import create_svg_html_content
from logits import calculate_all_logits_object_rank, save_all_logits_figures
from model import get_cache, get_output, visualize_model
from prompt import check_answer_correctness, get_random_prompt


@st.cache_resource
def load_model():
    return HookedTransformer.from_pretrained("gpt2-small", device="cpu")


# 初期設定
st.set_page_config(page_title="Visualize LLM Demo", layout="wide")
st.title("Visualize LLM Demo")
model = load_model()

# セッション状態の初期化
if "random_prompt" not in st.session_state:
    st.session_state.random_prompt = ""
if "random_answer" not in st.session_state:
    st.session_state.random_answer = ""

st.markdown("---")

# プロンプト入力エリア (プロンプト + Random ボタン)
prompt_col1, prompt_col2 = st.columns([4, 1])

with prompt_col1:
    prompt = st.text_input(
        label="📝 プロンプト入力（英語推奨 / 迷ったら Random Sample をクリック 👉️）",
        placeholder="例：Sendai is located in the country of",
        value=st.session_state.random_prompt,
        help="モデルに入力する文章を記入してください",
    )

with prompt_col2:
    st.markdown("<br>", unsafe_allow_html=True)  # ラベル分のスペース調整
    if st.button(
        "🎲 Random Sample",
        help="ランダムなプロンプトと答えのペアを生成",
        use_container_width=True,
    ):
        prompt_text, answer_text = get_random_prompt()
        st.session_state.random_prompt = prompt_text
        st.session_state.random_answer = answer_text
        st.rerun()

# 答え入力エリア (答え + Go ボタン)
answer_col1, answer_col2 = st.columns([4, 1])

with answer_col1:
    expected_answer = st.text_input(
        label="✅ 期待される答え（プロンプトの次の単語 / 入力したら Go をクリック 🚀）",
        placeholder="例：Japan",
        value=st.session_state.random_answer,
        help="このプロンプトに対してモデルが出力すると期待される答えを入力してください",
    )

with answer_col2:
    st.markdown("<br>", unsafe_allow_html=True)  # ラベル分のスペース調整
    run = st.button(
        "🚀 Go", help="分析を開始", use_container_width=True, type="primary"
    )

st.markdown("---")

# ボタンが押されたときだけ処理を実行
if run and prompt:
    # モデルのキャッシュと logits を取得
    logits, cache = get_cache(model, prompt)

    # モデルの出力を取得
    output = get_output(model, logits)

    # 一致チェック
    is_correct = check_answer_correctness(model, prompt, logits, expected_answer)

    # 各層の logits からオブジェクトの順位を計算
    object_ranks = calculate_all_logits_object_rank(
        model=model,
        cache=cache,
        prompt=prompt,
        object=expected_answer,
    )

    # 表示用のプレースホルダーを作成
    visualization_placeholder = st.empty()

    # 簡易版の HTML コンテンツを生成して表示 (後続の処理を待つ間に表示)
    init_path = "figures/graphs/graph_init.svg"
    visualize_model(
        model, filename=init_path, use_urls=False, object_ranks=object_ranks
    )
    max_height = 1000
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
        st.info(
            "⏳ 可視化を生成中です... 約30秒ほど待つとノードをクリックして詳細情報を閲覧できるようになります"
        )
        st.components.v1.html(
            html_content_init, height=max_height + margin, scrolling=False
        )

    # Attention Pattern の生成
    generate_attention_heatmaps(
        model=model,
        cache=cache,
        prompt=prompt,
        output_dir="figures/attention_patterns",
    )

    # logits の可視化
    save_all_logits_figures(model, cache)

    # モデルの可視化
    output_path = "figures/graphs/graph.svg"
    visualize_model(
        model, filename=output_path, use_urls=True, object_ranks=object_ranks
    )

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
        st.success(
            "✅ 可視化が完了しました！ノードをクリックして詳細情報を閲覧できます"
        )
        st.components.v1.html(
            html_content_final, height=max_height + margin, scrolling=False
        )
