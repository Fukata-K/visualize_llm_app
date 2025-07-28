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

# デモの説明
st.markdown("""
### 🧠 このデモについて

このアプリでは、**AI（大規模言語モデル）が次の単語を予測する仕組み**を可視化できます。

##### なぜ次の単語を予測するのか？
ChatGPT や Google 翻訳などの AI は、「文章の次の単語を予測する」ことを繰り返すことで文章を生成しています。  
例えば「私は明日、学校に」という文章に対して、AI は「行く」「向かう」などの適切な単語を予測します。  
この予測の仕組みを理解することで、AI がどのように文章を理解し、生成しているかが分かります。

---

### 📋 使い方
1. 文章を**途中まで**入力してください（例：「Sendai is located in the country of」）
2. **AI が予測すべき次の単語**を入力してください（例：「Japan」）
3. **Go ボタン**をクリックすると、AI の内部でどのように予測が行われているかが図として表示されます
4. すべての処理が完了したら、**図をクリック**して詳細情報を確認できます（30 秒ほどかかります）

💡 **迷ったら「Random Sample」ボタンを押すと、サンプル文章が自動で入力されます**
""")

# セッション状態の初期化
if "random_prompt" not in st.session_state:
    st.session_state.random_prompt = ""
if "random_answer" not in st.session_state:
    st.session_state.random_answer = ""

st.markdown("---")
st.markdown("### 📝 入力エリア")

# プロンプト入力エリア (プロンプト + Random ボタン)
prompt_col1, prompt_col2 = st.columns([4, 1])

with prompt_col1:
    prompt = st.text_input(
        label="✏️ 文章を**途中まで**入力してください（英語推奨 / 迷ったら **Random Sample** をクリック 👉️）",
        placeholder="例：Sendai is located in the country of",
        value=st.session_state.random_prompt,
        help="AI に続きを予測させたい文章を途中まで入力してください",
    )

with prompt_col2:
    st.markdown("<br>", unsafe_allow_html=True)  # ラベル分のスペース調整
    if st.button(
        "🎲 Random Sample",
        help="サンプル文章と答えを自動で入力します",
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
        label="✅ AI が予測すべき**次の単語**を入力してください（**入力したら Go ボタンをクリック** 🚀）",
        placeholder="例：Japan",
        value=st.session_state.random_answer,
        help="上の文章に続く次の単語として、AI が予測すべき正解を入力してください",
    )

with answer_col2:
    st.markdown("<br>", unsafe_allow_html=True)  # ラベル分のスペース調整
    run = st.button(
        "🚀 Go", help="分析を開始", use_container_width=True, type="primary"
    )

st.markdown("---")

# ボタンが押されたときだけ処理を実行
if run and prompt:
    st.markdown("""
    ### 🔍️ AI の内部可視化
    下の図は AI の「思考過程」を表しています。

    **図の見方**  
    ・**Input**：入力された文章を **AI が処理できる形に加工**する部分  
    ・**A0.H0, A1.H1** など：「**どの単語に注目すべきか**」を決める場所（Attention 層）  
    ・**MLP0, MLP1** など：注目した情報をもとに「**次に出す単語のヒント**」を作る場所（MLP 層）  
    ・**Output**：🎯 **最終的な予測結果**（AI が選ぶ次の単語）を決定する部分

    **色の意味**  
    ・**緑色が濃い**部分：入力した「期待される単語」を**上位で予測**している（正解に近い）  
    ・**白色**の部分：期待される単語の**順位が低い**（正解から遠い）

    💡 **AI は何万種類もの単語から次の単語を選んでいます。緑色が濃いほど、その部分で正解の単語が候補の上位に入っています。**

    ---

    ### 📍 クリックして詳細をみる
    """)

    st.warning("""
    ⚠️ **全ての処理が終了したら四角い箱をクリックしてみてください** ⚠️

    クリックすると、AI がどの単語に注目しているかや、どの単語を選びそうかが分かります。

    各四角い箱をクリックすると以下の情報が表示されます  
    ・📈 **注意パターン**：その処理部分がどの単語に注目しているか  
    ・🏆 **予測ランキング**：その時点での単語予測の順位

    **📊 おすすめ：濃い緑色の箱をクリック** 👈️ 正解に近い予測をしている部分の詳細が見られます
    """)

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
            "⏳ 可視化を生成中です... 約 30 秒ほど待つと四角い箱をクリックして詳細情報を閲覧できるようになります"
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

    # 完全版の HTML コンテンツを生成してプレースホルダーを更新
    output_path = "figures/graphs/graph.svg"
    visualize_model(
        model, filename=output_path, use_urls=True, object_ranks=object_ranks
    )
    html_content_final = create_svg_html_content(
        output_path,
        max_height=max_height,
        input=prompt,
        output=output,
        is_correct=is_correct,
    )

    # プレースホルダーに完全版を表示 (簡易版を上書き)
    with visualization_placeholder.container():
        st.success("🎉 可視化が完了しました！")
        st.components.v1.html(
            html_content_final, height=max_height + margin, scrolling=False
        )
