import os

from PIL import Image
from transformer_lens import HookedTransformer


def combine_attention_map_and_logits(
    model: HookedTransformer,
    attention_dir: str = "figures/attention_patterns",
    logits_dir: str = "figures/logits",
    output_dir: str = "figures/combined",
    target_height: int = 500,
):
    """
    Attention Map と Logits の画像を横に結合して保存する関数.

    Args:
        model (HookedTransformer): Transformer モデルのインスタンス.
        attention_dir (str): Attention Map 画像のディレクトリパス.
        logits_dir (str): Logits 画像のディレクトリパス.
        output_dir (str): 出力ディレクトリのベースパス (default: "figures/combined").
        target_height (int): 結合後の画像の高さ (default: 500).

    Returns:
        None
    """
    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)

    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            attn_file = f"L{layer:02d}_H{head:02d}.png"
            logits_file = f"L{layer:02d}_H{head:02d}.png"

            attention_path = os.path.join(attention_dir, attn_file)
            logits_path = os.path.join(logits_dir, logits_file)

            if os.path.exists(attention_path) and os.path.exists(logits_path):
                output_path = os.path.join(output_dir, f"L{layer:02d}_H{head:02d}.png")
                _combine_images_horizontally(
                    attention_path,
                    logits_path,
                    output_path,
                    target_height=target_height,
                )
        # MLP 層の Logits 画像はそのまま保存
        mlp_logits_path = os.path.join(logits_dir, f"L{layer:02d}.png")
        if os.path.exists(mlp_logits_path):
            output_path = os.path.join(output_dir, f"L{layer:02d}.png")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            Image.open(mlp_logits_path).save(output_path)


def _combine_images_horizontally(
    image_path1: str,
    image_path2: str,
    output_path: str,
    target_height: int = None,
    background_color: tuple = (255, 255, 255),
) -> str:
    """
    2つの画像を高さを揃えて横に並べた新しい画像を作成して保存する関数.

    Args:
        image_path1 (str): 1つ目の画像のパス
        image_path2 (str): 2つ目の画像のパス
        output_path (str): 出力画像のパス
        target_height (int, optional): 目標の高さ. None の場合は2つの画像の最小の高さに合わせる
        spacing (int): 画像間のスペース（ピクセル）. Defaults to 10.
        background_color (tuple): 背景色 (R, G, B). Defaults to (255, 255, 255).

    Returns:
        str: 保存された画像のパス
    """
    # 画像を読み込み
    img1 = Image.open(image_path1)
    img2 = Image.open(image_path2)

    # RGBA モードに変換（透明度対応）
    if img1.mode != "RGBA":
        img1 = img1.convert("RGBA")
    if img2.mode != "RGBA":
        img2 = img2.convert("RGBA")

    # 目標の高さを決定
    if target_height is None:
        target_height = min(img1.height, img2.height)

    # 高さを揃えるために画像をリサイズ
    # アスペクト比を保ちながらリサイズ
    aspect_ratio1 = img1.width / img1.height
    aspect_ratio2 = img2.width / img2.height

    new_width1 = int(target_height * aspect_ratio1)
    new_width2 = int(target_height * aspect_ratio2)

    img1_resized = img1.resize((new_width1, target_height), Image.Resampling.LANCZOS)
    img2_resized = img2.resize((new_width2, target_height), Image.Resampling.LANCZOS)

    # 新しい画像のサイズを計算
    total_width = new_width1 + new_width2
    total_height = target_height

    # 新しい画像を作成
    combined_img = Image.new(
        "RGBA", (total_width, total_height), background_color + (255,)
    )

    # 画像を配置
    combined_img.paste(img1_resized, (0, 0), img1_resized)
    combined_img.paste(img2_resized, (new_width1, 0), img2_resized)

    # 出力ディレクトリを作成
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 画像を保存
    combined_img.save(output_path, "PNG", optimize=True)

    return output_path


def create_svg_html_content(
    svg_path: str, max_height: int = 800, input: str = None, output: str = None
) -> str:
    """
    SVG 画像と入出力テキストを含む HTML コンテンツを生成する関数.

    Args:
        svg_path (str): SVG 画像のパス.
        max_height (int): 最大高さ.
        input (str): 入力テキスト (SVG 画像の下に表示).
        output (str): 出力テキスト (SVG 画像の上に表示).

    Returns:
        str: HTML コンテンツの文字列.
    """
    # SVG ファイルを読み込み
    with open(svg_path, "r", encoding="utf-8") as f:
        svg_content = f.read()

    # HTML コンテンツ
    html_content = f"""
    <div style="
        width: 100%;
        height: {max_height}px;
        border: 1px solid #000000;
        border-radius: 4px;
        background-color: #000000;
        padding: 10px;
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        align-items: center;
    ">
        <!-- 出力テキスト (上部) -->
        <div style="
            width: 100%;
            text-align: center;
            color: white;
            font-weight: bold;
            font-size: 16px;
            padding: 5px 0;
            background-color: #000000;
            border-radius: 4px;
            margin-bottom: 10px;
            {"display: block;" if output else "display: none;"}
        ">
            出力：{output if output else ""}
        </div>

        <!-- SVG 画像 (中央) -->
        <div id="svgContainer" style="
            flex: 1;
            width: 100%;
            display: flex;
            justify-content: center;
            align-items: center;
            overflow: auto;
        ">
            {svg_content.replace("<svg", '<svg style="max-width: 100%; max-height: 100%; width: auto; height: auto; object-fit: contain;" preserveAspectRatio="xMidYMid meet"')}
        </div>

        <!-- 入力テキスト (下部) -->
        <div style="
            width: 100%;
            text-align: center;
            color: white;
            font-weight: bold;
            font-size: 16px;
            padding: 5px 0;
            background-color: #000000;
            border-radius: 4px;
            margin-top: 10px;
            {"display: block;" if input else "display: none;"}
        ">
            入力：{input if input else ""}
        </div>
    </div>

    <!-- 画像表示用のモーダル -->
    <div id="imageModal" style="display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background-color: rgba(0,0,0,0.8); z-index: 1000; justify-content: center; align-items: center;">
        <div style="position: relative; max-width: 90%; max-height: 90%; display: flex; justify-content: center; align-items: center;">
            <img id="modalImage" style="max-width: 100%; max-height: 100%; object-fit: contain; display: block;">
            <button onclick="closeModal()" style="position: absolute; top: 10px; right: 10px; background: white; border: none; padding: 8px 12px; cursor: pointer; border-radius: 5px; box-shadow: 0 2px 10px rgba(0,0,0,0.3); font-size: 16px;">×</button>
        </div>
    </div>

    <style>
        #svgContainer svg {{
            max-width: 100% !important;
            max-height: 100% !important;
            width: auto !important;
            height: auto !important;
            object-fit: contain !important;
        }}
    </style>

    <script>
    function showImage(imageB64) {{
        console.log('showImage called with data:', imageB64 ? 'available' : 'none');
        const modal = document.getElementById('imageModal');
        const modalImage = document.getElementById('modalImage');

        if (imageB64) {{
            modalImage.src = 'data:image/png;base64,' + imageB64;
            modal.style.display = 'flex';

            // 画像読み込み完了後にサイズを調整
            modalImage.onload = function() {{
                const container = modalImage.parentElement;
                const containerWidth = container.clientWidth;
                const containerHeight = container.clientHeight;
                const imageWidth = modalImage.naturalWidth;
                const imageHeight = modalImage.naturalHeight;

                // アスペクト比を保ちながら最適なサイズを計算
                const widthRatio = containerWidth / imageWidth;
                const heightRatio = containerHeight / imageHeight;
                const ratio = Math.min(widthRatio, heightRatio);

                if (ratio < 1) {{
                    modalImage.style.width = (imageWidth * ratio) + 'px';
                    modalImage.style.height = (imageHeight * ratio) + 'px';
                }}
            }};
        }} else {{
            console.error('No image data provided');
            alert('画像データが見つかりません');
        }}
    }}

    function closeModal() {{
        const modal = document.getElementById('imageModal');
        modal.style.display = 'none';
    }}

    // SVG 内のクリックイベントを処理
    document.addEventListener('DOMContentLoaded', function() {{
        console.log('Setting up SVG click handlers');

        // SVG 内のすべてのリンク要素を取得
        setTimeout(() => {{
            const svgElements = document.querySelectorAll('#svgContainer svg *[href], #svgContainer svg *[xlink\\\\:href]');
            console.log('Found SVG elements with href:', svgElements.length);

            svgElements.forEach(element => {{
                const href = element.getAttribute('href') || element.getAttribute('xlink:href');
                if (href && href.startsWith('javascript:showImage')) {{
                    element.style.cursor = 'pointer';
                    console.log('Adding click handler to:', element.tagName, href);

                    element.addEventListener('click', function(e) {{
                        e.preventDefault();
                        console.log('Element clicked:', href);

                        try {{
                            // JavaScript URL から Base64 データを抽出して実行
                            const match = href.match(/showImage\\('([^']+)'\\)/);
                            if (match) {{
                                showImage(match[1]);
                            }} else {{
                                console.error('Could not extract image data from href:', href);
                            }}
                        }} catch (error) {{
                            console.error('Error executing JavaScript:', error);
                        }}
                    }});
                }}
            }});
        }}, 100);
    }});

    // モーダルのクリックイベント (背景クリックで閉じる)
    document.getElementById('imageModal').addEventListener('click', function(e) {{
        if (e.target === this) {{
            closeModal();
        }}
    }});

    // ESC キーでモーダルを閉じる
    document.addEventListener('keydown', function(e) {{
        if (e.key === 'Escape') {{
            closeModal();
        }}
    }});
    </script>
    """

    return html_content
