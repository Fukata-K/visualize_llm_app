def create_svg_html_content(
    svg_path: str,
    max_height: int = 800,
    input: str = None,
    output: str = None,
    is_correct: bool = None,
) -> str:
    """
    SVG 画像と入出力テキストを含む HTML コンテンツを生成する関数.

    Args:
        svg_path (str): SVG 画像のパス.
        max_height (int): 最大高さ.
        input (str): 入力テキスト (SVG 画像の下に表示).
        output (str): 出力テキスト (SVG 画像の上に表示).
        is_correct (bool): 出力が正しいかどうか (True: 正解/緑色, False: 不正解/赤色, None: デフォルト).

    Returns:
        str: HTML コンテンツの文字列.
    """
    # SVG ファイルを読み込み
    with open(svg_path, "r", encoding="utf-8") as f:
        svg_content = f.read()

    # 出力テキストの色とアイコンを決定
    if is_correct is None:
        output_color = "white"
        output_icon = ""
    elif is_correct:
        output_color = "#4CAF50"  # 緑色
        output_icon = "✅ "
    else:
        output_color = "#F44336"  # 赤色
        output_icon = "❌ "

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
            color: {output_color};
            font-weight: bold;
            font-size: 16px;
            padding: 5px 0;
            background-color: #000000;
            border-radius: 4px;
            margin-bottom: 10px;
            {"display: block;" if output else "display: none;"}
        ">
            出力：{output_icon}{output if output else ""}
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
        const modalContainer = modal.querySelector('div');

        if (imageB64) {{
            // モーダル内容をクリアして単一画像用のレイアウトに初期化
            modalContainer.innerHTML = '';

            // 単一画像用のコンテナを作成
            const imageContainer = document.createElement('div');
            imageContainer.style.cssText = `
                position: relative;
                max-width: 90%;
                max-height: 90%;
                display: flex;
                justify-content: center;
                align-items: center;
            `;

            // 画像要素を作成
            const modalImage = document.createElement('img');
            modalImage.id = 'modalImage';
            modalImage.style.cssText = `
                max-width: 100%;
                max-height: 100%;
                object-fit: contain;
                display: block;
            `;
            modalImage.src = 'data:image/png;base64,' + imageB64;

            // 閉じるボタンを作成
            const closeButton = document.createElement('button');
            closeButton.onclick = closeModal;
            closeButton.textContent = '×';
            closeButton.style.cssText = `
                position: absolute;
                top: 10px;
                right: 10px;
                background: white;
                border: none;
                padding: 8px 12px;
                cursor: pointer;
                border-radius: 5px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.3);
                font-size: 16px;
            `;

            imageContainer.appendChild(modalImage);
            imageContainer.appendChild(closeButton);
            modalContainer.appendChild(imageContainer);

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

    function showDualImages(imageB64_1, imageB64_2, title1, title2) {{
        console.log('showDualImages called');
        const modal = document.getElementById('imageModal');
        const modalContainer = modal.querySelector('div');

        if (imageB64_1 && imageB64_2) {{
            // モーダル内容をクリア
            modalContainer.innerHTML = '';

            // 2枚の画像を横並びで表示するコンテナを作成
            const dualContainer = document.createElement('div');
            dualContainer.style.cssText = `
                display: flex;
                gap: 20px;
                max-width: 90vw;
                max-height: 90vh;
                align-items: center;
                justify-content: center;
            `;

            // 左側の画像コンテナ
            const leftContainer = document.createElement('div');
            leftContainer.style.cssText = `
                flex: 1;
                display: flex;
                flex-direction: column;
                align-items: center;
                max-width: 45vw;
                max-height: 90vh;
            `;

            const leftTitle = document.createElement('h3');
            leftTitle.textContent = title1;
            leftTitle.style.cssText = `
                color: white;
                margin: 0 0 10px 0;
                font-size: 16px;
                text-align: center;
            `;

            const leftImage = document.createElement('img');
            leftImage.src = 'data:image/png;base64,' + imageB64_1;
            leftImage.style.cssText = `
                max-width: 100%;
                max-height: calc(90vh - 40px);
                object-fit: contain;
            `;

            leftContainer.appendChild(leftTitle);
            leftContainer.appendChild(leftImage);

            // 右側の画像コンテナ
            const rightContainer = document.createElement('div');
            rightContainer.style.cssText = `
                flex: 1;
                display: flex;
                flex-direction: column;
                align-items: center;
                max-width: 45vw;
                max-height: 90vh;
            `;

            const rightTitle = document.createElement('h3');
            rightTitle.textContent = title2;
            rightTitle.style.cssText = `
                color: white;
                margin: 0 0 10px 0;
                font-size: 16px;
                text-align: center;
            `;

            const rightImage = document.createElement('img');
            rightImage.src = 'data:image/png;base64,' + imageB64_2;
            rightImage.style.cssText = `
                max-width: 100%;
                max-height: calc(90vh - 40px);
                object-fit: contain;
            `;

            rightContainer.appendChild(rightTitle);
            rightContainer.appendChild(rightImage);

            // 閉じるボタン
            const closeButton = document.createElement('button');
            closeButton.onclick = closeModal;
            closeButton.textContent = '×';
            closeButton.style.cssText = `
                position: absolute;
                top: 10px;
                right: 10px;
                background: white;
                border: none;
                padding: 8px 12px;
                cursor: pointer;
                border-radius: 5px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.3);
                font-size: 16px;
                z-index: 1001;
            `;

            dualContainer.appendChild(leftContainer);
            dualContainer.appendChild(rightContainer);
            modalContainer.appendChild(dualContainer);
            modalContainer.appendChild(closeButton);

            modal.style.display = 'flex';
        }} else {{
            console.error('One or both image data missing');
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
                if (href && (href.startsWith('javascript:showImage') || href.startsWith('javascript:showDualImages'))) {{
                    element.style.cursor = 'pointer';
                    console.log('Adding click handler to:', element.tagName, href);

                    element.addEventListener('click', function(e) {{
                        e.preventDefault();
                        console.log('Element clicked:', href);

                        try {{
                            // JavaScript URL から関数呼び出しを抽出して実行
                            if (href.includes('showDualImages')) {{
                                const match = href.match(/showDualImages\\('([^']+)',\\s*'([^']+)',\\s*'([^']+)',\\s*'([^']+)'\\)/);
                                if (match) {{
                                    showDualImages(match[1], match[2], match[3], match[4]);
                                }} else {{
                                    console.error('Could not extract dual image data from href:', href);
                                }}
                            }} else if (href.includes('showImage')) {{
                                const match = href.match(/showImage\\('([^']+)'\\)/);
                                if (match) {{
                                    showImage(match[1]);
                                }} else {{
                                    console.error('Could not extract image data from href:', href);
                                }}
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
