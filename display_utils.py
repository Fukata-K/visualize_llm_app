import base64

import streamlit as st


def visualize_svg(svg_path: str, max_height: int = 800):
    """
    SVG 画像を Streamlit で表示する関数.

    Args:
        svg_path (str): SVG 画像のパス.
        max_height (int): 最大高さ.
    """
    with open(svg_path, "r") as f:
        svg_content = f.read()

    # SVGを Base64 エンコード
    svg_base64 = base64.b64encode(svg_content.encode("utf-8")).decode("utf-8")

    # HTML コンテンツ
    html_content = f"""
    <div style="
        width: 100%;
        height: {max_height}px;
        display: flex;
        justify-content: center;
        align-items: center;
        border: 1px solid #000000;
        border-radius: 4px;
        background-color: #000000;
        padding: 5px;
        box-sizing: border-box;
    ">
        <img
            src="data:image/svg+xml;base64,{svg_base64}"
            style="
                max-width: 100%;
                max-height: 100%;
                object-fit: contain;
            "
            alt="SVG Image"
        />
    </div>
    """

    # 画像が見切れるのを防ぐために, コンテナサイズにマージンを追加
    st.components.v1.html(html_content, height=max_height + 20, scrolling=False)
