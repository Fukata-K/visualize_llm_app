import base64
import os
from pathlib import Path

import pygraphviz as pgv
import torch
from transformer_lens import ActivationCache, HookedTransformer


def get_cache(
    model: HookedTransformer,
    prompt: str,
) -> tuple[torch.Tensor, dict | ActivationCache]:
    """
    モデルのキャッシュを取得する関数.

    Args:
        model (HookedTransformer): Transformer モデルのインスタンス.
        prompt (str): モデルに入力するプロンプト.

    Returns:
        tuple[torch.Tensor, dict | ActivationCache]: モデルの出力とキャッシュ.
    """
    token_ids = model.to_tokens(prompt, prepend_bos=False)
    with torch.no_grad():
        logits, cache = model.run_with_cache(token_ids)
    cache.model = None
    return logits, cache


def visualize_model(
    model: HookedTransformer,
    filename: str = "figures/graph.svg",
    fillcolors: dict[str, str] = None,
    use_urls: bool = False,
    base_width: float = 1.5,
    base_height: float = 0.6,
    base_fontsize: float = 24,
    node_border_width: float = 5.0,
    edge_width: float = 3.0,
) -> None:
    """
    モデルのノードとエッジを pygraphviz を使って可視化する関数.

    Args:
        model (HookedTransformer): Transformer モデルのインスタンス.
        filename (str): 出力ファイル名. デフォルトは "figures/graph.svg".
        fillcolors (dict[str, str]): ノードの名前と色のマッピング. デフォルトは None.
        use_urls (bool): ノードに URL を使用するかどうか. デフォルトは False.
        base_width (float): ノードの基本幅.
        base_height (float): ノードの基本高さ.
        base_fontsize (float): ノードの基本フォントサイズ.
        node_border_width (float): ノードの枠線の幅.
        edge_width (float): エッジの太さ.

    Returns:
        None
    """
    graph = pgv.AGraph(
        directed=True,
        layout="neato",
        bgcolor="#FFFFFF",
        overlap="true",
        splines="true",
    )

    # ノード名とエッジ名を取得
    node_list = _create_node_list(model)
    edge_list = _create_edge_list(model)

    # デフォルト値の設定
    if fillcolors is None:
        color_map = {
            "a": "#FF7777",  # light red for attention nodes
            "m": "#CCFFCC",  # light green for MLP nodes
            "l": "#FFD700",  # gold for logits nodes
        }
        fillcolors = {node: (color_map.get(node[0], "#808080")) for node in node_list}
    if use_urls:
        urls = _get_url_dict(model)
    else:
        urls = {node: "" for node in node_list}

    # ノードを追加
    for node in node_list:
        fillcolor = fillcolors.get(node, "#FFFFFF")
        url = urls.get(node, "")
        pos = _get_node_position(model, node, base_width, base_height)
        graph.add_node(
            node,
            fillcolor=fillcolor,
            fontname="Helvetica",
            fontsize=base_fontsize,
            shape="box",
            style="filled, rounded",
            width=base_width,
            height=base_height,
            fixedsize=True,
            penwidth=node_border_width,
            URL=url,
            pos=f"{pos[0]},{pos[1]}!",
        )

    # エッジを追加
    for edge in edge_list:
        graph.add_edge(edge[0], edge[1], penwidth=edge_width)

    # グラフを描画して保存
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    graph.layout(prog="neato")
    graph.draw(filename, format="svg")


def _get_url_dict(
    model: HookedTransformer,
) -> dict[str, str]:
    """
    モデルのノードに対応する URL の辞書を取得する関数.

    Args:
        model (HookedTransformer): Transformer モデルのインスタンス.

    Returns:
        dict[str, str]: ノード名と URL のマッピング.
    """
    url_dict = {}
    for node in _create_node_list(model):
        if node.startswith("a"):
            layer, head = map(int, node[1:].split(".h"))
            image_path = f"figures/attention_patterns/L{layer:02d}_H{head:02d}.png"
            url_dict[node] = f"javascript:showImage('{_image_to_base64(image_path)}')"
        else:
            url_dict[node] = ""
    return url_dict


def _image_to_base64(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return None


def _get_node_position(
    model: HookedTransformer,
    node_name: str,
    base_width: float = 1.5,
    base_height: float = 0.6,
) -> tuple[float, float]:
    """
    ノードの位置を計算するヘルパー関数.

    Args:
        model (HookedTransformer): Transformer モデルのインスタンス.
        node_name (str): ノードの名前.
        base_width (float): ノードの基本幅.
        base_height (float): ノードの基本高さ.

    Returns:
        tuple[float, float]: ノードの (x, y) 座標.
    """
    x_spacing = base_width * 1.5  # ノード幅の間隔
    y_spacing = base_height * 1.5  # ノード高さの間隔

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    if node_name == "input":
        return (0, 0)  # Input を基準点とする

    elif node_name.startswith("a"):
        layer = int(node_name[1:].split(".")[0])
        head = int(node_name.split(".h")[1])
        # Input を中央に配置するため Head を中央からの相対位置として計算
        # ノード幅の間隔を使用
        x = (head - (n_heads - 1) / 2) * x_spacing
        # Attention 層：layer * 2 + 1 の位置
        y = (layer * 2 + 1) * y_spacing
        return (x, y)

    elif node_name.startswith("m"):
        layer = int(node_name[1:])
        # MLP 層：layer * 2 + 2 の位置 (各 layer 内で Attention の後)
        y = (layer * 2 + 2) * y_spacing
        return (0, y)

    elif node_name == "logits":
        # 最終層の後に配置
        return (0, (n_layers * 2 + 1) * y_spacing)


def _create_node_list(model: HookedTransformer) -> list:
    """
    pygraphviz で描画する際のノードの名前リストを作成する関数.

    Args:
        model (HookedTransformer): Transformer モデルのインスタンス.

    Returns:
        list: ノードの名前リスト (例: ["input", "a0.h0", "a0.h1", ..., "m0", ..., "logits"]).
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    node_list = []

    # Input ノード
    node_list.append("input")

    # Attention Head と MLP
    for layer in range(n_layers):
        for head in range(n_heads):
            node_list.append(f"a{layer}.h{head}")
        node_list.append(f"m{layer}")

    # Output ノード
    node_list.append("logits")

    return node_list


def _create_edge_list(model: HookedTransformer) -> list:
    """
    pygraphviz で描画する際のエッジのリストを作成する関数.

    Args:
        model (HookedTransformer): Transformer モデルのインスタンス.

    Returns:
        list: エッジのリスト (例: [("input", "a0.h0"), ("a0.h0", "m0"), ...]).
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    edge_list = []

    # Input から最初の Attention Head へのエッジ
    for head in range(n_heads):
        edge_list.append(("input", f"a0.h{head}"))

    # Attention Head から MLP へのエッジ
    for layer in range(n_layers):
        for head in range(n_heads):
            edge_list.append((f"a{layer}.h{head}", f"m{layer}"))

    # MLP から次の Attention Head へのエッジ
    for layer in range(n_layers - 1):
        for head in range(n_heads):
            edge_list.append((f"m{layer}", f"a{layer + 1}.h{head}"))

    # MLP から logits へのエッジ
    edge_list.append((f"m{n_layers - 1}", "logits"))

    return edge_list
