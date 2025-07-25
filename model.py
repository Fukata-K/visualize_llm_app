import base64
import os
from pathlib import Path

import numpy as np
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
    token_ids = model.to_tokens(prompt, prepend_bos=False).cpu()
    with torch.no_grad():
        logits, cache = model.run_with_cache(token_ids)
    cache.model = None
    return logits, cache


def get_output(model: HookedTransformer, logits: torch.Tensor) -> str:
    """
    モデルの出力を取得する関数.
    ここでは, logits の最後のトークン位置の top1 トークンを取得して文字列に変換する.

    Args:
        model (HookedTransformer): Transformer モデルのインスタンス.
        logits (torch.Tensor): モデルの出力ロジット.

    Returns:
        str: 予測された次のトークン文字列.
    """
    top1_token_id = logits[0, -1].argmax().item()
    return model.to_string(top1_token_id)


def visualize_model(
    model: HookedTransformer,
    filename: str = "figures/graph.svg",
    fillcolors: dict[str, str] = None,
    use_urls: bool = False,
    object_ranks: dict[str, int] = None,
    base_width: float = 1.5,
    base_height: float = 0.6,
    base_fontsize: float = 24,
    node_border_width: float = 7.5,
    edge_width: float = 3.0,
) -> None:
    """
    モデルのノードとエッジを pygraphviz を使って可視化する関数.

    Args:
        model (HookedTransformer): Transformer モデルのインスタンス.
        filename (str): 出力ファイル名. デフォルトは "figures/graph.svg".
        fillcolors (dict[str, str]): ノードの名前と色のマッピング. デフォルトは None.
        use_urls (bool): ノードに URL を使用するかどうか. デフォルトは False.
        object_ranks (dict[str, int]): 各ノードのオブジェクトの順位を示す辞書. デフォルトは None.
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
        bgcolor="#000000",
        overlap="true",
        splines="true",
    )

    # ノード名とエッジ名を取得
    node_list = _create_node_list(model)
    edge_list = _create_edge_list(model)

    # デフォルト値の設定
    vocab_size = model.cfg.d_vocab
    valid_rank = vocab_size // 10
    colors = {}
    for node in node_list:
        colors[node] = (
            _get_color_from_rank(object_ranks.get(node, vocab_size), valid_rank)
            if object_ranks
            else "#FFFFFF"
        )
    if fillcolors is None:
        color_map = {
            "a": "#FF7777",  # light red for attention nodes
            "m": "#77FF77",  # light green for MLP nodes
            "o": "#FF7700",  # gold for output nodes
        }
        if object_ranks is None:
            fillcolors = {
                node: (color_map.get(node[0], "#808080")) for node in node_list
            }
        else:
            fillcolors = {}
            for node in node_list:
                if node.startswith("A"):
                    fillcolors[node] = color_map.get("a", "#808080")
                elif node.startswith("MLP"):
                    fillcolors[node] = _get_color_from_rank(
                        object_ranks.get(node, vocab_size), valid_rank
                    )
                elif node == "Output":
                    fillcolors[node] = _get_color_from_rank(
                        object_ranks.get(node, vocab_size), valid_rank
                    )
                else:
                    fillcolors[node] = "#808080"
    if use_urls:
        urls = _get_url_dict(model)
    else:
        urls = {node: "" for node in node_list}

    # ノードを追加
    for node in node_list:
        color = colors.get(node, "#FFFFFF")
        fillcolor = fillcolors.get(node, "#FFFFFF")
        url = urls.get(node, "")
        pos = _get_node_position(model, node, base_width, base_height)
        graph.add_node(
            node,
            color=color,
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
        if object_ranks is None:
            graph.add_edge(edge[0], edge[1], penwidth=edge_width, color="#FFFFFF")
        else:
            rank = object_ranks.get(edge[0], vocab_size)
            color = _get_color_from_rank(rank, valid_rank)
            graph.add_edge(edge[0], edge[1], penwidth=edge_width, color=color)

    # グラフを描画して保存
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    graph.layout(prog="neato")
    graph.draw(filename, format="svg")


def _get_color_from_rank(rank: int, rank_threshold: int) -> str:
    """
    target の rank に基づいて色を計算するヘルパー関数.
    順位が閾値以上 (数値が大きい) であれば白色, それ以外は順位が良い (数値が小さい) ほど緑色, 悪い (数値が大きい) ほど白色になる.

    Args:
        rank (int): target のランク
        rank_threshold (int): 順位の閾値 (この値未満は白色になる)

    Returns:
        str: 16進カラーコード (例: "#A1B2C3")
    """
    # 順位が閾値以上であればで白色
    if rank >= rank_threshold:
        return "#FFFFFF"

    log_rank = np.log(rank + 1e-8)
    log_threshold = np.log(rank_threshold + 1e-8)
    normalized_log_rank = log_rank / log_threshold if log_threshold > 0 else 0.0

    # 順位が良いほど緑, 悪いほど白になるように調整
    r = int(255 * normalized_log_rank)
    g = 255
    b = int(255 * normalized_log_rank)
    return f"#{r:02x}{g:02x}{b:02x}"


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
        if node.startswith("A"):
            layer, head = map(int, node[1:].split(".H"))
            attention_path = f"figures/attention_patterns/L{layer:02d}_H{head:02d}.png"
            logits_path = f"figures/logits/L{layer:02d}_H{head:02d}.png"
            attention_b64 = _image_to_base64(attention_path)
            logits_b64 = _image_to_base64(logits_path)
            url_dict[node] = (
                f"javascript:showDualImages('{attention_b64}', '{logits_b64}', 'Attention Pattern', 'Logits')"
            )
        elif node.startswith("MLP"):
            layer = int(node[3:])
            logits_path = f"figures/logits/L{layer:02d}.png"
            logits_b64 = _image_to_base64(logits_path)
            url_dict[node] = f"javascript:showImage('{logits_b64}')"
        elif node == "Output":
            logits_path = f"figures/logits/L{model.cfg.n_layers - 1:02d}.png"
            logits_b64 = _image_to_base64(logits_path)
            url_dict[node] = f"javascript:showImage('{logits_b64}')"
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

    if node_name == "Input":
        return (0, -y_spacing)

    elif node_name.startswith("A"):
        layer = int(node_name[1:].split(".")[0])
        head = int(node_name.split(".H")[1])
        # Input を中央に配置するため Head を中央からの相対位置として計算
        # ノード幅の間隔を使用
        x = (head - (n_heads - 1) / 2) * x_spacing
        # Attention 層：layer * 2 + 1 の位置
        y = (layer * 2 + 1) * y_spacing
        return (x, y)

    elif node_name.startswith("MLP"):
        layer = int(node_name[3:])
        # MLP 層：layer * 2 + 2 の位置 (各 layer 内で Attention の後)
        y = (layer * 2 + 2) * y_spacing
        return (0, y)

    elif node_name == "Output":
        # 最終層の後に配置 (少し間隔を広めに取る)
        return (0, (n_layers * 2 + 2) * y_spacing)


def _create_node_list(model: HookedTransformer) -> list:
    """
    pygraphviz で描画する際のノードの名前リストを作成する関数.

    Args:
        model (HookedTransformer): Transformer モデルのインスタンス.

    Returns:
        list: ノードの名前リスト (例: ["Input", "A0.H0", "A0.H1", ..., "MLP0", ..., "Output"]).
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    node_list = []

    # Input ノード
    node_list.append("Input")

    # Attention Head と MLP
    for layer in range(n_layers):
        for head in range(n_heads):
            node_list.append(f"A{layer}.H{head}")
        node_list.append(f"MLP{layer}")

    # Output ノード
    node_list.append("Output")

    return node_list


def _create_edge_list(model: HookedTransformer) -> list:
    """
    pygraphviz で描画する際のエッジのリストを作成する関数.

    Args:
        model (HookedTransformer): Transformer モデルのインスタンス.

    Returns:
        list: エッジのリスト (例: [("Input", "A0.H0"), ("A0.H0", "MLP0"), ...]).
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    edge_list = []

    # Input から最初の Attention Head へのエッジ
    for head in range(n_heads):
        edge_list.append(("Input", f"A0.H{head}"))

    # Attention Head から MLP へのエッジ
    for layer in range(n_layers):
        for head in range(n_heads):
            edge_list.append((f"A{layer}.H{head}", f"MLP{layer}"))

    # MLP から次の Attention Head へのエッジ
    for layer in range(n_layers - 1):
        for head in range(n_heads):
            edge_list.append((f"MLP{layer}", f"A{layer + 1}.H{head}"))

    # MLP から output へのエッジ
    edge_list.append((f"MLP{n_layers - 1}", "Output"))

    return edge_list
