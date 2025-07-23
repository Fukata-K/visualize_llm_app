import pygraphviz as pgv
import torch
from transformer_lens import HookedTransformer


def visualize_model(
    model: HookedTransformer,
    base_width: float = 1.5,
    base_height: float = 0.6,
    filename: str = "model_visualization.svg",
):
    """
    モデルのノードとエッジを pygraphviz を使って可視化する関数.

    Args:
        model (HookedTransformer): Transformer モデルのインスタンス.
        filename (str): 出力ファイル名. デフォルトは "model_visualization.svg".
    """
    graph = pgv.AGraph(
        directed=True,
        layout="neato",
        bgcolor="#FFFFFF",
        overlap="true",
        splines="true",
    )

    # ノードを追加
    node_list = create_node_list(model)
    for node in node_list:
        pos = _get_node_position(model, node, base_width, base_height)
        graph.add_node(
            node,
            pos=f"{pos[0]},{pos[1]}!",
        )

    # エッジを追加
    edge_list = create_edge_list(model)
    for edge in edge_list:
        graph.add_edge(edge[0], edge[1])

    # グラフを描画
    graph.layout(prog="neato")
    graph.draw(filename, format="svg")


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


def create_node_list(model: HookedTransformer) -> list:
    """
    pygraphviz で描画する際のノードの名前リストを作成する関数.

    Args:
        model (HookedTransformer): Transformer モデルのインスタンス.

    Returns:
        list: ノードの名前リスト. (例: ["input", "a0.h0", "a0.h1", ..., "m0", ..., "logits"]).
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


def create_edge_list(model: HookedTransformer) -> list:
    """
    pygraphviz で描画する際のエッジのリストを作成する関数.

    Args:
        model (HookedTransformer): Transformer モデルのインスタンス.

    Returns:
        list: エッジのリスト. (例: [("input", "a0.h0"), ("a0.h0", "m0"), ...]).
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


def get_cache(
    model: HookedTransformer,
    prompt: str,
) -> tuple[torch.Tensor, dict]:
    """
    モデルのキャッシュを取得する関数.

    Args:
        model (HookedTransformer): Transformer モデルのインスタンス.
        prompt (str): モデルに入力するプロンプト.

    Returns:
        tuple[torch.Tensor, dict]: モデルの出力とキャッシュ.
    """
    token_ids = model.to_tokens(prompt, prepend_bos=False)
    print(token_ids)
    with torch.no_grad():
        logits, cache = model.run_with_cache(token_ids)
    cache.model = None
    return logits, cache


# モデルの設定
model_name = "gpt2-small"
model = HookedTransformer.from_pretrained(model_name)

visualize_model(
    model, base_width=1.5, base_height=0.6, filename="model_visualization.svg"
)
