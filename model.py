from transformer_lens import HookedTransformer


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

    # Input nodes
    node_list.append("input")

    # Attention Heads and MLP
    for layer in range(n_layers):
        for head in range(n_heads):
            node_list.append(f"a{layer}.h{head}")
        node_list.append(f"m{layer}")

    # Output nodes
    node_list.append("logits")

    return node_list
