from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import FormatStrFormatter
from transformer_lens import ActivationCache, HookedTransformer
from transformer_lens.utils import get_act_name


def generate_all_attention_heatmaps(
    model: HookedTransformer,
    cache: dict | ActivationCache,
    prompt: str,
    output_dir: str = "figures/attention_patterns",
) -> None:
    """
    データセットの全レイヤー・全ヘッドについて平均 Attention Pattern の Heatmap を生成・保存する関数.

    Args:
        model (HookedTransformer): トークナイズ用モデル
        cache (dict | ActivationCache): attention cache
        prompt (str): プロンプト
        output_dir (str): 出力ディレクトリのベースパス (default: "out/attention_patterns")

    Returns:
        None
    """
    # モデル構成の取得
    num_layers = model.cfg.n_layers
    num_heads = model.cfg.n_heads

    # プロンプトのトークン化
    tokens = model.to_str_tokens(prompt, prepend_bos=False)
    tokens = [token.replace(" ", "_") for token in tokens]

    # 各レイヤー・ヘッドについて処理
    for layer in range(num_layers):
        for head in range(num_heads):
            # 保存パスの設定
            save_path = f"{output_dir}/layer{layer:02d}_head{head:02d}.png"
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)

            # タイトルの設定
            title = f"Layer {layer}, Head {head}"

            # Attention Pattern の取得
            attn_pattern = get_attention_pattern(
                cache, data_idx=0, layer=layer, head=head, seq_len=len(tokens)
            )

            # ヒートマップの生成・保存
            create_attention_heatmap(
                attn_pattern=attn_pattern,
                tokens=tokens,
                title=title,
                save_path=save_path,
            )


def visualize_attention_from_cache(
    model: HookedTransformer,
    cache: dict | ActivationCache,
    data_idx: int,
    layer: int,
    head: int,
    prompt: str,
    show: bool = False,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    キャッシュから特定のヘッドの attention pattern を可視化する関数.

    Args:
        model (HookedTransformer): トークナイズ用モデル
        cache (dict | ActivationCache): attention cache
        data_idx (int): バッチ内のデータインデックス
        layer (int): 層番号
        head (int): ヘッド番号
        prompt (str): 元のプロンプト文字列
        show (bool): 図を表示するかどうか (default: False)
        save_path (Optional[str]): 保存先パス (None なら保存しない, default: None)

    Returns:
        plt.Figure: 作成された図
    """
    # トークンの取得
    tokens = model.to_str_tokens(prompt, prepend_bos=False)

    # attention pattern の取得
    attn_pattern = get_attention_pattern(
        cache, data_idx, layer, head, seq_len=len(tokens)
    )

    # タイトルの作成
    title = f"Layer {layer}, Head {head} - Attention Pattern"

    # 可視化
    return create_attention_heatmap(
        attn_pattern=attn_pattern,
        tokens=tokens,
        title=title,
        show=show,
        save_path=save_path,
    )


def get_attention_pattern(
    cache: dict | ActivationCache,
    data_idx: int,
    layer: int,
    head: int,
    seq_len: int = None,
) -> torch.Tensor:
    """
    指定した cache, data_idx, layer, head から対応する Attention Pattern を取得し，
    seq_len が指定されていればそのサイズにトリミングして返す関数.

    Args:
        cache (dict | ActivationCache): transformer_lens の run_with_cache で得られる cache
        data_idx (int): バッチ内のデータインデックス
        layer    (int): 層番号 (0始まり)
        head     (int): ヘッド番号 (0始まり)
        seq_len (int, optional): トリミング後の系列長 (None なら全体を返す, default: None)

    Returns:
        torch.Tensor: [seq_len, seq_len] 形式の Attention Pattern
    """
    key = get_act_name("attn", layer)
    attn = cache[key][data_idx, head]
    if seq_len is not None:
        return attn[:seq_len, :seq_len]
    return attn


def create_attention_heatmap(
    attn_pattern: torch.Tensor,
    tokens: List[str],
    title: str = "Attention Pattern",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    font_size: int = 10,
) -> plt.Figure:
    """
    Attention Pattern から Attention Map の画像を作成する関数.

    Args:
        attn_pattern (torch.Tensor): [seq_len, seq_len] 形式の Attention Pattern
        tokens (List[str]): seq_len と同じ長さのトークン文字列リスト
        title (str): 図のタイトル (default: "Attention Pattern")
        figsize (Tuple[int, int]): 図のサイズ (width, height) (default: (10, 8))
        save_path (Optional[str]): 保存先パス (None なら保存しない, default: None)
        font_size (int): フォントサイズ (default: 10)

    Returns:
        plt.Figure: 作成された matplotlib Figure オブジェクト
    """
    # numpy 配列に変換
    attn_array = attn_pattern.detach().cpu().numpy()

    # 図の作成
    fig, ax = plt.subplots(figsize=figsize)

    # ヒートマップの作成
    vmax = attn_array.max()
    im = ax.imshow(attn_array, cmap="Blues", aspect="equal", vmin=0, vmax=vmax)

    # カラーバーの追加
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Attention Weight", fontsize=font_size)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    # 軸ラベルの設定
    seq_len = attn_pattern.size(0)
    ax.set_xticks(range(seq_len))
    ax.set_yticks(range(seq_len))
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=font_size - 2)
    ax.set_yticklabels(tokens, fontsize=font_size - 2)
    ax.set_xlabel("Key (Attended To)", fontsize=font_size)
    ax.set_ylabel("Query (Attending From)", fontsize=font_size)

    # タイトル
    ax.set_title(title, fontsize=font_size + 2, pad=20)

    # レイアウトの調整
    plt.tight_layout()

    # 保存
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Attention heatmap saved to: {save_path}")

    # 図を閉じる
    plt.close(fig)

    return fig
