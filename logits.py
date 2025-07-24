from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FormatStrFormatter
from transformer_lens import HookedTransformer

# グローバル変数でフォント設定の状態を管理
_font_configured = False


def _setup_japanese_font():
    """
    日本語表示用のフォントを設定する関数.

    macOS で利用可能な日本語フォントを優先順位順に試行し、
    最初に見つかったフォントを matplotlib のデフォルトとして設定する。
    """
    global _font_configured

    # 既に設定済みの場合はスキップ
    if _font_configured:
        return

    import matplotlib.font_manager as fm

    # macOS で一般的に利用可能な日本語フォントのリスト (優先順位順)
    japanese_fonts = [
        "Hiragino Sans",  # macOS標準
        "Hiragino Kaku Gothic Pro",  # macOS標準
        "Arial Unicode MS",  # macOS/Office
        "Yu Gothic",  # Windows/macOS
        "YuGothic",  # macOSでの表記
        "Meiryo",  # Windows
        "Takao Gothic",  # Linux
        "IPAexGothic",  # Linux
        "DejaVu Sans",  # フォールバック
    ]

    # 利用可能なフォントファミリーのリストを取得
    available_fonts = [f.name for f in fm.fontManager.ttflist]

    # 日本語フォントの中から利用可能なものを探す
    selected_font = None
    for font_name in japanese_fonts:
        if font_name in available_fonts:
            selected_font = font_name
            break

    # フォントが見つかった場合は設定
    if selected_font:
        # matplotlib の全フォント関連パラメータを統一的に設定
        plt.rcParams["font.family"] = [selected_font]
        plt.rcParams["font.sans-serif"] = [selected_font]
        plt.rcParams["axes.unicode_minus"] = False  # マイナス記号の表示問題を回避
        plt.rcParams["text.usetex"] = False  # LaTeX解釈を無効化

        # フォント警告を抑制
        import warnings

        warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

        print(f"日本語フォントを設定しました: {selected_font}")
    else:
        print("日本語フォントが見つかりませんでした。デフォルトフォントを使用します。")
        # デフォルトフォントでも警告を減らすため、フォールバックを設定
        plt.rcParams["font.family"] = ["DejaVu Sans", "sans-serif"]
        plt.rcParams["axes.unicode_minus"] = False
        plt.rcParams["text.usetex"] = False

        # フォント警告を抑制
        import warnings

        warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

    _font_configured = True


def _escape_latex_chars(text: str) -> str:
    """
    matplotlib で LaTeX 特殊文字として解釈される文字を安全な表示用に変換する関数.

    Args:
        text (str): 変換する文字列

    Returns:
        str: 安全に表示できる文字列
    """
    # 表示できない文字や問題のある文字を事前にフィルタリング
    # 置換文字や制御文字を除去
    filtered_chars = []
    for char in text:
        char_code = ord(char)
        # 基本的なASCII文字、日本語（ひらがな、カタカナ、漢字）、基本的なラテン文字を保持
        if (
            (0x20 <= char_code <= 0x7E)  # 基本ASCII
            or (0x3040 <= char_code <= 0x309F)  # ひらがな
            or (0x30A0 <= char_code <= 0x30FF)  # カタカナ
            or (0x4E00 <= char_code <= 0x9FAF)  # CJK統合漢字
            or (0xFF00 <= char_code <= 0xFFEF)  # 全角英数字
        ):
            filtered_chars.append(char)
        else:
            # 表示できない文字は簡単な代替表現に置換
            if 0x0590 <= char_code <= 0x05FF:  # ヘブライ語
                filtered_chars.append("[HE]")
            elif 0x0600 <= char_code <= 0x06FF:  # アラビア語
                filtered_chars.append("[AR]")
            elif 0x0370 <= char_code <= 0x03FF:  # ギリシャ語
                filtered_chars.append("[GR]")
            elif char_code == 0xFFFD:  # 置換文字
                filtered_chars.append("[?]")
            else:
                filtered_chars.append("[?]")

    text = "".join(filtered_chars)

    # 日本語文字が含まれている場合は、LaTeX特殊文字の置換を最小限にする
    # （日本語フォントが適切に設定されていれば問題ないため）
    if any(0x3040 <= ord(char) <= 0x9FAF for char in text):  # 日本語文字が含まれる場合
        # 問題を引き起こす可能性の高い文字のみ置換
        minimal_replacements = {
            "$": "USD",  # ドル記号のみ
        }
        for char, replacement in minimal_replacements.items():
            if char in text:
                text = text.replace(char, replacement)
        return text

    # ASCII文字のみの場合は、通常の置換を実行
    replacements = {
        "$": "USD",  # より短い表現
        "_": "-",  # アンダーバーをハイフンに
        "^": "EXP",  # 指数記号
        "{": "(",  # 波括弧を丸括弧に
        "}": ")",  # 波括弧を丸括弧に
        "\\": "/",  # バックスラッシュをスラッシュに
        "%": "PCT",  # パーセント
        "&": "AND",  # アンパサンド
        "#": "NUM",  # ハッシュ
        "~": "-",  # チルダをハイフンに
    }

    for char, replacement in replacements.items():
        if char in text:
            text = text.replace(char, replacement)

    return text


def save_all_logits_figures(
    model: HookedTransformer,
    cache: Dict[str, torch.Tensor],
) -> None:
    """
    モデルの全ての層の logits を可視化し、画像として保存する関数.

    Args:
        model (HookedTransformer): 分析対象の HookedTransformer モデル
        cache (Dict[str, torch.Tensor]): モデル実行時のアクティベーションキャッシュ

    Returns:
        None
    """
    layer_logits, head_logits = _compute_all_components_logits(model, cache)

    for layer_idx in range(model.cfg.n_layers):
        print(f"Processing Layer {layer_idx}/{model.cfg.n_layers - 1}...")
        for head_idx in range(model.cfg.n_heads):
            _visualize_top_k_tokens(
                head_logits[layer_idx, head_idx],
                model,
                top_k=10,
                title=f"Layer {layer_idx} Head {head_idx} Logits",
                save_path=f"figures/logits/L{layer_idx:02d}_H{head_idx:02d}.png",
            )
        _visualize_top_k_tokens(
            layer_logits[layer_idx],
            model,
            top_k=10,
            title=f"Layer {layer_idx} Logits",
            save_path=f"figures/logits/L{layer_idx:02d}.png",
        )


def _visualize_top_k_tokens(
    logits: torch.Tensor,
    model: HookedTransformer,
    top_k: int = 10,
    title: str = "Top-K Token Logits",
    figsize: Tuple[int, int] = (4, 6),
    cmap: str = "Blues",
    save_path: Optional[str] = None,
    font_size: int = 10,
) -> None:
    """
    単一の logits から上位 K 個のトークンを 1次元 Heatmap で可視化する関数.

    Args:
        logits (torch.Tensor): 可視化する logits [d_vocab]
        model (HookedTransformer): トークナイザーを含むモデル
        top_k (int): 表示する上位トークン数. Defaults to 10.
        title (str): グラフのタイトル. Defaults to "Top-K Token Logits".
        figsize (Tuple[int, int]): 図のサイズ (width, height). Defaults to (4, 6).
        cmap (str): カラーマップ名. Defaults to "Blues".
        save_path (Optional[str]): 保存パス. Defaults to None.
        font_size (int): フォントサイズ. Defaults to 10.

    Returns:
        None
    """
    # 日本語フォントを設定
    _setup_japanese_font()
    # logits を numpy 配列に変換
    if isinstance(logits, torch.Tensor):
        logits_np = logits.detach().cpu().numpy()
    else:
        logits_np = np.array(logits)

    # 1次元でない場合はエラー
    if logits_np.ndim != 1:
        raise ValueError(f"Expected 1D logits, got {logits_np.ndim}D")

    # 上位 K 個のトークンインデックスを取得
    top_k_indices = np.argsort(logits_np)[-top_k:][::-1]  # 降順
    top_k_values = logits_np[top_k_indices]

    # 対応するトークンテキストを取得
    token_texts = []
    for idx in top_k_indices:
        try:
            token_text = model.tokenizer.decode([idx])
            # matplotlib で LaTeX として解釈される特殊文字をエスケープ
            token_text = _escape_latex_chars(token_text)

            # 特殊文字や空白を見やすく表示し、長さを統一
            if token_text.strip() == "":
                token_text = "[SPACE]"
            elif len(token_text) > 10:  # 長いトークンは一律10文字に短縮
                token_text = token_text[:10] + "..."

            token_texts.append(token_text)
        except Exception:
            token_texts.append(f"[UNK_{idx}]")

    # 1次元 Heatmap 用のデータ準備 [top_k, 1] の形に変形
    heatmap_data = top_k_values.reshape(-1, 1)

    # 図の作成
    fig, ax = plt.subplots(figsize=figsize)

    # カラーマップの範囲を設定
    vmax = max(heatmap_data)
    vmin = min(heatmap_data)

    # Heatmap の作成
    im = ax.imshow(heatmap_data, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

    # カラーバーの追加
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Logit Value", fontsize=font_size)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    # 軸ラベルの設定
    ax.set_yticks([])  # y軸のティックを非表示
    ax.set_xticks([])  # x軸は不要

    # 動的フォントサイズの計算
    def calculate_dynamic_font_size(figsize, top_k, base_font_size):
        """
        画像サイズとトークン数に基づいて最適なフォントサイズを計算する関数.

        トークン長制限（10文字 + "..."）を考慮して一貫したフォントサイズを計算し、
        同じ引数であれば常に同じサイズの画像を生成する。

        Args:
            figsize: 図のサイズ (width, height)
            top_k: 表示するトークン数
            base_font_size: ベースとなるフォントサイズ

        Returns:
            int: 計算された最適なフォントサイズ
        """
        _, height = figsize

        # セルあたりの高さ (インチ単位)
        cell_height_inch = height / top_k

        # セルの高さをポイント単位に変換 (1インチ = 72ポイント)
        cell_height_points = cell_height_inch * 72

        # 制限後の最大トークン長を固定値として使用（10文字 + "..." = 13文字）
        # ただし、"[SPACE]"のような特殊表示も考慮して少し余裕を持たせる
        max_token_length = 13

        # フォントサイズの計算
        # セルの高さの 60% を目安とし, 固定の最大トークン長に応じて調整
        dynamic_size = cell_height_points * 0.6

        # 固定の最大トークン長に基づいてフォントサイズを調整
        if max_token_length > 15:
            dynamic_size *= 0.6
        elif max_token_length > 10:
            dynamic_size *= 0.8

        # 最小・最大フォントサイズの制限
        min_font_size = 6
        max_font_size = min(base_font_size + 4, 20)

        calculated_size = max(min_font_size, min(max_font_size, dynamic_size))

        return int(calculated_size)

    # 動的フォントサイズを計算（トークンテキストのリストは不要）
    dynamic_font_size = calculate_dynamic_font_size(figsize, top_k, font_size)

    # Heatmap 上にトークンを表示
    for i, (value, token_text) in enumerate(zip(top_k_values, token_texts)):
        # 正規化された値を計算 (0-1 の範囲)
        normalized_value = (value - vmin) / (vmax - vmin) if vmax != vmin else 0.5

        # 背景色とのコントラストを考慮してテキスト色を決定
        text_color = "white" if normalized_value > 0.5 else "black"

        ax.text(
            0,
            i,
            token_text,
            ha="center",
            va="center",
            color=text_color,
            fontsize=dynamic_font_size,
            fontweight="bold",
        )

    # タイトル
    ax.set_title(title, fontsize=font_size + 2, pad=20)

    # レイアウトの調整
    plt.tight_layout()

    # 保存
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)  # 明示的な figure close


def _compute_all_components_logits(
    model: HookedTransformer,
    cache: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    モデルの全てのコンポーネントの logits を計算する関数.

    Args:
        model (HookedTransformer): 分析対象の HookedTransformer モデル
        cache (Dict[str, torch.Tensor]): モデル実行時のアクティベーションキャッシュ

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 各層の logits とキャッシュ
    """
    # 各層の出力を logit へ変換
    logits = _compute_layer_logit_contribution(model, cache)

    # 各 Head の logits を計算
    head_logits = _compute_heads_logits_contribution(model, cache)

    return logits, head_logits


def _compute_layer_logit_contribution(
    model: HookedTransformer,
    cache: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    各 layer の出力を logit へ変換する関数.

    Note: この関数は各層の出力に Final Layer Normalization を適用してから Unembedding を行う.
    これにより実際の GPT-2 モデルの処理フローと一致する正確な寄与度が計算される.

    Args:
        model (HookedTransformer): 分析対象の HookedTransformer モデル
        cache (Dict[str, torch.Tensor]): モデル実行時のアクティベーションキャッシュ

    Returns:
        torch.Tensor: [n_layers, d_vocab]
    """
    with torch.no_grad():
        # Unembedding 行列とバイアス項を取得
        W_U = model.W_U  # [d_model, d_vocab]
        b_U = getattr(model, "b_U", None)  # [d_vocab] (存在しない場合は None)

        n_layers = model.cfg.n_layers

        # 全ての層の出力を取得してスタック
        layer_out_list = []
        for layer_idx in range(n_layers):
            cache_key = f"blocks.{layer_idx}.hook_resid_post"
            layer_out = cache[cache_key][0, -1]
            layer_out_list.append(layer_out)

        # layer_out を [n_layers, d_model] の形にスタック
        layer_out = torch.stack(layer_out_list, dim=0)

        # 各出力に Final Layer Normalization を適用
        # バッチ処理のために次元を調整: [n_layers, d_model] -> [n_layers, 1, d_model]
        layer_out_batched = layer_out.unsqueeze(1)
        normalized_layer_out_list = []
        for layer_idx in range(n_layers):
            normalized = model.ln_final(layer_out_batched[layer_idx]).squeeze(0)
            normalized_layer_out_list.append(normalized)
        normalized_layer_out = torch.stack(normalized_layer_out_list, dim=0)

        # einsum を使用して効率的に計算: normalized_layer_out @ W_U
        # normalized_layer_out: [n_layers, d_model], W_U: [d_model, d_vocab]
        # 結果: [n_layers, d_vocab]
        logits = torch.einsum("lm,mv->lv", normalized_layer_out, W_U)

        # バイアス項が存在する場合は各層に追加
        if b_U is not None:
            logits = logits + b_U.unsqueeze(0)

        return logits


def _compute_heads_logits_contribution(
    model: HookedTransformer,
    cache: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    モデル内の全ての Head の logits を計算する関数.

    計算の流れ:
    1. モデル内の全ての層の Head の Attention 出力 z を取得してスタック
    2. 全ての Head に出力重み W_O を一度に適用
    3. Unembedding 行列 W_U を適用して logits を計算

    Args:
        model (HookedTransformer): 分析対象の HookedTransformer モデル
        cache (Dict[str, torch.Tensor]): モデル実行時のアクティベーションキャッシュ

    Returns:
        torch.Tensor: 各 Head の logits tensor [n_layers, n_heads, d_vocab]
    """
    with torch.no_grad():
        n_layers = model.cfg.n_layers

        # 全ての層の Head の Attention 出力を取得してスタック
        z_list = []
        for layer_idx in range(n_layers):
            cache_key = f"blocks.{layer_idx}.attn.hook_z"
            z = cache[cache_key][0, -1]
            z_list.append(z)

        # z を [n_layers, n_heads, d_head] の形にスタック
        z = torch.stack(z_list, dim=0)

        # 出力重み行列 ([n_layers, n_heads, d_head, d_model]) を取得
        W_O = model.W_O

        # Unembedding 行列とバイアス項を取得
        W_U = model.W_U  # [d_model, d_vocab]
        b_U = getattr(model, "b_U", None)  # [d_vocab] (存在しない場合は None)

        # 各 Head の出力を残差ストリーム次元に変換: z @ W_O
        # z: [n_layers, n_heads, d_head], W_O: [n_layers, n_heads, d_head, d_model]
        # 結果: [n_layers, n_heads, d_model]
        head_outputs = torch.einsum("lhd,lhdm->lhm", z, W_O)

        # 各 Head 出力に Final Layer Normalization を適用
        n_layers, n_heads, _ = head_outputs.shape
        normalized_head_outputs = []
        for layer_idx in range(n_layers):
            layer_normalized = []
            for head_idx in range(n_heads):
                normalized = model.ln_final(
                    head_outputs[layer_idx, head_idx].unsqueeze(0)
                ).squeeze(0)
                layer_normalized.append(normalized)
            normalized_head_outputs.append(torch.stack(layer_normalized, dim=0))
        normalized_head_outputs = torch.stack(normalized_head_outputs, dim=0)

        # einsum を使用して効率的に計算: normalized_head_outputs @ W_U
        # normalized_head_outputs: [n_layers, n_heads, d_model], W_U: [d_model, d_vocab]
        # 結果: [n_layers, n_heads, d_vocab]
        logits = torch.einsum("lhm,mv->lhv", normalized_head_outputs, W_U)

        # バイアス項が存在する場合は各層・各ヘッドに追加
        if b_U is not None:
            logits = logits + b_U.unsqueeze(0).unsqueeze(0)

        return logits
