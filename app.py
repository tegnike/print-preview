"""
Print Preview - RGB→CMYK 印刷シミュレーションツール
Streamlit GUI
"""

import io
import subprocess
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from color_engine import (
    INTENTS,
    apply_tone_curves,
    auto_adjust_for_print,
    auto_optimize,
    compute_delta_e,
    compute_warning_stats,
    create_warning_overlay,
    export_cmyk,
    load_profiles,
    soft_proof,
)

# カスタムトーンカーブコンポーネント
_tone_curve_component = components.declare_component(
    "tone_curve",
    path=str(Path(__file__).parent / "components" / "tone_curve"),
)

st.set_page_config(
    page_title="Print Preview",
    page_icon="\U0001f3a8",
    layout="wide",
)

st.title("Print Preview - RGB\u2192CMYK \u5370\u5237\u30b7\u30df\u30e5\u30ec\u30fc\u30b7\u30e7\u30f3")

# --- サイドバー ---
with st.sidebar:
    st.header("\u8a2d\u5b9a")

    uploaded_file = st.file_uploader(
        "\u753b\u50cf\u3092\u30a2\u30c3\u30d7\u30ed\u30fc\u30c9",
        type=["png", "jpg", "jpeg", "tiff", "webp", "bmp"],
    )

    profiles = load_profiles()
    if not profiles:
        st.error("CMYK ICC\u30d7\u30ed\u30d5\u30a1\u30a4\u30eb\u304c\u898b\u3064\u304b\u308a\u307e\u305b\u3093")
        st.stop()

    profile_name = st.selectbox(
        "CMYK\u30d7\u30ed\u30d5\u30a1\u30a4\u30eb",
        options=list(profiles.keys()),
    )

    intent_name = st.selectbox(
        "\u30ec\u30f3\u30c0\u30ea\u30f3\u30b0\u30a4\u30f3\u30c6\u30f3\u30c8",
        options=list(INTENTS.keys()),
        index=0,
    )

    st.divider()
    st.subheader("\u8272\u8b66\u544a\u8a2d\u5b9a")
    threshold = st.slider(
        "DeltaE \u95be\u5024",
        min_value=1.0,
        max_value=30.0,
        value=5.0,
        step=0.5,
        help="\u3053\u306e\u5024\u4ee5\u4e0a\u306e\u8272\u5dee\u304c\u3042\u308b\u30d4\u30af\u30bb\u30eb\u3092\u8b66\u544a\u8868\u793a\u3057\u307e\u3059",
    )

# --- メインエリア ---
if uploaded_file is None:
    st.info("\u2b05 \u30b5\u30a4\u30c9\u30d0\u30fc\u304b\u3089\u753b\u50cf\u3092\u30a2\u30c3\u30d7\u30ed\u30fc\u30c9\u3057\u3066\u304f\u3060\u3055\u3044")
    st.markdown(
        """
    ### \u4f7f\u3044\u65b9
    1. \u30b5\u30a4\u30c9\u30d0\u30fc\u3067RGB\u753b\u50cf\u3092\u30a2\u30c3\u30d7\u30ed\u30fc\u30c9
    2. CMYK\u30d7\u30ed\u30d5\u30a1\u30a4\u30eb\u3068\u30ec\u30f3\u30c0\u30ea\u30f3\u30b0\u30a4\u30f3\u30c6\u30f3\u30c8\u3092\u9078\u629e
    3. \u300c\u6bd4\u8f03\u300d\u30bf\u30d6\u3067RGB\u3068CMYK\u30b7\u30df\u30e5\u30ec\u30fc\u30b7\u30e7\u30f3\u3092\u4e26\u3079\u3066\u78ba\u8a8d
    4. \u300c\u8272\u8b66\u544a\u300d\u30bf\u30d6\u3067\u8272\u304c\u5927\u304d\u304f\u5909\u308f\u308b\u7b87\u6240\u3092\u30d2\u30fc\u30c8\u30de\u30c3\u30d7\u3067\u78ba\u8a8d
    5. \u300c\u8272\u8abf\u6574\u300d\u30bf\u30d6\u3067\u81ea\u52d5\u88dc\u6b63\uff0b\u30c8\u30fc\u30f3\u30ab\u30fc\u30d6\u3067\u5fae\u8abf\u6574
    6. \u5fc5\u8981\u306a\u3089CMYK\u753b\u50cf\u3092\u30c0\u30a6\u30f3\u30ed\u30fc\u30c9
    """
    )
    st.stop()

# 画像読み込み
original = Image.open(uploaded_file)
original_rgb = original.convert("RGB")
cmyk_profile_path = profiles[profile_name]

# プレビュー用リサイズ（大画像対策）
MAX_PREVIEW_SIZE = 2000
w, h = original_rgb.size
if max(w, h) > MAX_PREVIEW_SIZE:
    ratio = MAX_PREVIEW_SIZE / max(w, h)
    preview_size = (int(w * ratio), int(h * ratio))
    preview_rgb = original_rgb.resize(preview_size, Image.LANCZOS)
else:
    preview_rgb = original_rgb

# ソフトプルーフ実行
with st.spinner("\u30bd\u30d5\u30c8\u30d7\u30eb\u30fc\u30d5\u3092\u8a08\u7b97\u4e2d..."):
    simulated = soft_proof(preview_rgb, cmyk_profile_path, intent_name)

# タブ
tab_compare, tab_warning, tab_adjust, tab_export = st.tabs([
    "\U0001f50d \u6bd4\u8f03", "\u26a0\ufe0f \u8272\u8b66\u544a", "\U0001f527 \u8272\u8abf\u6574", "\U0001f4e5 \u30a8\u30af\u30b9\u30dd\u30fc\u30c8"
])

with tab_compare:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original (RGB)")
        st.image(preview_rgb, use_container_width=True)
    with col2:
        st.subheader(f"CMYK Simulation ({profile_name})")
        st.image(simulated, use_container_width=True)

    st.caption(
        f"\u30d7\u30ed\u30d5\u30a1\u30a4\u30eb: {profile_name} / "
        f"\u30a4\u30f3\u30c6\u30f3\u30c8: {intent_name} / "
        f"\u30b5\u30a4\u30ba: {original_rgb.size[0]}x{original_rgb.size[1]}px"
    )

with tab_warning:
    with st.spinner("\u8272\u5dee\u3092\u8a08\u7b97\u4e2d..."):
        delta_e = compute_delta_e(preview_rgb, simulated)
        stats = compute_warning_stats(delta_e, threshold)

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    col_s1.metric("\u5e73\u5747 DeltaE", f"{stats['mean_delta_e']:.1f}")
    col_s2.metric("\u6700\u5927 DeltaE", f"{stats['max_delta_e']:.1f}")
    col_s3.metric("\u6ce8\u610f\u30d4\u30af\u30bb\u30eb", f"{stats['warning_percent']:.1f}%")
    col_s4.metric("\u8b66\u544a\u30d4\u30af\u30bb\u30eb", f"{stats['severe_percent']:.1f}%")

    overlay = create_warning_overlay(preview_rgb, delta_e, threshold)

    col_w1, col_w2 = st.columns(2)
    with col_w1:
        st.subheader("CMYK\u30b7\u30df\u30e5\u30ec\u30fc\u30b7\u30e7\u30f3")
        st.image(simulated, use_container_width=True)
    with col_w2:
        st.subheader("\u8272\u8b66\u544a\u30d2\u30fc\u30c8\u30de\u30c3\u30d7")
        st.image(overlay, use_container_width=True)

    st.markdown(
        f"""
    **\u51e1\u4f8b**: \U0001f7e8 \u9ec4\u8272 = \u6ce8\u610f (DeltaE {threshold:.1f}\uff5e{threshold*2:.1f}) / \U0001f7e5 \u8d64 = \u8b66\u544a (DeltaE {threshold*2:.1f}\u4ee5\u4e0a)

    > DeltaE 3\u4ee5\u4e0b: \u307b\u307c\u6c17\u3065\u304b\u306a\u3044\u5dee / 3\uff5e5: \u6ce8\u610f\u6df1\u304f\u898b\u308b\u3068\u308f\u304b\u308b / 5\u4ee5\u4e0a: \u660e\u3089\u304b\u306b\u9055\u3046
    """
    )

with tab_adjust:

    @st.fragment
    def adjust_fragment():
        st.subheader("\u8272\u8abf\u6574")

        # --- 自動最適化ボタン ---
        if st.button(
            "\u2728 \u81ea\u52d5\u6700\u9069\u5316",
            type="primary",
            help="\u88dc\u6b63\u5f37\u5ea6\u3068\u30c8\u30fc\u30f3\u30ab\u30fc\u30d6\u3092\u81ea\u52d5\u3067\u6700\u9069\u5024\u306b\u8a2d\u5b9a\u3057\u307e\u3059\u3002\u5c11\u3057\u6642\u9593\u304c\u304b\u304b\u308a\u307e\u3059\u3002",
        ):
            progress_bar = st.progress(0, text="\u6700\u9069\u5316\u4e2d...\u88dc\u6b63\u5f37\u5ea6\u3092\u63a2\u7d22")

            def on_progress(pct):
                if pct < 0.15:
                    text = "\u6700\u9069\u5316\u4e2d...\u88dc\u6b63\u5f37\u5ea6\u3092\u63a2\u7d22"
                elif pct < 0.9:
                    text = "\u6700\u9069\u5316\u4e2d...\u30c8\u30fc\u30f3\u30ab\u30fc\u30d6\u3092\u8abf\u6574"
                else:
                    text = "\u6700\u9069\u5316\u4e2d...\u3082\u3046\u3059\u3050\u5b8c\u4e86"
                progress_bar.progress(min(pct, 1.0), text=text)

            opt_strength, opt_points = auto_optimize(
                preview_rgb, cmyk_profile_path, intent_name,
                progress_callback=on_progress,
            )
            st.session_state["opt_strength"] = opt_strength
            st.session_state["opt_points"] = opt_points
            # コンポーネント再作成のためバージョンを更新
            st.session_state["opt_version"] = st.session_state.get("opt_version", 0) + 1
            progress_bar.empty()
            st.rerun(scope="fragment")

        # 最適化結果がある場合はそれを初期値に使う
        default_strength = st.session_state.get("opt_strength", 0.7)
        default_points = st.session_state.get("opt_points", None)
        opt_version = st.session_state.get("opt_version", 0)

        # 最適化結果のサマリー表示
        if default_points:
            identity = [(0, 0), (64, 64), (128, 128), (192, 192), (255, 255)]
            opt_channel_points = {
                ch: [(p[0], p[1]) for p in default_points[ch]]
                for ch in ["r", "g", "b"]
            }
            has_curve_changes = any(
                opt_channel_points[ch] != identity for ch in ["r", "g", "b"]
            )
            summary_parts = [f"\u88dc\u6b63\u5f37\u5ea6: **{default_strength}**"]
            if has_curve_changes:
                for ch, label in [("r", "R"), ("g", "G"), ("b", "B")]:
                    pts = opt_channel_points[ch]
                    changes = []
                    for i, (inp, out) in enumerate(pts):
                        if inp != out:
                            names = ["Black", "Shadow", "Mid", "Highlight", "White"]
                            changes.append(f"{names[i]} {inp}\u2192{out}")
                    if changes:
                        summary_parts.append(f"{label}: {', '.join(changes)}")
            else:
                summary_parts.append("\u30c8\u30fc\u30f3\u30ab\u30fc\u30d6: \u5909\u66f4\u306a\u3057\uff08\u30ea\u30cb\u30a2\u304c\u6700\u9069\uff09")
            st.info("\u2728 \u6700\u9069\u5316\u7d50\u679c: " + " / ".join(summary_parts))

        st.markdown(
            "**Step 1**: \u30c8\u30fc\u30f3\u30ab\u30fc\u30d6\u3067\u8272\u57df\u88dc\u6b63 \u2192 "
            "**Step 2**: \u81ea\u52d5\u88dc\u6b63\u3067\u30d4\u30af\u30bb\u30eb\u5358\u4f4d\u306e\u5fae\u8abf\u6574"
        )

        # --- Step 1: トーンカーブ ---
        st.markdown("### Step 1: \u30c8\u30fc\u30f3\u30ab\u30fc\u30d6")
        st.caption("\u30dd\u30a4\u30f3\u30c8\u3092\u30c9\u30e9\u30c3\u30b0\u3057\u3066R/G/B\u5404\u30c1\u30e3\u30f3\u30cd\u30eb\u3092\u8abf\u6574\u3001\u307e\u305f\u306f\u300c\u81ea\u52d5\u6700\u9069\u5316\u300d\u3067\u81ea\u52d5\u8a2d\u5b9a")

        # カスタムインタラクティブトーンカーブ
        if default_points:
            init_points = {
                ch: [list(p) for p in default_points[ch]]
                for ch in ["r", "g", "b"]
            }
        else:
            init_points = {
                "r": [[0, 0], [64, 64], [128, 128], [192, 192], [255, 255]],
                "g": [[0, 0], [64, 64], [128, 128], [192, 192], [255, 255]],
                "b": [[0, 0], [64, 64], [128, 128], [192, 192], [255, 255]],
            }

        # opt_versionをkeyに含めて最適化後にコンポーネントを再作成
        curve_result = _tone_curve_component(
            channel_points=init_points,
            key=f"tone_curve_editor_{opt_version}",
            default=init_points,
        )

        # コンポーネントの結果をタプルリストに変換
        if curve_result and isinstance(curve_result, dict):
            channel_points = {
                ch: [(p[0], p[1]) for p in curve_result[ch]]
                for ch in ["r", "g", "b"]
            }
        else:
            channel_points = {
                ch: [(p[0], p[1]) for p in init_points[ch]]
                for ch in ["r", "g", "b"]
            }

        # カーブが変更されているか判定
        identity = [(0, 0), (64, 64), (128, 128), (192, 192), (255, 255)]
        curve_is_identity = all(
            channel_points[ch] == identity
            for ch in ["r", "g", "b"]
        )

        # トーンカーブを元画像に適用
        if curve_is_identity:
            curved_rgb = preview_rgb
        else:
            curved_rgb = apply_tone_curves(
                preview_rgb,
                r_points=channel_points["r"],
                g_points=channel_points["g"],
                b_points=channel_points["b"],
            )

        # --- Step 2: 自動補正 ---
        st.markdown("### Step 2: \u81ea\u52d5\u88dc\u6b63")
        adjust_strength = st.slider(
            "\u88dc\u6b63\u306e\u5f37\u3055",
            min_value=0.0,
            max_value=1.0,
            value=default_strength,
            step=0.1,
            help="CMYK\u5909\u63db\u6642\u306e\u8272\u30ba\u30ec\u3092\u9006\u65b9\u5411\u306b\u88dc\u6b63\u3002\u8d64\u30fb\u7dd1\u30fb\u30aa\u30ec\u30f3\u30b8\u7cfb\u306b\u52b9\u679c\u5927\u3002",
        )

        with st.spinner("\u81ea\u52d5\u88dc\u6b63\u4e2d..."):
            final_rgb, _, _ = auto_adjust_for_print(
                curved_rgb, cmyk_profile_path, intent_name, adjust_strength
            )

        # 最終結果のシミュレーション
        final_sim = soft_proof(final_rgb, cmyk_profile_path, intent_name)
        final_de = compute_delta_e(preview_rgb, final_sim)  # オリジナルとの差
        final_stats = compute_warning_stats(final_de, threshold)

        orig_de = compute_delta_e(preview_rgb, simulated)
        orig_stats = compute_warning_stats(orig_de, threshold)

        # --- 統計比較 ---
        st.markdown("### \u8abf\u6574\u7d50\u679c")
        col_b, col_a = st.columns(2)
        with col_b:
            st.markdown("**\u8abf\u6574\u524d**")
            st.metric("\u5e73\u5747 DeltaE", f"{orig_stats['mean_delta_e']:.1f}")
            st.metric("\u8b66\u544a\u30d4\u30af\u30bb\u30eb", f"{orig_stats['warning_percent']:.1f}%")
        with col_a:
            label = "**\u30ab\u30fc\u30d6 + \u81ea\u52d5\u88dc\u6b63**" if not curve_is_identity else "**\u81ea\u52d5\u88dc\u6b63\u5f8c**"
            st.markdown(label)
            st.metric(
                "\u5e73\u5747 DeltaE",
                f"{final_stats['mean_delta_e']:.1f}",
                delta=f"{final_stats['mean_delta_e'] - orig_stats['mean_delta_e']:.1f}",
                delta_color="inverse",
            )
            st.metric(
                "\u8b66\u544a\u30d4\u30af\u30bb\u30eb",
                f"{final_stats['warning_percent']:.1f}%",
                delta=f"{final_stats['warning_percent'] - orig_stats['warning_percent']:.1f}%",
                delta_color="inverse",
            )

        # --- 画像比較 ---
        st.markdown("### \u753b\u50cf\u6bd4\u8f03")
        col_img1, col_img2, col_img3 = st.columns(3)
        with col_img1:
            st.caption("Original (RGB)")
            st.image(preview_rgb, use_container_width=True)
        with col_img2:
            st.caption("\u8abf\u6574\u6e08\u307f (RGB)")
            st.image(final_rgb, use_container_width=True)
        with col_img3:
            st.caption("\u5370\u5237\u4e88\u60f3 (CMYK Sim)")
            st.image(final_sim, use_container_width=True)

        st.markdown(
            "> \u5de6\u306e\u300c\u30aa\u30ea\u30b8\u30ca\u30eb\u300d\u3068\u53f3\u306e\u300c\u5370\u5237\u4e88\u60f3\u300d\u304c\u8fd1\u3065\u304f\u3088\u3046\u306b\u8abf\u6574\u3057\u3066\u304f\u3060\u3055\u3044\u3002"
        )

        # --- ダウンロード ---
        base_name = uploaded_file.name.rsplit(".", 1)[0]

        if st.button("\U0001f504 \u30aa\u30ea\u30b8\u30ca\u30eb\u30b5\u30a4\u30ba\u3067\u66f8\u304d\u51fa\u3057", type="secondary"):
            with st.spinner(
                f"\u30aa\u30ea\u30b8\u30ca\u30eb\u30b5\u30a4\u30ba ({original_rgb.size[0]}x{original_rgb.size[1]}) \u3067\u51e6\u7406\u4e2d..."
            ):
                # パイプライン: トーンカーブ → 自動補正
                if not curve_is_identity:
                    full_curved = apply_tone_curves(
                        original_rgb,
                        r_points=channel_points["r"],
                        g_points=channel_points["g"],
                        b_points=channel_points["b"],
                    )
                else:
                    full_curved = original_rgb
                full_adjusted, _, _ = auto_adjust_for_print(
                    full_curved, cmyk_profile_path, intent_name, adjust_strength
                )
                buf = io.BytesIO()
                full_adjusted.save(buf, format="PNG")
                st.session_state["adjusted_data"] = buf.getvalue()
                st.session_state["adjusted_filename"] = f"{base_name}_adjusted.png"
            st.success("\u5b8c\u4e86\uff01\u4e0b\u306e\u30dc\u30bf\u30f3\u304b\u3089\u30c0\u30a6\u30f3\u30ed\u30fc\u30c9\u3067\u304d\u307e\u3059\u3002")
            st.rerun(scope="fragment")

        if "adjusted_data" in st.session_state:
            st.download_button(
                label=f"\u2b07\ufe0f {st.session_state.get('adjusted_filename', base_name + '_adjusted.png')} \u3092\u30c0\u30a6\u30f3\u30ed\u30fc\u30c9",
                data=st.session_state["adjusted_data"],
                file_name=st.session_state.get("adjusted_filename", f"{base_name}_adjusted.png"),
                mime="image/png",
                type="primary",
            )

    adjust_fragment()

with tab_export:
    st.subheader("CMYK\u753b\u50cf\u30c0\u30a6\u30f3\u30ed\u30fc\u30c9")
    st.markdown(
        f"**\u30d7\u30ed\u30d5\u30a1\u30a4\u30eb**: {profile_name} / **\u30a4\u30f3\u30c6\u30f3\u30c8**: {intent_name}"
    )

    export_format = st.radio(
        "\u51fa\u529b\u5f62\u5f0f",
        options=["tiff", "jpg"],
        horizontal=True,
        help="TIFF: \u7121\u52a3\u5316\u5727\u7e2e\u3001\u5370\u5237\u5165\u7a3f\u5411\u3051 / JPG: \u30d5\u30a1\u30a4\u30eb\u30b5\u30a4\u30ba\u5c0f",
    )

    base_name = uploaded_file.name.rsplit(".", 1)[0]
    ext = "tif" if export_format == "tiff" else "jpg"
    output_filename = f"{base_name}_CMYK.{ext}"

    with st.spinner("CMYK\u5909\u63db\u4e2d..."):
        try:
            cmyk_data = export_cmyk(
                original_rgb,
                cmyk_profile_path,
                intent_name,
                export_format,
            )
            st.success(
                f"\u5909\u63db\u5b8c\u4e86\uff01\u30d5\u30a1\u30a4\u30eb\u30b5\u30a4\u30ba: {len(cmyk_data) / 1024:.0f} KB"
            )
            st.download_button(
                label=f"\u2b07\ufe0f {output_filename} \u3092\u30c0\u30a6\u30f3\u30ed\u30fc\u30c9",
                data=cmyk_data,
                file_name=output_filename,
                mime=f"image/{export_format}",
                type="primary",
            )
        except FileNotFoundError:
            st.error(
                "ImageMagick\u304c\u898b\u3064\u304b\u308a\u307e\u305b\u3093\u3002`brew install imagemagick` \u3067\u30a4\u30f3\u30b9\u30c8\u30fc\u30eb\u3057\u3066\u304f\u3060\u3055\u3044\u3002"
            )
        except subprocess.CalledProcessError as e:
            st.error(f"CMYK\u5909\u63db\u306b\u5931\u6557\u3057\u307e\u3057\u305f: {e}")
