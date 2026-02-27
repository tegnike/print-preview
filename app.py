"""
Print Preview - RGB→CMYK 印刷シミュレーションツール
Streamlit GUI
"""

import subprocess

import streamlit as st
from PIL import Image

from color_engine import (
    INTENTS,
    auto_adjust_for_print,
    compute_delta_e,
    compute_warning_stats,
    create_warning_overlay,
    export_cmyk,
    load_profiles,
    soft_proof,
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
    5. \u5fc5\u8981\u306a\u3089CMYK\u753b\u50cf\u3092\u30c0\u30a6\u30f3\u30ed\u30fc\u30c9
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

    # 統計表示
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    col_s1.metric("\u5e73\u5747 DeltaE", f"{stats['mean_delta_e']:.1f}")
    col_s2.metric("\u6700\u5927 DeltaE", f"{stats['max_delta_e']:.1f}")
    col_s3.metric("\u6ce8\u610f\u30d4\u30af\u30bb\u30eb", f"{stats['warning_percent']:.1f}%")
    col_s4.metric("\u8b66\u544a\u30d4\u30af\u30bb\u30eb", f"{stats['severe_percent']:.1f}%")

    # ヒートマップオーバーレイ
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
    st.subheader("\u8272\u57df\u5916\u30ab\u30e9\u30fc\u306e\u81ea\u52d5\u8abf\u6574")
    st.markdown(
        "CMYK\u3067\u518d\u73fe\u3067\u304d\u306a\u3044\u9bae\u3084\u304b\u306a\u8272\u306e**\u5f69\u5ea6\u3092\u81ea\u52d5\u3067\u4e0b\u3052\u3066**\u3001"
        "\u5370\u5237\u3067\u8272\u304c\u5d29\u308c\u306b\u304f\u3044RGB\u753b\u50cf\u3092\u4f5c\u6210\u3057\u307e\u3059\u3002"
    )

    adjust_strength = st.slider(
        "\u8abf\u6574\u306e\u5f37\u3055",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="0.0=\u5909\u66f4\u306a\u3057\u3001 1.0=\u6700\u5927\u9650\u8abf\u6574\u3002\u5f37\u3044\u307b\u3069\u8272\u57df\u5185\u306b\u53ce\u307e\u308b\u304c\u3001\u5f69\u5ea6\u304c\u843d\u3061\u308b\u3002",
    )

    with st.spinner("\u8272\u8abf\u6574\u4e2d..."):
        adjusted_rgb, adjusted_sim, adjusted_de = auto_adjust_for_print(
            preview_rgb, cmyk_profile_path, intent_name, adjust_strength
        )
        orig_de = compute_delta_e(preview_rgb, simulated)
        orig_stats = compute_warning_stats(orig_de, threshold)
        adjusted_stats = compute_warning_stats(adjusted_de, threshold)

    # Before/After 統計比較
    st.markdown("### \u8abf\u6574\u524d\u5f8c\u306e\u6bd4\u8f03")
    col_b, col_a = st.columns(2)
    with col_b:
        st.markdown("**\u8abf\u6574\u524d**")
        st.metric("\u5e73\u5747 DeltaE", f"{orig_stats['mean_delta_e']:.1f}")
        st.metric("\u8b66\u544a\u30d4\u30af\u30bb\u30eb", f"{orig_stats['warning_percent']:.1f}%")
    with col_a:
        st.markdown("**\u8abf\u6574\u5f8c**")
        st.metric(
            "\u5e73\u5747 DeltaE",
            f"{adjusted_stats['mean_delta_e']:.1f}",
            delta=f"{adjusted_stats['mean_delta_e'] - orig_stats['mean_delta_e']:.1f}",
            delta_color="inverse",
        )
        st.metric(
            "\u8b66\u544a\u30d4\u30af\u30bb\u30eb",
            f"{adjusted_stats['warning_percent']:.1f}%",
            delta=f"{adjusted_stats['warning_percent'] - orig_stats['warning_percent']:.1f}%",
            delta_color="inverse",
        )

    # 画像比較
    st.markdown("### \u753b\u50cf\u6bd4\u8f03")
    col_img1, col_img2, col_img3 = st.columns(3)
    with col_img1:
        st.caption("Original (RGB)")
        st.image(preview_rgb, use_container_width=True)
    with col_img2:
        st.caption("\u8abf\u6574\u6e08\u307f (RGB)")
        st.image(adjusted_rgb, use_container_width=True)
    with col_img3:
        st.caption("\u8abf\u6574\u6e08\u307f\u306eCMYK\u30b7\u30df\u30e5\u30ec\u30fc\u30b7\u30e7\u30f3")
        st.image(adjusted_sim, use_container_width=True)

    st.markdown(
        "> \u300c\u8abf\u6574\u6e08\u307f (RGB)\u300d\u3092\u30c0\u30a6\u30f3\u30ed\u30fc\u30c9\u3057\u3066\u3001"
        "\u305d\u306e\u30d5\u30a1\u30a4\u30eb\u3092\u5370\u5237\u5165\u7a3f\u3059\u308c\u3070\u3001\u53f3\u306e\u30b7\u30df\u30e5\u30ec\u30fc\u30b7\u30e7\u30f3\u306b\u8fd1\u3044\u8272\u3067\u5370\u5237\u3055\u308c\u307e\u3059\u3002"
    )

    # 調整済みRGBダウンロード（ボタンを押した時だけオリジナルサイズで処理）
    base_name = uploaded_file.name.rsplit(".", 1)[0]

    if st.button("\U0001f504 \u30aa\u30ea\u30b8\u30ca\u30eb\u30b5\u30a4\u30ba\u3067\u8abf\u6574\u3092\u5b9f\u884c", type="secondary"):
        import io
        with st.spinner(
            f"\u30aa\u30ea\u30b8\u30ca\u30eb\u30b5\u30a4\u30ba ({original_rgb.size[0]}x{original_rgb.size[1]}) \u3067\u8abf\u6574\u4e2d..."
        ):
            full_adjusted, _, _ = auto_adjust_for_print(
                original_rgb, cmyk_profile_path, intent_name, adjust_strength
            )
            buf = io.BytesIO()
            full_adjusted.save(buf, format="PNG")
            st.session_state["adjusted_data"] = buf.getvalue()
            st.session_state["adjusted_filename"] = f"{base_name}_adjusted.png"
        st.success("\u5909\u63db\u5b8c\u4e86\uff01\u4e0b\u306e\u30dc\u30bf\u30f3\u304b\u3089\u30c0\u30a6\u30f3\u30ed\u30fc\u30c9\u3067\u304d\u307e\u3059\u3002")
        st.rerun()

    if "adjusted_data" in st.session_state:
        st.download_button(
            label=f"\u2b07\ufe0f {st.session_state.get('adjusted_filename', base_name + '_adjusted.png')} \u3092\u30c0\u30a6\u30f3\u30ed\u30fc\u30c9",
            data=st.session_state["adjusted_data"],
            file_name=st.session_state.get("adjusted_filename", f"{base_name}_adjusted.png"),
            mime="image/png",
            type="primary",
        )

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

    # 画像アップロード時点で自動変換
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
