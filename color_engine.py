"""
RGB→CMYK 印刷シミュレーションエンジン
ICCプロファイルを使用したソフトプルーフ、色差計算、CMYK書き出し
"""

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms

# システムICCプロファイルパス
SYSTEM_PROFILES_DIR = Path("/System/Library/ColorSync/Profiles")
SRGB_PROFILE_PATH = SYSTEM_PROFILES_DIR / "sRGB Profile.icc"
GENERIC_CMYK_PROFILE_PATH = SYSTEM_PROFILES_DIR / "Generic CMYK Profile.icc"

# バンドルプロファイルパス
BUNDLE_PROFILES_DIR = Path(__file__).parent / "profiles"

# レンダリングインテント
INTENTS = {
    "Perceptual": ImageCms.Intent.PERCEPTUAL,
    "Relative Colorimetric": ImageCms.Intent.RELATIVE_COLORIMETRIC,
    "Saturation": ImageCms.Intent.SATURATION,
    "Absolute Colorimetric": ImageCms.Intent.ABSOLUTE_COLORIMETRIC,
}


def load_profiles() -> dict[str, Path]:
    """利用可能なCMYK ICCプロファイルを検出して返す"""
    profiles = {}

    # システムプロファイル
    if GENERIC_CMYK_PROFILE_PATH.exists():
        profiles["Generic CMYK (macOS)"] = GENERIC_CMYK_PROFILE_PATH

    # バンドルプロファイル
    if BUNDLE_PROFILES_DIR.exists():
        for icc_file in BUNDLE_PROFILES_DIR.glob("*.icc"):
            name = icc_file.stem.replace("_", " ").replace("-", " ")
            profiles[name] = icc_file

    return profiles


def soft_proof(
    image: Image.Image,
    cmyk_profile_path: Path,
    intent_name: str = "Perceptual",
) -> Image.Image:
    """
    ソフトプルーフ: RGB画像をCMYK経由でRGBに戻し、印刷後の色をシミュレーション
    """
    image = image.convert("RGB")

    srgb_profile = ImageCms.getOpenProfile(str(SRGB_PROFILE_PATH))
    cmyk_profile = ImageCms.getOpenProfile(str(cmyk_profile_path))

    proof_intent = INTENTS.get(intent_name, ImageCms.Intent.PERCEPTUAL)

    proof_transform = ImageCms.buildProofTransform(
        inputProfile=srgb_profile,
        outputProfile=srgb_profile,
        proofProfile=cmyk_profile,
        inMode="RGB",
        outMode="RGB",
        renderingIntent=ImageCms.Intent.PERCEPTUAL,
        proofRenderingIntent=proof_intent,
        flags=ImageCms.Flags.SOFTPROOFING,
    )

    simulated = image.copy()
    ImageCms.applyTransform(simulated, proof_transform, inPlace=True)
    return simulated


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    """sRGB [0,1] -> linear RGB [0,1]"""
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _srgb_to_lab(arr_uint8: np.ndarray) -> np.ndarray:
    """sRGB uint8 画像を CIE L*a*b* に変換 (D65)"""
    rgb = arr_uint8.astype(np.float64) / 255.0
    linear = _srgb_to_linear(rgb)

    # sRGB -> XYZ (D65)
    r, g, b = linear[:, :, 0], linear[:, :, 1], linear[:, :, 2]
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    # D65 白色点で正規化
    x /= 0.95047
    z /= 1.08883

    def f(t):
        delta = 6.0 / 29.0
        return np.where(t > delta**3, np.cbrt(t), t / (3 * delta**2) + 4.0 / 29.0)

    fx, fy, fz = f(x), f(y), f(z)

    lab = np.zeros_like(rgb)
    lab[:, :, 0] = 116.0 * fy - 16.0   # L*
    lab[:, :, 1] = 500.0 * (fx - fy)    # a*
    lab[:, :, 2] = 200.0 * (fy - fz)    # b*
    return lab


def compute_delta_e(
    original: Image.Image, simulated: Image.Image
) -> np.ndarray:
    """
    CIE76 DeltaE を Lab 色空間で計算
    Returns: 2D numpy array of per-pixel DeltaE values
    """
    orig_arr = np.array(original.convert("RGB"))
    sim_arr = np.array(simulated.convert("RGB"))

    orig_lab = _srgb_to_lab(orig_arr)
    sim_lab = _srgb_to_lab(sim_arr)

    diff = orig_lab - sim_lab
    delta_e = np.sqrt(np.sum(diff**2, axis=2))
    return delta_e


def create_warning_overlay(
    original: Image.Image, delta_e: np.ndarray, threshold: float
) -> Image.Image:
    """
    色差ヒートマップオーバーレイを生成
    緑=OK, 黄=注意, 赤=警告
    """
    original = original.convert("RGBA")
    h, w = delta_e.shape

    overlay = np.zeros((h, w, 4), dtype=np.uint8)

    # 閾値以下: 透明（表示しない）
    # 閾値〜閾値*2: 黄色（注意）
    # 閾値*2以上: 赤（警告）
    moderate = (delta_e >= threshold) & (delta_e < threshold * 2)
    severe = delta_e >= threshold * 2

    # 黄色オーバーレイ (R=255, G=200, B=0, A=100)
    overlay[moderate] = [255, 200, 0, 100]
    # 赤オーバーレイ (R=255, G=0, B=0, A=120)
    overlay[severe] = [255, 0, 0, 120]

    overlay_image = Image.fromarray(overlay, "RGBA")
    result = Image.alpha_composite(original, overlay_image)
    return result


def compute_warning_stats(delta_e: np.ndarray, threshold: float) -> dict:
    """色差の統計情報を計算"""
    total_pixels = delta_e.size
    warning_pixels = np.sum(delta_e >= threshold)
    severe_pixels = np.sum(delta_e >= threshold * 2)

    return {
        "mean_delta_e": float(np.mean(delta_e)),
        "max_delta_e": float(np.max(delta_e)),
        "median_delta_e": float(np.median(delta_e)),
        "warning_percent": float(warning_pixels / total_pixels * 100),
        "severe_percent": float(severe_pixels / total_pixels * 100),
        "total_pixels": int(total_pixels),
    }


def auto_adjust_for_print(
    image: Image.Image,
    cmyk_profile_path: Path,
    intent_name: str = "Perceptual",
    strength: float = 1.0,
) -> tuple[Image.Image, Image.Image, np.ndarray]:
    """
    逆補正アプローチで印刷後の色をオリジナルに近づける。

    仕組み:
    1. ソフトプルーフで「CMYK変換すると色がどうズレるか」を計算
    2. そのズレの逆方向にあらかじめ色を補正（逆補正）
    3. 反復しながら、ピクセルごとに「改善する場合のみ補正を採用」
    4. 補正した画像をCMYK変換すると、元の色に近くなる

    Returns: (adjusted_rgb, adjusted_simulated, adjusted_delta_e)
    """
    image = image.convert("RGB")
    orig_arr = np.array(image, dtype=np.float64)

    # オリジナルのLab（比較基準）
    orig_lab = _srgb_to_lab(np.array(image))

    # 最良結果を追跡
    best_arr = orig_arr.copy()
    best_sim = soft_proof(image, cmyk_profile_path, intent_name)
    best_de = compute_delta_e(image, best_sim)

    adjusted_arr = orig_arr.copy()

    # 反復逆補正（3回固定、減衰あり）
    for i in range(3):
        damping = strength * (0.8 ** i)  # 反復ごとに減衰

        current_img = Image.fromarray(
            np.clip(adjusted_arr, 0, 255).astype(np.uint8)
        )
        current_sim = soft_proof(current_img, cmyk_profile_path, intent_name)
        sim_arr = np.array(current_sim, dtype=np.float64)

        # ズレ = オリジナル - 現在のシミュレーション結果
        error = orig_arr - sim_arr

        # 逆補正を加算
        candidate_arr = np.clip(adjusted_arr + error * damping, 0, 255)

        # 候補をソフトプルーフして効果を検証
        candidate_img = Image.fromarray(candidate_arr.astype(np.uint8))
        candidate_sim = soft_proof(candidate_img, cmyk_profile_path, intent_name)

        # ピクセルごとにDeltaEを比較して、改善した場合のみ採用
        candidate_sim_lab = _srgb_to_lab(np.array(candidate_sim))
        candidate_de = np.sqrt(np.sum((orig_lab - candidate_sim_lab) ** 2, axis=2))

        improved = candidate_de < best_de
        improved_3d = improved[:, :, np.newaxis]

        # 改善したピクセルだけ更新
        best_arr = np.where(improved_3d, candidate_arr, best_arr)
        best_de = np.minimum(best_de, candidate_de)
        adjusted_arr = np.where(improved_3d, candidate_arr, adjusted_arr)

    adjusted_rgb = Image.fromarray(
        np.clip(best_arr, 0, 255).astype(np.uint8)
    )
    adjusted_sim = soft_proof(adjusted_rgb, cmyk_profile_path, intent_name)
    adjusted_de = compute_delta_e(image, adjusted_sim)

    return adjusted_rgb, adjusted_sim, adjusted_de


def _build_lut(control_points: list[tuple[int, int]]) -> np.ndarray:
    """
    コントロールポイントからスプライン補間で256エントリのLUTを作成。
    control_points: [(input, output), ...] 0-255の範囲
    """
    # 始点と終点を保証
    points = sorted(control_points, key=lambda p: p[0])
    if points[0][0] != 0:
        points.insert(0, (0, 0))
    if points[-1][0] != 255:
        points.append((255, 255))

    xs = np.array([p[0] for p in points], dtype=np.float64)
    ys = np.array([p[1] for p in points], dtype=np.float64)

    # 全入力値(0-255)に対して補間
    all_x = np.arange(256, dtype=np.float64)
    lut = np.interp(all_x, xs, ys)
    return np.clip(lut, 0, 255).astype(np.uint8)


def apply_tone_curves(
    image: Image.Image,
    r_points: list[tuple[int, int]] | None = None,
    g_points: list[tuple[int, int]] | None = None,
    b_points: list[tuple[int, int]] | None = None,
) -> Image.Image:
    """
    RGB各チャンネルにトーンカーブ(LUT)を適用する。

    各チャンネルのcontrol_points: [(input, output), ...]
    Noneの場合はそのチャンネルは変更なし（リニア）。
    """
    image = image.convert("RGB")
    arr = np.array(image)

    identity = [(0, 0), (255, 255)]

    r_lut = _build_lut(r_points if r_points else identity)
    g_lut = _build_lut(g_points if g_points else identity)
    b_lut = _build_lut(b_points if b_points else identity)

    arr[:, :, 0] = r_lut[arr[:, :, 0]]
    arr[:, :, 1] = g_lut[arr[:, :, 1]]
    arr[:, :, 2] = b_lut[arr[:, :, 2]]

    return Image.fromarray(arr)


def export_cmyk(
    image: Image.Image,
    cmyk_profile_path: Path,
    intent_name: str = "Relative Colorimetric",
    output_format: str = "tiff",
) -> bytes:
    """
    ImageMagickを使用してCMYK画像を書き出し、バイト列として返す
    """
    intent_map = {
        "Perceptual": "Perceptual",
        "Relative Colorimetric": "Relative",
        "Saturation": "Saturation",
        "Absolute Colorimetric": "Absolute",
    }
    im_intent = intent_map.get(intent_name, "Relative")

    ext = ".tif" if output_format == "tiff" else f".{output_format}"

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.png")
        output_path = os.path.join(tmpdir, f"output{ext}")

        image.convert("RGB").save(input_path, "PNG")

        cmd = [
            "magick", input_path,
            "-profile", str(SRGB_PROFILE_PATH),
            "-intent", im_intent,
            "-profile", str(cmyk_profile_path),
        ]
        if output_format == "tiff":
            cmd.extend(["-compress", "LZW"])
        cmd.append(output_path)

        subprocess.run(cmd, check=True, capture_output=True)

        with open(output_path, "rb") as f:
            return f.read()
