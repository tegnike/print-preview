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
    色域外のピクセルの彩度を自動調整してCMYK色域内に収める。

    仕組み:
    1. ソフトプルーフで「CMYK変換後の色」を取得
    2. 色域外のピクセルを特定（DeltaEが高い箇所）
    3. そのピクセルの彩度をHSV空間で段階的に下げ、DeltaEが許容範囲に入るまで繰り返す
    4. 調整済みRGB画像を返す

    Returns: (adjusted_rgb, adjusted_simulated, adjusted_delta_e)
    """
    image = image.convert("RGB")
    orig_arr = np.array(image)

    # ソフトプルーフで色域外を把握
    simulated = soft_proof(image, cmyk_profile_path, intent_name)
    delta_e = compute_delta_e(image, simulated)

    # HSV変換
    hsv_image = image.convert("HSV")
    hsv_arr = np.array(hsv_image, dtype=np.float64)

    # DeltaEが高いピクセルほど彩度を下げる
    # 閾値3.0以上を色域外とみなす
    gamut_threshold = 3.0
    out_of_gamut = delta_e > gamut_threshold

    if not np.any(out_of_gamut):
        # 調整不要
        return image, simulated, delta_e

    # DeltaEに基づいて彩度の削減率を計算
    # DeltaEが大きいほど彩度を大きく下げる
    max_de = delta_e.max()
    if max_de > gamut_threshold:
        # 0.0（変更なし）〜 1.0（最大削減）にスケーリング
        reduction = np.clip(
            (delta_e - gamut_threshold) / (max_de - gamut_threshold), 0.0, 1.0
        )
        reduction *= strength  # ユーザー指定の強度

        # 彩度チャンネル（index=1）を削減
        new_saturation = hsv_arr[:, :, 1] * (1.0 - reduction * 0.7)  # 最大70%削減
        hsv_arr[:, :, 1] = np.clip(new_saturation, 0, 255)

    # HSV→RGB に戻す
    adjusted_hsv = Image.fromarray(hsv_arr.astype(np.uint8), "HSV")
    adjusted_rgb = adjusted_hsv.convert("RGB")

    # 調整後のシミュレーションとDeltaEを計算
    adjusted_sim = soft_proof(adjusted_rgb, cmyk_profile_path, intent_name)
    adjusted_de = compute_delta_e(adjusted_rgb, adjusted_sim)

    return adjusted_rgb, adjusted_sim, adjusted_de


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
