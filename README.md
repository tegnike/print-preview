# Print Preview - RGB→CMYK 印刷シミュレーション

RGB画像をCMYK変換した際の色変化を事前に確認できるツールです。

![Print Preview](image.png)

## 機能

- **比較表示**: RGB原画とCMYKシミュレーション画像をSide-by-Sideで比較
- **色警告ヒートマップ**: 印刷で色が変わる箇所を黄色（注意）/赤（警告）で可視化
- **統計表示**: 平均DeltaE、最大DeltaE、警告ピクセルの割合
- **CMYKエクスポート**: ICCプロファイル埋め込み済みのCMYK画像（TIFF/JPG）をダウンロード

## 対応ICCプロファイル

| プロファイル | 用途 |
|---|---|
| JapanColor2001Coated | 日本の印刷所に入稿する場合（コート紙標準） |
| Generic CMYK (macOS) | 汎用 |

## セットアップ

```bash
cd print-preview
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 起動

```bash
source .venv/bin/activate
streamlit run app.py
```

ブラウザで http://localhost:8501 が開きます。

## 使い方

1. サイドバーから画像をアップロード
2. CMYKプロファイルとレンダリングインテントを選択
3. **「色警告」タブ** で印刷時に色が変わる箇所を確認
4. 必要なら「エクスポート」タブからCMYK変換済み画像をダウンロード

## 必要環境

- Python 3.10+
- ImageMagick（CMYKエクスポート機能に必要。`brew install imagemagick`）
