#!/bin/bash
# setup_env.sh ＝ 比較実験に必要な既製OCR を sudo なし・ダイアログなしで用意する（2026-10-18）
#
#  Tesseract は apt で入れるには管理者権限が要るが、この環境の sudo は対話的な認証を求める
#  （無人ジョブではダイアログを出せない）。そこで Ubuntu 公式アーカイブから .deb を
#  `apt-get download`（権限不要）で取り、`dpkg-deb -x` でホーム配下へ展開して使う。
#  EasyOCR・PaddleOCR は pip。python3.12 は uv が持ってくる（PaddlePaddle に python3.14 用の
#  配布物が無いため・下の「分かったこと」参照）。
#
# 分かったこと（2026-10-18 実測）
#   - WSL Ubuntu 26.04 の既定 python は 3.14 で、paddlepaddle には cp314 の配布物が無い
#     （`pip install paddlepaddle` → No matching distribution found）。
#   - uv で python3.12 を入れれば、pytesseract・EasyOCR・PaddleOCR の3つが同じ環境に揃う。
#   - 本番の venv（~/karaoke-vision/.venv）は汚さない。実験は ~/ocr_compare_20261018/.venv312。
set -e
BASE="$HOME/ocr_compare_20261018"
mkdir -p "$BASE/pkg"

# ---- 1) Tesseract 本体（公式アーカイブの .deb をホームへ展開） ----
cd "$BASE/pkg"
apt-get download tesseract-ocr tesseract-ocr-eng tesseract-ocr-osd libtesseract5 libleptonica6
# libleptonica6 / libtesseract5 の実行時依存（システムに無いものだけ）
apt-get download libgif7 libjpeg8 libjpeg-turbo8 libopenjp2-7 libtiff6 libwebp7 libwebpmux3 \
                 libgomp1 libicu78 libcairo2 libpango-1.0-0 libpangocairo-1.0-0 libharfbuzz0b \
                 libfontconfig1 libglib2.0-0t64 libdeflate0 libjbig0 liblerc4 libzstd1 \
                 liblzma5 libbz2-1.0 libsharpyuv0 || true
# -updates 側の索引が古く 404 になることがあるので版を固定して取る
apt-get download libarchive13t64=3.8.5-1ubuntu2
cd "$BASE"
mkdir -p opt
for d in pkg/*.deb; do dpkg-deb -x "$d" opt/ ; done
cat > "$BASE/env.sh" <<EOF
export LD_LIBRARY_PATH="$BASE/opt/usr/lib/x86_64-linux-gnu:$BASE/opt/lib/x86_64-linux-gnu"
export PATH="$BASE/opt/usr/bin:\$PATH"
export TESSDATA_PREFIX="$BASE/opt/usr/share/tesseract-ocr/5/tessdata"
EOF
source "$BASE/env.sh"
ldd opt/usr/bin/tesseract | grep -i "not found" && { echo "共有ライブラリが足りない"; exit 1; } || true
tesseract --version | head -2

# ---- 2) python3.12 の venv（uv・権限不要） ----
if [ ! -d "$BASE/.venv" ]; then python3.14 -m venv "$BASE/.venv"; fi
"$BASE/.venv/bin/python" -m pip install -q --upgrade pip uv
"$BASE/.venv/bin/uv" python install 3.12
"$BASE/.venv/bin/uv" venv --python 3.12 "$BASE/.venv312"
P="$BASE/.venv312/bin/python"
$P -m ensurepip --upgrade
$P -m pip install -q --upgrade pip

# ---- 3) 既製OCR ----
$P -m pip install "numpy<3" opencv-python-headless pillow pytesseract
$P -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
$P -m pip install easyocr
$P -m pip install paddlepaddle paddleocr

$P - <<'PY'
import importlib
for m in ['cv2','numpy','PIL','pytesseract','torch','easyocr','paddle','paddleocr']:
    try:
        mm = importlib.import_module(m); print(m, 'OK', getattr(mm, '__version__', '?'))
    except Exception as e:
        print(m, 'NO', type(e).__name__, str(e)[:120])
PY
echo "setup_env.sh done"
