FROM python:3.10-slim

# システムライブラリのインストール
# GUI (Tkinter) および OpenCVに必要な依存関係
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    python3-tk \
    tk \
    xvfb \
    x11-apps \
    fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# パッケージのインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ソースコードや設定ディレクトリをコンテナ内に配置するためのマウントポイントは
# docker-compose.yml 側で定義します

# デフォルトの実行コマンド（docker-composeで上書き可能）
CMD ["python", "main.py"]
