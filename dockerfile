# Python 3 のイメージをベースにする
FROM python:3

# コンテナの実行ユーザーは指定していません

# ロケールの設定
RUN apt-get update && \
    apt-get -y install locales && \
    localedef -f UTF-8 -i ja_JP ja_JP.UTF-8

# ロケール関連の環境変数を設定
ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
ENV LC_ALL ja_JP.UTF-8

# タイムゾーンを設定
ENV TZ JST-9

# ターミナルの設定
ENV TERM xterm

# Vim と less のインストール
RUN apt-get install -y vim less

# pip のアップグレード
RUN pip install --upgrade pip

# setuptools のアップグレード
RUN pip install --upgrade setuptools

# JupyterLab のインストール
RUN python -m pip install jupyterlab

# Torch のインストール
RUN python -m pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu118

# Ninja のインストール
RUN python -m pip install ninja

# Flash-Attn のインストール
RUN python -m pip install -U flash-attn --no-build-isolation

# exllamav2 のインストール
RUN python -m pip install https://github.com/turboderp/exllamav2/releases/download/v0.0.18/exllamav2-0.0.18+cu118-cp310-cp310-linux_x86_64.whl

# dbutils.library.restartPython() はコメントアウトされています
# dbutils は Dockerfile のコンテキストからはわかりませんが、必要であれば適切なコマンドに置き換えてください

