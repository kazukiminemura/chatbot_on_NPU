# Chatbot on AI PC

Intel AI PC (NPU) 上で OpenVINO を利用し、DeepSeek-R1-Distill-Qwen-1.5B-int4-cw-ov モデルを動作させる個人向けチャットボットのサンプルです。完全ローカルでの推論、Web UI からの利用、WebSocket ストリーミング応答を想定しています。

## プロジェクト概要

- **高速推論**: Intel AI PC の NPU 向けに最適化された OpenVINO モデルを使用
- **ブラウザ操作**: FastAPI + WebSocket によるリアルタイム応答
- **完全ローカル**: すべての処理をローカルマシン内で完結

```
chatbot_on_AIPC/
├── docs/                          # プロジェクト文書
│   ├── requirements_definition.md # 要件定義書
│   └── technical_specification.md # 技術仕様書
├── app/                           # アプリケーションコード
├── static/                        # Web フロントエンド
├── models/                        # 推論用 OpenVINO モデル
├── config.json                    # 設定ファイル
└── requirements.txt               # Python 依存関係
```

## 推奨環境

- Windows 10 / 11
- Python 3.9 以上 (3.12 までを推奨)
- Intel AI PC (NPU 搭載)
- RAM 8 GB 以上

## セットアップ手順

1. リポジトリを取得します。
   ```powershell
   git clone https://github.com/kazukiminemura/chatbot_on_AIPC.git
   cd chatbot_on_AIPC
   ```
2. 仮想環境を作成し、必ずクリーンな状態から開始します。
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate
   python -m pip install --upgrade pip wheel
   ```
3. 依存関係をインストールします。既存環境からの混入を避けるため、上記のクリーンな仮想環境内で実行してください。
   ```powershell
   pip install -r requirements.txt
   ```
4. アプリケーションを起動します。
   ```powershell
   python run.py
   ```
5. ブラウザで `http://localhost:8000` にアクセスするとチャットボット UI が表示されます。

## 依存関係インストール時の注意点

- `starlette` のバージョン競合が発生した場合  
  FastAPI 0.110 以降は `starlette>=0.37.2,<0.38.0` を要求します。既に `starlette==0.36.3` が入っている環境で `pip install -r requirements.txt` を実行すると次のエラーになります。
  ```
  ERROR: Cannot install -r requirements.txt (line 1) and starlette==0.36.3 because these package versions have conflicting dependencies.
  ```
  **対策**
  - 仮想環境を作り直す、または
  - `pip uninstall starlette` を実行してから再度 `pip install -r requirements.txt` を行う、もしくは
  - 明示的に互換バージョンをインストールする: `pip install "starlette>=0.37.2,<0.38.0"`

- `huggingface_hub.errors` が見つからない場合  
  `optimum` から `huggingface_hub.errors.OfflineModeIsEnabled` を参照します。旧バージョンの `huggingface-hub` だと同モジュールが存在しないため、以下で最新版へ更新してください。
  ```powershell
  pip install --upgrade "huggingface-hub>=0.23.0"
  ```

## モデルの配置

アプリケーション起動時には OpenVINO 形式のモデルファイル `openvino_model.bin` と `openvino_model.xml` が `models/` 配下に存在する必要があります。

### 推奨ダウンロード手順

1. Hugging Face CLI をインストール (未導入の場合)
   ```powershell
   pip install "huggingface_hub[cli]"
   ```
2. モデルをローカルにダウンロードし、既定のパスへ展開します。
   ```powershell
   huggingface-cli download \
     OpenVINO/DeepSeek-R1-Distill-Qwen-1.5B-int4-cw-ov \
     --local-dir models/models--OpenVINO--DeepSeek-R1-Distill-Qwen-1.5B-int4-cw-ov
   ```
3. `models/models--OpenVINO--DeepSeek-R1-Distill-Qwen-1.5B-int4-cw-ov/snapshots/<commit-id>/openvino_model.bin` が存在することを確認してください。ファイルが見つからない場合はダウンロードに失敗した可能性があるため、再実行してください。

> **NOTE**  
> Windows でパスが長くなる場合は `git config --system core.longpaths true` を設定するか、短いディレクトリ名にクローンしてからダウンロードすると解消できます。

## よくあるエラーと対処

| エラー | 原因 | 解決方法 |
| --- | --- | --- |
| `Cannot install -r requirements.txt (line 1) and starlette==0.36.3...` | 既存環境に古い `starlette` が残っている | 新しい仮想環境を作る / `pip uninstall starlette` 後に再インストール / `pip install "starlette>=0.37.2,<0.38.0"` |
| `ModuleNotFoundError: No module named 'huggingface_hub.errors'` | `huggingface-hub` が旧バージョン | `pip install --upgrade "huggingface-hub>=0.23.0"` を実行 |
| `Can not open file ... openvino_model.bin for mapping` | OpenVINO モデルファイルが未配置または破損 | 上記「モデルの配置」の手順で再ダウンロード。パスが正しいか、アクセス権限があるかも確認 |

問題が解決しない場合は、エラーログとともに Issue を作成してください。

## ドキュメント

- `docs/requirements_definition.md`
- `docs/technical_specification.md`

---

Made with 🤖 AI assistance
