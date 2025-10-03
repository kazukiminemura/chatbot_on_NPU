# 技術実装仕様書 - DeepSeek-R1-Distill-Qwen-1.5B版

## 1. プロジェクト構成

```
chatbot_on_NPU/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPIメインアプリケーション
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model_manager.py # モデル管理クラス
│   │   └── ov_inference.py  # OpenVINO GenAI推論エンジン
│   ├── api/
│   │   ├── __init__.py
│   │   ├── chat.py          # チャットAPI
│   │   └── websocket.py     # WebSocket通信
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py        # 設定管理
│   │   └── logger.py        # ログ設定
│   ├── utils/
│       ├── __init__.py
│       └── download.py      # モデル自動ダウンロード
├── static/
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── chat.js
│   └── index.html
├── models/                   # モデル保存ディレクトリ（自動作成）
├── logs/                     # ログファイル
├── config.json              # 設定ファイル
├── requirements.txt         # Python依存関係
├── run.py                   # アプリケーション起動スクリプト
└── README.md               # セットアップ・使用方法
```

## 2. 技術コンポーネント詳細

### 2.1 バックエンド (FastAPI)

#### メインアプリケーション (main.py)
- FastAPIアプリケーションの初期化
- 静的ファイル配信設定
- CORS設定
- WebSocketエンドポイント設定

#### モデル管理 (model_manager.py)
- DeepSeek-R1-Distill-Qwen-1.5Bモデルの管理
- OpenVINO形式での最適化済みモデル読み込み
- NPUデバイス検出・設定
- モデルキャッシュ管理

#### OpenVINO GenAI推論エンジン (ov_inference.py)
- OpenVINO GenAI LLMPipelineを使用
- 自動モデルダウンロード機能
- NPUデバイス自動検出・フォールバック
- ネイティブストリーミング応答
- 最適化されたトークン生成処理

### 2.2 フロントエンド

#### HTML (index.html)
- チャットインターフェース
- レスポンシブデザイン
- システム情報表示エリア

#### JavaScript (chat.js)
- WebSocket通信管理
- リアルタイムメッセージ表示
- ユーザー入力処理
- ストリーミング応答表示

#### CSS (style.css)
- モダンなチャットUI
- ダークモード対応
- アニメーション効果

### 2.3 OpenVINO GenAI推論システム

#### OpenVINO GenAI移行
- **旧実装**: OpenVINO + Optimum Intel + Transformers
- **新実装**: OpenVINO GenAI LLMPipeline
- **利点**: 
  - 2-3倍の推論速度向上
  - メモリ使用量削減
  - コード複雑性の大幅軽減
  - ネイティブストリーミングサポート

#### 依存関係の最適化
```python
# 削除された依存関係
- optimum-intel
- torch (推論用途)
- transformers (一部機能)

# 追加された依存関係
- openvino-genai>=2024.4.0
```

#### NPU最適化設定
```python
# NPU専用設定
device_config = {
    "NPU_USE_NPUW": "YES",
    "PERFORMANCE_HINT": "LATENCY",
    "CACHE_MODE": "OPTIMIZE_SPEED",
    "CACHE_DIR": "./cache"
}

# CPU フォールバック設定
cpu_config = {
    "PERFORMANCE_HINT": "LATENCY",
    "INFERENCE_NUM_THREADS": "4",
    "CACHE_DIR": "./cache"
}
```

### 2.4 自動モデル管理システム

#### 自動ダウンロード機能
- **モデル**: `OpenVINO/DeepSeek-R1-Distill-Qwen-1.5B-int4-cw-ov`
- **サイズ**: 約1.2GB (INT4量子化)
- **ダウンロード先**: `./models/OpenVINO_DeepSeek-R1-Distill-Qwen-1.5B-int4-cw-ov/`

#### モデルキャッシュ管理
```python
# モデル検証
required_files = [
    "openvino_model.xml",  # モデル構造
    "openvino_model.bin",  # モデル重み
    "config.json"          # 設定ファイル
]

# キャッシュ確認フロー
1. ローカルモデル存在確認
2. 必要ファイルの整合性チェック
3. 不完全な場合は自動再ダウンロード
4. フォールバック: リモートから直接読み込み
```

#### ダウンロードツール
```cmd
# 自動ダウンロード（推奨）
python run.py  # 初回起動時に自動ダウンロード

# 手動ダウンロード（必要に応じて）
python -c "from app.utils.download import download_model; download_model('OpenVINO/DeepSeek-R1-Distill-Qwen-1.5B-int4-cw-ov')"
```

## 3. API仕様

### 3.1 REST API

#### GET /health
システムヘルスチェック
```json
{
  "status": "healthy",
  "model_loaded": true,
  "npu_available": true,
  "memory_usage": "2.1GB"
}
```

#### POST /api/chat
単発チャット（非ストリーミング）
```json
// Request
{
  "message": "こんにちは",
  "max_tokens": 500,
  "temperature": 0.7
}

// Response
{
  "response": "こんにちは！何かお手伝いできることはありますか？",
  "inference_time": 1.23,
  "tokens_generated": 15
}
```

### 3.2 WebSocket API

#### /ws/chat
リアルタイムチャット
```json
// Send
{
  "type": "message",
  "data": {
    "message": "質問内容",
    "settings": {
      "max_tokens": 500,
      "temperature": 0.7
    }
  }
}

// Receive (ストリーミング)
{
  "type": "token",
  "data": {
    "token": "こんにちは",
    "is_final": false
  }
}

// Receive (完了)
{
  "type": "complete",
  "data": {
    "inference_time": 2.45,
    "total_tokens": 47
  }
}
```

## 4. セットアップ手順

### 4.1 環境要件
- Windows 10/11
- Python 3.9以上
- Intel NPU対応デバイス（推奨）
- 6GB以上のRAM
- 8GB以上の空きストレージ（モデル含む）

### 4.2 自動セットアップ（推奨）
```cmd
# 1. 仮想環境作成・アクティベート
python -m venv venv
venv\Scripts\activate

# 2. 依存関係インストール
pip install -r requirements.txt

# 3. アプリケーション起動（初回時モデル自動ダウンロード）
python run.py
```

### 4.3 手動セットアップ
```cmd
# 1. 仮想環境作成
python -m venv venv
call venv\Scripts\activate.bat

# 2. 依存関係インストール
pip install openvino>=2024.4.0
pip install openvino-genai>=2024.4.0
pip install -r requirements.txt

# 3. モデル事前ダウンロード（オプション）
python download_model.py

# 4. アプリケーション起動
python run.py
```

### 4.4 トラブルシューティング

#### OpenVINO GenAI インポートエラー
```
ImportError: No module named 'openvino_genai'
```
**解決策:**
- `pip install openvino-genai>=2024.4.0` を実行
- 正しい仮想環境がアクティブか確認

#### モデルダウンロードエラー
```
HTTPSConnectionPool... Connection failed
```
**解決策:**
1. インターネット接続確認
2. `pip install --upgrade huggingface-hub`
3. ファイアウォール設定確認

#### NPU利用不可
```
WARNING - NPU device not available. Falling back to CPU.
```
**解決策:**
- Intel NPU対応デバイス確認
- 最新ドライバーインストール
- CPU フォールバックで動作継続

### 4.5 設定ファイル (config.json)
```json
{
  "model": {
    "name": "DeepSeek-R1-Distill-Qwen-1.5B",
    "repo_id": "OpenVINO/DeepSeek-R1-Distill-Qwen-1.5B-int4-cw-ov",
    "model_type": "qwen",
    "max_context_length": 32768,
    "precision": "INT4"
  },
  "inference": {
    "max_tokens": 1000,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1,
    "do_sample": true
  },
  "server": {
    "host": "localhost",
    "port": 8000,
    "log_level": "INFO"
  },
  "hardware": {
    "device": "NPU",
    "precision": "FP16",
    "batch_size": 1
  }
}
```

## 5. パフォーマンス最適化

### 5.1 OpenVINO GenAI 最適化
- **推論速度**: 従来比2-3倍向上
- **メモリ効率**: 大幅なメモリ使用量削減
- **ネイティブストリーミング**: リアルタイム応答生成
- **自動最適化**: デバイス別自動設定

### 5.2 NPU最適化
```python
# NPU専用最適化設定
npu_config = {
    "NPU_USE_NPUW": "YES",
    "PERFORMANCE_HINT": "LATENCY",
    "CACHE_MODE": "OPTIMIZE_SPEED"
}
```

### 5.3 モデル管理最適化
- **ローカルキャッシュ**: 初回ダウンロード後は高速起動
- **自動検証**: モデルファイル整合性チェック
- **フォールバック**: 自動デバイス切り替え

### 5.4 通信最適化
- WebSocketによる低遅延通信
- ストリーミング応答
- 効率的なJSON serialization

## 6. エラーハンドリング

### 6.1 モデル関連エラー
- **モデルダウンロード失敗**: 自動リトライ・フォールバック
- **モデルファイル不完全**: 自動再ダウンロード
- **NPU利用不可**: CPU自動フォールバック

### 6.2 OpenVINO GenAI エラー
- **LLMPipeline初期化失敗**: デバイス自動切り替え
- **メモリ不足**: バッチサイズ調整・警告表示
- **推論タイムアウト**: 適切なエラーメッセージ

### 6.3 ネットワーク・通信エラー
- **WebSocket接続断**: 自動再接続試行
- **ダウンロードエラー**: 詳細エラー情報・解決方法表示
- **JSON解析エラー**: 入力検証・エラー応答

## 7. テスト戦略

### 7.1 単体テスト
- OpenVINO GenAI推論エンジン
- 自動モデルダウンロード機能
- API エンドポイント
- エラーハンドリング

### 7.2 統合テスト
- エンドツーエンドチャット
- モデル自動ダウンロード・キャッシュ
- NPU/CPU フォールバック
- ストリーミング応答

### 7.3 パフォーマンステスト
- 応答時間測定（OpenVINO GenAI vs 従来）
- メモリ使用量監視
- NPU vs CPU パフォーマンス比較
- ダウンロード速度・進捗確認

### 7.4 Migration テスト
- 旧環境からの移行確認
- 依存関係の互換性
- 設定ファイル移行
- 機能回帰テスト

---

## 8. Migration Summary (OpenVINO GenAI)

| 項目 | 旧実装 (Optimum Intel) | 新実装 (OpenVINO GenAI) |
|------|----------------------|----------------------|
| **依存関係** | 5+パッケージ | 2メインパッケージ |
| **コード複雑性** | 高（手動トークン化） | 低（内蔵処理） |
| **ストリーミング** | シミュレート | ネイティブ |
| **パフォーマンス** | 良好 | 優秀（2-3倍向上） |
| **メモリ使用量** | 高 | 低 |
| **ハードウェア対応** | 限定的 | 強化 |
| **セットアップ** | 複雑 | 簡単 |
| **保守性** | 難 | 易 |

この技術仕様書に基づいて、OpenVINO GenAIを使用した最新の実装を運用することができます。