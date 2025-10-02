# 技術実装仕様書

## 1. プロジェクト構成

```
chatbot_on_NPU/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPIメインアプリケーション
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model_manager.py # モデル管理クラス
│   │   └── ov_inference.py  # OpenVINO推論エンジン
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
│       ├── model_converter.py # HuggingFace→OpenVINO変換
│       └── download.py       # モデルダウンロード
├── static/
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── chat.js
│   └── index.html
├── models/                   # モデル保存ディレクトリ
├── logs/                     # ログファイル
├── config.json              # 設定ファイル
├── requirements.txt         # Python依存関係
├── setup.py                 # セットアップスクリプト
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
- モデルダウンロード管理
- HuggingFace→OpenVINO変換処理
- NPUデバイス検出・設定
- モデルキャッシュ管理

#### OpenVINO推論エンジン (ov_inference.py)
- OpenVINOランタイム初期化
- NPUコンパイル設定
- トークン生成処理
- ストリーミング応答生成

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

### 2.3 NPU最適化設定

#### OpenVINOコンパイル設定
```python
compile_config = {
    "NPU_USE_NPUW": "YES",
    "NPU_COMPILATION_MODE_PARAMS": "compute-layers-with-higher-precision=Softmax,Add",
    "INFERENCE_PRECISION_HINT": "f16"
}
```

#### モデル量子化
- INT8量子化によるパフォーマンス向上
- NPU専用最適化パイプライン

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
- Intel NPU対応デバイス
- 6GB以上のRAM
- 8GB以上の空きストレージ

### 4.2 インストール手順
1. リポジトリクローン
2. 仮想環境作成
3. 依存関係インストール
4. 初回セットアップ実行
5. サーバー起動

### 4.3 設定ファイル (config.json)
```json
{
  "model": {
    "name": "Gemma-3-1B-it",
    "repo_id": "unsloth/gemma-3-1b-it",
    "model_type": "gemma3",
    "max_context_length": 8192
  },
  "inference": {
    "max_tokens": 500,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1
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

### 5.1 NPU最適化
- モデルの事前コンパイル
- 効率的なメモリ管理
- バッチ処理の最適化

### 5.2 推論最適化
- KVキャッシュの活用
- 動的バッチサイズ調整
- プリフィルとデコードの分離

### 5.3 通信最適化
- WebSocketによる低遅延通信
- ストリーミング応答
- 効率的なJSON serialization

## 6. エラーハンドリング

### 6.1 モデル関連エラー
- モデルダウンロード失敗
- 変換エラー
- NPU利用不可

### 6.2 推論エラー
- メモリ不足
- タイムアウト
- 不正な入力

### 6.3 通信エラー
- WebSocket接続断
- ネットワークエラー
- JSON解析エラー

## 7. テスト戦略

### 7.1 単体テスト
- モデル変換機能
- 推論エンジン
- API エンドポイント

### 7.2 統合テスト
- エンドツーエンドチャット
- パフォーマンステスト
- エラーシナリオテスト

### 7.3 パフォーマンステスト
- 応答時間測定
- スループット測定
- メモリ使用量監視

---

この技術仕様書に基づいて、実際のコード実装を進めることができます。