# Gemma-3-1B-it NPU チャットボット

> **AI駆動開発サンプルプロジェクト**  
> このリポジトリは、AI技術を活用した開発プロセスの実例を示すサンプルプロジェクトです。

## 🤖 プロジェクト概要

Gemma-3-1B-itモデルをOpenVINOを使用してIntel NPU上で実行し、ブラウザから利用可能な個人用チャットボットアプリケーションを構築するプロジェクトです。

### ✨ 主な特徴

- **🚀 高速推論**: Intel NPUによる最適化されたAI推論
- **🌐 ブラウザベース**: Webインターフェースによる直感的な操作
- **🔒 完全ローカル**: プライバシーを重視した完全ローカル実行
- **⚡ リアルタイム**: WebSocketによるストリーミング応答
- **🎯 軽量**: 1Bパラメータモデルによる効率的なリソース使用

## 🏗️ AI駆動開発について

このプロジェクトは、以下のAI駆動開発手法を実践しています：

### 📋 要件定義フェーズ
- AIアシスタントによる包括的な要件分析
- ユーザーニーズの構造化と明文化
- 技術制約の特定と解決策の提案

### 🔧 設計・実装フェーズ
- アーキテクチャ設計のAI支援
- コード生成とベストプラクティスの適用
- リアルタイムな問題解決とコード最適化

### 📚 ドキュメント生成
- 自動的な技術仕様書作成
- API文書の生成
- ユーザーガイドの作成

## 📁 プロジェクト構造

```
chatbot_on_NPU/
├── docs/                          # 📖 プロジェクト文書
│   ├── requirements_definition.md # 要件定義書
│   └── technical_specification.md # 技術仕様書
├── app/                           # 🚀 アプリケーションコード
├── static/                        # 🌐 Webフロントエンド
├── models/                        # 🧠 AIモデル保存場所
├── config.json                    # ⚙️ 設定ファイル
└── requirements.txt               # 📦 Python依存関係
```

## 🚀 クイックスタート

### 前提条件
- Windows 10/11
- Python 3.9以上
- Intel NPU対応デバイス
- 8GB以上のRAM

### インストール
```bash
# リポジトリのクローン
git clone https://github.com/kazukiminemura/chatbot_on_NPU.git
cd chatbot_on_NPU

# 仮想環境の作成
python -m venv venv
venv\Scripts\activate

# 依存関係のインストール
pip install -r requirements.txt

# アプリケーションの起動
python run.py
```

### アクセス
ブラウザで `http://localhost:8000` にアクセスしてチャットボットを利用できます。

## 🛠️ 技術スタック

- **AI Framework**: OpenVINO Runtime
- **Backend**: FastAPI
- **Frontend**: HTML5/CSS3/JavaScript
- **Model**: Gemma-3-1B-it
- **Hardware**: Intel NPU
- **Communication**: WebSocket + REST API

## 📊 パフォーマンス目標

- **応答時間**: 平均2秒以内
- **生成速度**: 秒間50トークン以上
- **メモリ使用量**: 4GB以下
- **NPU使用率**: 80%以上

## 🤝 AI駆動開発の成果

このプロジェクトでは、以下の開発プロセスにAIを活用しています：

- **📝 要件定義**: 自然言語での要求からの構造化された仕様作成
- **🏗️ アーキテクチャ設計**: 最適な技術選択とシステム設計
- **💻 コード実装**: ベストプラクティスに従った高品質なコード生成
- **📚 ドキュメント作成**: 包括的で理解しやすい技術文書の自動生成
- **🔍 コードレビュー**: 継続的な品質改善とバグ検出

## 📖 ドキュメント

- [要件定義書](docs/requirements_definition.md)
- [技術仕様書](docs/technical_specification.md)

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 🌟 謝辞

このプロジェクトは、AI駆動開発の可能性を探求し、従来の開発プロセスとAI技術の融合による効率的な開発手法の実現を目指しています。

---

**AI駆動開発サンプルプロジェクト** | Made with 🤖 AI Assistance
