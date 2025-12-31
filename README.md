# 2wins-test 不良品識別モデル構築プロジェクト

## 1. 目的
製造ラインにおける外観検査の自動化を目的とした、二値分類モデルの構築を行う。

## 2. 目標・KPI
製造現場における不良品の流出（見逃し）を最小限に抑えるという運用条件を最優先とする。
* **メインKPI**: 不良品（bad）の再現率（Recall）$1.00$
* **設計指針**: モデルが判断に迷うグレーゾーンの個体については、積極的に「不良品（bad）」側へ倒すことで、安全性を最大化した検品を実現する。

## 3. 実行環境
以下の手順で環境構築を行ってください。

### Dependency（依存ライブラリ）
* **Python**: 3.10+
* **主要ライブラリ**: PyTorch, torchvision, scikit-learn, Pillow, matplotlib, seaborn

### セットアップ手順
```powershell
# 1. 仮想環境の作成と有効化
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2. 依存ライブラリの一括インストール
pip install -r requirements.txt

# 3. 実行
python main.py
