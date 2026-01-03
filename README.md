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
```

## 4. 結果
### Validデータの混同行列 (Confusion Matrix)
| | 予測: bad | 予測: good |
| :--- | :---: | :---: |
| **正解: bad** | **TP** 52 | **FN** 0 |
| **正解: good** | **FP** 1 | **TN** 149 |

### Validデータのスコア
| 指標 | スコア |
| :--- | :--- |
| **Accuracy（正解率）** | 0.99 |
| **Recall（bad）** | 1.00 |


### Testデータの混同行列 (Confusion Matrix)
| | 予測: bad | 予測: good |
| :--- | :---: | :---: |
| **正解: bad** | **TP** 49 | **FN** 4 |
| **正解: good** | **FP** 0 | **TN** 150 |

### Testデータのスコア
| 指標 | スコア |
| :--- | :--- |
| **Accuracy（正解率）** | 0.98 |
| **Recall（bad）** | 0.92 |

## 5. 分析
### 成功の場合(予測 bad 結果 bad)
<img src="analysis_results_example/resized_1.png" width="48%">
<img src="analysis_results_example/gradcam_1.png" width="48%">

### 失敗の場合１(予測 good 結果 bad) 傷の場所を捉えられていない
<img src="analysis_results_example/resized_3.png" width="48%">
<img src="analysis_results_example/gradcam_3.png" width="48%">

### 失敗の場合２(予測 good 結果 bad) 傷の場所を捉えているが、分類ミス
<img src="analysis_results_example/resized_5.png" width="48%">
<img src="analysis_results_example/gradcam_5.png" width="48%">

