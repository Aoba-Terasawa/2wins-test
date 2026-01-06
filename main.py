import torch
from anomalib.data import Folder
from anomalib.models import EfficientAd
from anomalib.engine import Engine
from anomalib.metrics import Evaluator, create_anomalib_metric, AUROC, F1Score
from torchmetrics.classification import BinaryRecall
from my_utils import fix_seed

# 設定
DATA_DIR = r"C:\Users\user\2wins-test\dataset"
MAX_EPOCHS = 200
LEARNING_RATE = 0.0001
NORMAL_SPLIT_RATIO = 0.2
VAL_SPLIT_RATIO = 0.5
TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
NUM_WORKERS = 0
AnomalibRecall = create_anomalib_metric(BinaryRecall)

if __name__ == '__main__':
    fix_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    datamodule = Folder(
        name="defect_detection",
        root=DATA_DIR,
        normal_dir="good",
        abnormal_dir="bad",
        normal_split_ratio=NORMAL_SPLIT_RATIO,   
        val_split_ratio=VAL_SPLIT_RATIO,     
        train_batch_size=TRAIN_BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )
    datamodule.setup()

    test_metrics = [
        AnomalibRecall(fields=["pred_label", "gt_label"], prefix="image_"),
        AUROC(fields=["pred_score", "gt_label"], prefix="image_"),
        F1Score(fields=["pred_label", "gt_label"], prefix="image_"),
    ]
    evaluator = Evaluator(test_metrics=test_metrics)  

    model = EfficientAd(lr=LEARNING_RATE, evaluator=evaluator)
    engine = Engine(max_epochs=MAX_EPOCHS, default_root_dir="results")

    engine.fit(model=model, datamodule=datamodule)

    results_list = engine.test(model=model, datamodule=datamodule)
    results = results_list[0] if isinstance(results_list, list) and len(results_list) else results_list

    print("\n==== TEST METRICS ====")
    print("keys:", list(results.keys()))
    print("image Recall:", results.get("image_BinaryRecall"))
    print("image AUROC :", results.get("image_AUROC"))
    print("image F1    :", results.get("image_F1Score"))
