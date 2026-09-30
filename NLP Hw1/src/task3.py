"""微调 BERT-base-uncased 并在固定 NYT 测试集上评价分类效果。"""

import argparse
from collections import Counter
import hashlib
import math
import os
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # 避免训练后子进程重复初始化分词线程。
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
import transformers

from task1 import ROOT, SEED, evaluate, get_split, load_data, save_json


MODEL_NAME = "google-bert/bert-base-uncased"  # 作业指定的预训练模型。
MAX_LENGTH = 64  # 作业指定的最大序列长度，包含特殊标记。
EPOCHS = 3  # 作业指定的完整训练轮数。
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01


def file_sha256(path):
    """分块计算原始数据和划分文件的指纹。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def choose_device(requested):
    """按用户指定或本机可用情况选择训练设备。"""
    if requested == "auto":
        requested = "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("本机 PyTorch 无法使用 MPS，请改用 --device cpu。")
    return torch.device(requested)


def make_dataset(texts, labels, indices, tokenizer, label_to_id):
    """按固定行索引构建长度为六十四的分类张量数据集。"""
    selected_texts = [str(texts[int(index)]) for index in indices]
    encoded = tokenizer(selected_texts, truncation=True, padding="max_length",
                        max_length=MAX_LENGTH, return_tensors="pt")
    class_ids = torch.tensor([label_to_id[str(labels[int(index)])] for index in indices],
                             dtype=torch.long)
    if encoded["input_ids"].shape != (len(indices), MAX_LENGTH):
        raise ValueError("分词器没有生成预期长度的输入张量。")
    return TensorDataset(encoded["input_ids"], encoded["attention_mask"], class_ids)


def evaluate_model(model, loader, device, classes):
    """在指定数据集上计算损失、预测与分类评价指标。"""
    model.eval()
    total_loss = 0.0
    predictions, truth = [], []
    with torch.inference_mode():
        for input_ids, attention_mask, labels in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += output.loss.item() * len(labels)
            predictions.extend(output.logits.argmax(dim=-1).cpu().tolist())
            truth.extend(labels.cpu().tolist())
    if not math.isfinite(total_loss):
        raise ValueError("评价损失出现非有限数值。")
    result = evaluate([classes[index] for index in truth],
                      [classes[index] for index in predictions], classes)
    result["loss"] = total_loss / len(truth)
    return result, predictions


def train_epoch(model, loader, optimizer, scheduler, device, epoch):
    """完整遍历一次 NYT 训练集并更新 BERT 全部参数。"""
    model.train()
    total_loss = 0.0
    for step, (input_ids, attention_mask, labels) in enumerate(loader, 1):
        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids=input_ids.to(device),
                       attention_mask=attention_mask.to(device), labels=labels.to(device))
        if not torch.isfinite(output.loss).item():
            raise ValueError(f"第 {epoch} 轮第 {step} 步的训练损失无效。")
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += output.loss.detach().item() * len(labels)
        if step % 100 == 0 or step == len(loader):
            print(f"第 {epoch}/{EPOCHS} 轮：已完成 {step}/{len(loader)} 批", flush=True)
    return total_loss / len(loader.dataset)


def main():
    """执行三轮预训练 BERT 微调并保存结构化测试结果。"""
    parser = argparse.ArgumentParser(description="运行实验一的 BERT 文本分类。")
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto",
                        help="训练设备；默认优先使用 Mac GPU")
    args = parser.parse_args()
    data_path = ROOT / "data/nyt.csv"
    split_path = ROOT / "results/nyt_split.json"
    if not split_path.is_file():
        raise FileNotFoundError("缺少 Task 1 的固定数据划分文件。")
    texts, labels = load_data(data_path)
    splits = get_split(split_path, len(texts), file_sha256(data_path))
    classes = sorted(set(labels))
    if set(labels[splits["train"]]) != set(classes):
        raise ValueError("训练集没有覆盖全部类别。")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = choose_device(args.device)
    print(f"使用 {device} 微调 {MODEL_NAME}，最大长度 {MAX_LENGTH}，共 {EPOCHS} 轮", flush=True)
    cache_dir = ROOT / "data/hf_cache"  # 预训练参数仅缓存在 Git 忽略的数据目录。
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, cache_dir=cache_dir)
    label_to_id = {label: index for index, label in enumerate(classes)}
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(classes), id2label=dict(enumerate(classes)),
        label2id=label_to_id, cache_dir=cache_dir)
    model.to(device)

    datasets = {part: make_dataset(texts, labels, indices, tokenizer, label_to_id)
                for part, indices in splits.items()}
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(datasets["train"], batch_size=TRAIN_BATCH_SIZE,
                              shuffle=True, generator=generator)
    validation_loader = DataLoader(datasets["validation"], batch_size=EVAL_BATCH_SIZE)
    test_loader = DataLoader(datasets["test"], batch_size=EVAL_BATCH_SIZE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                 num_training_steps=total_steps)
    history = []
    start = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        training_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        validation, _ = evaluate_model(model, validation_loader, device, classes)
        history.append({"epoch": epoch, "train_loss": training_loss,
                        "validation": validation})
        print(f"第 {epoch} 轮验证：Accuracy={validation['accuracy']:.6f}，"
              f"Macro-F1={validation['macro_f1']:.6f}", flush=True)
    test, predicted_ids = evaluate_model(model, test_loader, device, classes)
    elapsed = time.perf_counter() - start

    results = {
        "model": MODEL_NAME, "model_revision": model.config._commit_hash,
        "tokenizer_fast": tokenizer.is_fast, "max_length": MAX_LENGTH,
        "epochs_completed": len(history), "seed": SEED, "device": str(device),
        "batch_size_train": TRAIN_BATCH_SIZE, "batch_size_eval": EVAL_BATCH_SIZE,
        "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
        "scheduler": "linear", "warmup_steps": 0, "optimizer": "AdamW",
        "gradient_clip_norm": 1.0, "training_steps": total_steps,
        "elapsed_training_and_evaluation_seconds": elapsed,
        "classes": classes,
        "split_counts": {part: dict(Counter(labels[index])) for part, index in splits.items()},
        "input_hashes": {"nyt_sha256": file_sha256(data_path),
                         "split_sha256": file_sha256(split_path)},
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        "torch": torch.__version__, "transformers": transformers.__version__},
        "history": history, "test": test,
    }
    save_json(ROOT / "results/task3_metrics.json", results)
    save_json(ROOT / "results/task3_predictions.json", {
        "row_indices": splits["test"].tolist(),
        "true_labels": labels[splits["test"]].tolist(),
        "predicted_labels": [classes[index] for index in predicted_ids],
    })
    print(f"NYT Test：Accuracy={test['accuracy']:.6f}，Macro-F1={test['macro_f1']:.6f}",
          flush=True)


if __name__ == "__main__":
    main()
