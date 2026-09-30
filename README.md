# 自然语言处理作业

本仓库保存自然语言处理课程作业。作业一的原始要求和数据位于 `NLP Hw1/`。

## 环境

使用 Python 3.12，在仓库根目录创建虚拟环境并安装依赖：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## 作业一运行

数据文件放在 `NLP Hw1/data/`。各任务的实验代码位于 `NLP Hw1/src/`，结构化结果保存到 `NLP Hw1/results/`。

```bash
.venv/bin/python "NLP Hw1/src/task1.py"
.venv/bin/python "NLP Hw1/src/prepare_glove.py"
.venv/bin/python "NLP Hw1/src/task2.py"
.venv/bin/python "NLP Hw1/src/task3.py"
```

Task 2 的 GloVe 文件首次运行时需要下载；Task 3 的 BERT 预训练模型首次运行时会自动下载。两者均保存在不被 Git 追踪的 `data/` 目录。Task 2 和 Task 3 复用 Task 1 保存的 NYT 数据划分。
