# 自然语言处理作业

本仓库保存自然语言处理课程作业。作业一的原始要求和数据位于 `NLP Hw1/`。

## 环境配置

使用 Python 3.12（本次实际版本为 3.12.14），在仓库根目录执行：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

依赖版本固定在 `requirements.txt`。实验仅使用 CPU，无需 CUDA，也无需下载额外分词模型。

## 数据

- `NLP Hw1/data/nyt.csv`：新闻分类数据，包含 `text` 和 `label` 两列。
- `NLP Hw1/data/ag.csv`：供后续词向量任务使用，Task 1 不使用。
- `NLP Hw1/作业一要求.docx`：课程提供的作业说明。

原始数据随仓库保留；虚拟环境和缓存不提交。

## Git

远程仓库：<https://github.com/NKU-yxy/NLP_Homework>。

提交信息统一使用简明中文。先提交环境配置，再提交 Task 1 实现与实验结果。
