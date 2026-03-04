# SocialOmni：面向 Omni 模型的音视频社会交互基准

<p align="center">
  <img src="assets/hero.svg" alt="SocialOmni Hero" width="100%" />
</p>

<p align="center">
  <a href="../README.md">English</a>
  ·
  <a href="#快速开始">快速开始</a>
  ·
  <a href="#基准概览">基准概览</a>
  ·
  <a href="#评测协议">评测协议</a>
</p>

SocialOmni 是一个评测 Omni 模型**音视频社会交互能力**的基准。
与只关注最终答案正确性的评测不同，SocialOmni 显式评估交互三要素：

- **Who**：谁在说话（说话人识别）
- **When**：什么时候该打断（时机判断）
- **How**：如何自然回应（打断生成）

本仓库提供统一评测流水线、模型客户端/服务端以及可复现实验入口。

## 为什么是 SocialOmni

现有多模态基准大多偏向“理解型评测”（静态问答、最终答案准确率）。
SocialOmni 关注动态对话中的关键瓶颈：模型是否能在正确时机做出社会化交互行为。

在真实对话中，语义正确并不等于交互成功。过早/过晚打断，或不自然的续说，都会显著影响体验。

## 基准概览

<p align="center">
  <img src="assets/socialomni_overview.png" alt="SocialOmni Overview" width="100%" />
</p>

### 数据组成

- 总样本：**2,209**（来自 **2,000** 个短视频）
- 感知任务：2,000 条时间戳级别问答
- 生成任务：209 条打断式交互样本
- 覆盖 **15** 个对话领域
- 感知任务中 A-V 一致性划分：
  - 一致：86.25%
  - 不一致：13.75%

### 标注质量

- 双轮专家复核
- 标注一致性（IAA）：
  - 感知任务：**94.2%**
  - 生成任务：**91.8%**

## 任务定义

### 任务一：感知（Who）

给定视频与时间点 `t`，回答：

> “在时间点 `t`，谁在说话？”

模型从 `{A, B, C, D}` 中选择一个选项。

### 任务二：生成（When + How）

给定视频前缀 `V[0:t]` 与候选说话者 `X`，模型完成：

- **Q1（When）**：判断 `X` 是否应在 `t` 后立即打断
- **Q2（How）**：若 Q1 预测应打断，生成自然的打断内容

## 评测协议

### 感知任务指标

- Top-1 Accuracy（总体）
- 一致/不一致子集准确率
- Gap：`Δ = Acc_consistent - Acc_inconsistent`

### 生成任务指标

- **Q1**：在容忍窗口（如 δ=0.2s）下计算 Accuracy / Precision / Recall / F1
- **Q2**：在 `{0, 25, 50, 75, 100}` 打分集上的评委评分

论文协议中的 Q2 默认三评委：

- GPT-4o
- Gemini 3 Pro
- Qwen3-Omni

## 主要结果（论文表）

| 模型 | 感知总体(%) | Q1 准确率(%) | Q2 分数(/100) |
|---|---:|---:|---:|
| Gemini 3 Pro Preview | 64.99 | **66.99** | 81.77 |
| Qwen3-Omni | **69.25** | 63.64 | 45.57 |
| Gemini 2.5 Flash | 47.03 | 58.85 | **85.08** |
| GPT-4o | 36.75 | 46.89 | 69.64 |

结论：感知能力排名与生成质量并非严格一致，说明 who/when/how 联合评测是必要的。

## 仓库结构

```text
SocialOmni/
├── models/                  # 模型服务、客户端与共享流水线
├── config/                  # 运行时/模型/评测配置
├── data/                    # 本地数据（默认不入库）
├── results/                 # 本地输出（默认不入库）
├── scripts/                 # 工具脚本
├── docs/                    # 文档与可视化资源
├── run_benchmark.py         # 任务一（Level1）入口
├── run_benchmark_level2.py  # 任务二（Level2）入口
└── README.md
```

## 快速开始

### 1) 克隆并安装

```bash
git clone https://github.com/Alexisxty/SocialOmni.git
cd SocialOmni
uv sync
```

### 2) 配置运行参数

编辑 `config/config.yaml`，至少配置：

- API 地址与密钥（或环境变量）
- 本地模型路径与 `server_url`
- 数据集路径与输出路径

支持环境变量覆盖，例如：

- `OPENAI_API_KEY` / `OPENAI_API_BASE`
- `GEMINI_API_KEY` / `GEMINI_API_BASE`

### 3) 启动本地模型服务（示例）

```bash
uv run models/model_server/qwen3_omni/qwen3_omni_server.py
```

其他服务入口位于 `models/model_server/*/*_server.py`。

### 4) 运行评测

任务一（感知）：

```bash
uv run run_benchmark.py --model qwen3_omni
```

任务二（生成）：

```bash
uv run run_benchmark_level2.py --model qwen3_omni --resume
```

## 支持的模型键

命令行 `--model` 可选值：

`gpt4o`, `gemini_2_5_flash`, `gemini_2_5_pro`, `gemini_3_flash_preview`, `gemini_3_pro_preview`, `qwen3_omni`, `qwen3_omni_thinking`, `qwen2_5_omni`, `miniomni_2`, `omnivinci`, `vita_1_5`, `baichuan_omni_1_5`, `ming`

## 可复现建议

- 数据与输出目录建议保持本地，不纳入版本库
- 跨模型比较时使用固定提示词与固定配置
- 报告改进时建议同时给出置信区间与子集指标

## 引用

如果你在研究中使用 SocialOmni，请引用论文：

```bibtex
@article{socialomni2026,
  title={SocialOmni: Benchmarking Audio-Visual Social Interactivity in Omni Models},
  author={Anonymous},
  journal={ECCV},
  year={2026}
}
```

（请在最终公开版本中替换为 camera-ready 的正式元数据。）

## 许可与数据使用

- 代码与评测协议：以仓库 License 为准（待最终发布）
- 视频资产与元数据：遵循原始来源许可，使用时请合规
