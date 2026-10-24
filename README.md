# MM811 Assignment 最终整理版

这个文件夹只保留最终报告对应的实验代码、必要的辅助模块、依赖说明和报告文件。原项目中的早期尝试、失败版本、大型模型权重、数据集缓存以及 10,000 张生成图片均未复制。

## 建议阅读顺序

1. 阅读 `results/FINAL_RESULTS.md`，先了解每项任务的最终指标。
2. 阅读 `report/assignment_report.pdf`，查看完整实验设计、结构图、训练过程和结果分析。
3. 按 Task 1 到 Task 4 的顺序阅读 `code/` 中的程序。
4. 需要重新运行时参考 `RUN_COMMANDS.md`。

## 目录结构

```text
811 assignment/
├── README.md
├── RUN_COMMANDS.md
├── code/                 最终实验代码，放在同一目录以保持 Python import 正常
├── report/               LaTeX 源文件、图片资源和最终 PDF
├── requirements/         原始依赖文件与合并依赖文件
└── results/              最终指标汇总和少量可核对结果
```

## 代码与任务对应关系

| 任务 | 主程序 | 辅助程序 |
|---|---|---|
| Task 1 | `task1_classroom.py` | 无 |
| Task 2 | `task2_optimized.py` | 无 |
| Task 3 | `task3_autoregressive.py` | `task2_optimized.py` 提供二值编码器结构；`task3_resample_histogram.py` 用于温度消融 |
| Task 4.1 | `task4_autoencoders.py`, `task4_vqvae_masked.py` | `task4_models.py`, `task4_vqvae_simple.py` |
| Task 4.2 | `task4_vq_diffusion_final.py` | `task4_vq_diffusion_gemini.py`, `task4_vqvae_simple.py` |
| Task 4.3 | `task4_continuous_ae_perceptual.py`, `task4_continuous_token_diffusion.py` | `task4_models.py` |
| FID 评估 | `task4_fid_standard.py` | `task4_models.py` |

## 说明

- Task 2 的 48-bit 二值瓶颈在 256 张训练图像上超过 25 dB，但在独立测试集上没有超过 25 dB；报告中明确保留了这个限制。
- Task 4.2 的最终 FID 为 37.4657，满足 FID 小于 40。
- Task 4.3 的最终 FID 为 28.8195，独立种子复现结果为 28.8515，均满足要求。
- 训练 checkpoint 很大且包含服务器绝对路径，因此没有放进本阅读文件夹。重新训练时程序会在自己的输出目录生成新 checkpoint。

