# 最终实验结果

| 任务 | 最终结果 | 要求 | 状态 |
|---|---:|---:|---|
| Task 1 MNIST 分类 | Test accuracy 92.45% | > 90% | 达标 |
| Task 2 二值自编码器 | Train PSNR 26.23 dB；Test PSNR 15.91 dB | PSNR > 25 dB | 训练集达标，测试集未达标 |
| Task 3.1 Sequence-to-Token | 100% exact memorization | > 80% | 达标 |
| Task 3.2 Sequence-to-Sequence | 100% exact memorization | > 80% | 达标 |
| Task 3.3 Sequence-to-Histogram | 99.90% exact memorization，使用 1,024 张图像 | > 99%，至少 512 张图像 | 达标 |
| Task 4.1 Continuous AE | Test PSNR 30.30 dB，latent 5 x 8 x 8 | > 25 dB | 达标 |
| Task 4.1 Generator AE | Test PSNR 29.09 dB，latent 5 x 8 x 8 | > 25 dB | 达标 |
| Task 4.1 VQ-VAE | Test PSNR 25.24 dB，8 x 8 indices，10-D embeddings | > 25 dB | 达标 |
| Task 4.2 Discrete latent diffusion | FID 37.4657，10,000 张生成图像 | FID < 40 | 达标 |
| Task 4.3 Continuous latent Transformer | FID 28.8195；独立种子 28.8515 | FID < 40 | 达标 |

这些数字与 `report/assignment_report.pdf` 中的最终汇总一致。Task 4 的 FID 使用 10,000 张生成图像与 CIFAR-10 官方 10,000 张测试图像，通过标准 `pytorch-fid` 0.3.0 的 2,048 维 Inception 特征计算。

