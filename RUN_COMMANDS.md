# 运行命令

先创建环境、安装依赖并进入代码目录：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements/requirements-all.txt
cd code
```

## Task 1

```bash
python task1_classroom.py
```

## Task 2

```bash
python task2_optimized.py --activation relu --epochs 250
```

## Task 3

先将 Task 2 产生的 checkpoint 路径填入命令：

```bash
python task3_autoregressive.py \
  --checkpoint "/path/to/task2_checkpoint.pt" \
  --generate 1000 \
  --temperature 0.2
```

使用已有 Task 3 输出重新测试 Sequence-to-Histogram 的采样温度：

```bash
python task3_resample_histogram.py \
  --run "/path/to/task3_run" \
  --temperature 0.7 \
  --generate 1000
```

## Task 4.1

连续和基础离散自编码器：

```bash
python task4_autoencoders.py --epochs 100
```

VQ-VAE：

```bash
python task4_vqvae_masked.py --epochs 150
```

## Task 4.2

将上一步得到的 VQ-VAE checkpoint 填入命令：

```bash
python task4_vq_diffusion_final.py \
  --vqvae "/path/to/vqvae.pt" \
  --epochs 200 \
  --augmentation pixel-flip \
  --generate 10000
```

## Task 4.3

先使用 Task 4.1 的连续 AE checkpoint 进行感知损失微调：

```bash
python task4_continuous_ae_perceptual.py \
  --init-ae "/path/to/continuous_autoencoder.pt" \
  --output "outputs/task4_continuous_ae_perceptual" \
  --epochs 30 \
  --lpips-weight 0.08 \
  --latent-noise 0.1
```

然后训练连续 latent Transformer 和 per-token diffusion head：

```bash
python task4_continuous_token_diffusion.py \
  --autoencoder "/path/to/perceptual_autoencoder.pt" \
  --output "outputs/task4_continuous_token_diffusion" \
  --epochs 50 \
  --augmentation pixel-flip \
  --sample-steps 50 \
  --generate 10000
```

## 标准 FID

```bash
python task4_fid_standard.py "/path/to/generated_images.pt" --images 10000
```

## 编译报告

在 `811 assignment` 目录运行：

```bash
tectonic report/assignment_report.tex --outdir report
```

