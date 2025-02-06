# 3SFC
The official code for "E-3SFC: Communication-Efficient Federated Learning with Double-way Features Synthesizing", TNNLS 2025

## Abstract
> The exponential growth in model sizes has significantly increased the communication burden in Federated Learning (FL). Existing methods to alleviate this burden by transmitting compressed gradients often face high compression errors, which slow down the model's convergence. To simultaneously achieve high compression effectiveness and lower compression errors, we study the gradient compression problem from a novel perspective. Specifically, we propose a systematical algorithm termed Extended Single-Step Synthetic Features Compressing (E-3SFC), which consists of three sub-components, i.e., the Single-Step Synthetic Features Compressor (3SFC), a double-way compression algorithm, and a communication budget scheduler. First, we regard the process of gradient computation of a model as decompressing gradients from corresponding inputs, while the inverse process is considered as compressing the gradients. Based on this, we introduce a novel gradient compression method termed 3SFC, which utilizes the model itself as a decompressor, leveraging training priors such as model weights and objective functions. 3SFC compresses raw gradients into tiny synthetic features in a single-step simulation, incorporating error feedback to minimize overall compression errors. To further reduce communication overhead, 3SFC is extended to E-3SFC, allowing double-way compression and dynamic communication budget scheduling. Our theoretical analysis under both strongly convex and non-convex conditions demonstrates that 3SFC achieves linear and sub-linear convergence rates with aggregation noise. Extensive experiments across six datasets and six models reveal that 3SFC outperforms state-of-the-art methods by up to 13.4% while reducing communication costs by 111.6 times. These findings suggest that 3SFC can significantly enhance communication efficiency in FL without compromising model performance.

## Examples

Train MLP with MNIST using 3SFC on a cluster containing 10 clients.

```bash
python main.py --method ours --n_client 10 --n_epoch 200 --n_client_epoch 5 --dataset mnist --batch_size 64 --lr 1e-2 --model mlp --ours_n_sample 1
```

## Citation
@misc{zhou2025e3sfccommunicationefficientfederatedlearning,
      title={E-3SFC: Communication-Efficient Federated Learning with Double-way Features Synthesizing}, 
      author={Yuhao Zhou and Yuxin Tian and Mingjia Shi and Yuanxi Li and Yanan Sun and Qing Ye and Jiancheng Lv},
      year={2025},
      eprint={2502.03092},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.03092}, 
}
