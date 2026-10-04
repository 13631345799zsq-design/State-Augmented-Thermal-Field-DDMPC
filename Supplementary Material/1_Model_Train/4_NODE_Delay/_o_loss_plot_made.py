import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def find_latest_complete_epoch(loss_dir):
    train_epochs = set()
    for path in loss_dir.glob('Glosses_train_*.npy'):
        match = re.fullmatch(r'Glosses_train_(\d+)\.npy', path.name)
        if match:
            train_epochs.add(int(match.group(1)))

    val_epochs = set()
    for path in loss_dir.glob('Glosses_val_*.npy'):
        match = re.fullmatch(r'Glosses_val_(\d+)\.npy', path.name)
        if match:
            val_epochs.add(int(match.group(1)))
    complete_epochs = train_epochs & val_epochs
    if not complete_epochs:
        raise FileNotFoundError(f'未在 {loss_dir} 中找到成对的训练和验证 loss 文件')
    return max(complete_epochs)


def plot_loss_map(train_loss, val_loss, save_path):
    train_loss = np.asarray(train_loss).reshape(-1)
    val_loss = np.asarray(val_loss).reshape(-1)
    train_epochs = np.arange(1, len(train_loss) + 1)
    val_epochs = np.arange(1, len(val_loss) + 1)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(train_epochs, train_loss, label='Training Loss',
            color='#1f77b4', linewidth=2)
    ax.plot(val_epochs, val_loss, label='Validation Loss',
            color='#d62728', linewidth=2)
    ax.set_yscale('log')
    ax.set_title('Training Loss History (Log Scale)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Composite Loss (Log Scale)', fontsize=12)
    ax.set_ylim(1e-4, 1e-1)
    ax.grid(True, which='both', linestyle='--', alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Loss 曲线已保存至: {save_path}')


def main():
    parser = argparse.ArgumentParser(description='绘制稳态模型训练/验证 loss 曲线')
    parser.add_argument('--fold', type=int, default=1, help='fold 编号，默认 1')
    parser.add_argument(
        '--epoch', type=int, default=None,
        help='读取指定 epoch 的快照；默认自动选择最新的成对文件',
    )
    parser.add_argument(
        '--output', default='Loss_Curve_Analysis.png',
        help='输出图片文件名',
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    fold_dir = script_dir / f'fold_result_{args.fold}'
    loss_dir = fold_dir / 'losses_1'
    epoch = args.epoch if args.epoch is not None else find_latest_complete_epoch(loss_dir)

    train_path = loss_dir / f'Glosses_train_{epoch}.npy'
    val_path = loss_dir / f'Glosses_val_{epoch}.npy'
    if not train_path.is_file() or not val_path.is_file():
        raise FileNotFoundError(f'epoch {epoch} 的训练/验证 loss 文件不完整: {loss_dir}')

    print(f'读取 epoch {epoch} 的 loss 快照')
    train_loss = np.load(train_path)
    val_loss = np.load(val_path)
    plot_loss_map(train_loss, val_loss, fold_dir / args.output)


if __name__ == '__main__':
    main()
