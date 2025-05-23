import torch
import pytest
from enigma.plugboard import Plugboard

def test_index_plugboard_equivalence():
    d = 16
    # 随机生成置换
    perm = torch.randperm(d)
    pb = Plugboard(d)
    pb.set_permutation(perm)
    # 构造测试输入
    x = torch.randn(4, d)
    # 矩阵版结果
    mat = torch.zeros(d, d, device=x.device)
    mat[torch.arange(d), perm] = 1.0
    y_mat = x @ mat
    # 索引版结果
    y_idx = pb(x)
    assert torch.allclose(y_mat, y_idx)

def test_inverse_consistency():
    d = 16
    # 随机生成置换
    perm = torch.randperm(d)
    pb = Plugboard(d)
    pb.set_permutation(perm)
    # 测试输入
    x = torch.randn(4, d)
    # 前向传播
    y = pb(x)
    # 逆向传播
    x_rec = pb.inverse(y)
    # 验证重构误差
    assert torch.allclose(x, x_rec, atol=1e-6)

def test_identity():
    d = 16
    pb = Plugboard(d)
    # 设置为恒等变换
    pb.freeze_identity()
    # 测试输入
    x = torch.randn(4, d)
    # 前向传播
    y = pb(x)
    # 验证不变
    assert torch.allclose(x, y) 