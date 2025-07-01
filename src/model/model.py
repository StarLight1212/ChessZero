import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from typing import Tuple, List, Dict, Optional

from ..config import Config
from ..utils import normalize_encoding


class ConvBlock(nn.Module):
    """卷积块：卷积+批归一化+激活函数"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                             padding=kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x


class ResidualBlock(nn.Module):
    """残差块：两个卷积层+残差连接"""
    
    def __init__(self, channels: int):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(channels, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout(0.3)  # 添加Dropout以减少过拟合
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.dropout(x)  # 在第一个卷积后应用Dropout
        x = self.conv2(x)
        x = self.bn(x)
        x += residual  # 残差连接
        x = F.relu(x)
        return x


class AlphaZeroNetwork(nn.Module):
    """AlphaZero网络架构，包含策略头和价值头"""
    
    def __init__(self, config: Config):
        super(AlphaZeroNetwork, self).__init__()
        
        self.config = config
        model_config = config.ModelConfig
        
        # 输入层
        self.conv_input = ConvBlock(
            model_config.input_shape[0],  # 输入通道数(114)
            model_config.filters,         # 滤波器数量
            kernel_size=3
        )
        
        # 残差层
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(model_config.filters) 
            for _ in range(model_config.residual_blocks)
        ])
        
        # 策略头
        self.policy_conv = ConvBlock(model_config.filters, 32, kernel_size=3)
        self.policy_fc = nn.Linear(32 * 8 * 8, model_config.policy_output_dim)
        
        # 价值头
        self.value_conv = ConvBlock(model_config.filters, 32, kernel_size=3)
        self.value_fc1 = nn.Linear(32 * 8 * 8, model_config.value_fc_size)
        self.value_fc2 = nn.Linear(model_config.value_fc_size, 1)
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        参数:
            x: 输入的棋盘状态 [batch_size, 119, 8, 8]
            
        返回:
            policy_logits: 策略网络输出的走子概率对数 [batch_size, 4672]
            value: 价值网络输出的局面评估 [batch_size, 1]
        """
        # 输入层
        x = self.conv_input(x)
        
        # 残差层
        for block in self.residual_blocks:
            x = block(x)
        
        # 策略头
        policy = self.policy_conv(x)
        policy = policy.view(policy.size(0), -1)  # 展平
        policy_logits = self.policy_fc(policy)
        
        # 价值头
        value = self.value_conv(x)
        value = value.view(value.size(0), -1)  # 展平
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy_logits, value


class AlphaZeroModel:
    """AlphaZero模型类，包含网络和实用方法"""
    
    def __init__(self, config: Config):
        """
        初始化AlphaZero模型
        
        参数:
            config: 配置对象
        """
        self.config = config
        self.model = AlphaZeroNetwork(config)
        
        # 设置设备
        use_cuda = torch.cuda.is_available() and self.config.config.get('system', {}).get('cuda', True)
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.model.to(self.device)
        
        # 打印模型信息
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"模型参数数量: {num_params:,}")
        print(f"使用设备: {self.device}")
        
    def predict(self, state: np.ndarray, valid_moves: List[str] = None) -> Tuple[np.ndarray, float]:
        """
        预测给定状态的策略和价值
        
        参数:
            state: 棋盘状态 [119, 8, 8]
            valid_moves: 有效移动列表，如果提供则使用normalize_encoding校验
            
        返回:
            policy: 策略网络输出的动作概率 [4672]
            value: 价值网络输出的局面评估 (标量)
        """
        self.model.eval()
        with torch.no_grad():
            # 转换为张量并添加批次维度
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # 进行预测
            policy_logits, value = self.model(state_tensor)
            
            # 将策略转换为概率分布
            policy = F.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
            
            # 如果提供了合法移动，进行校验和归一化
            if valid_moves is not None and len(valid_moves) > 0:
                policy = normalize_encoding(policy, valid_moves, self.config.move_lookup)
                
            value = value.item()
            
        return policy, value
    
    def predict_batch(self, states: np.ndarray, valid_moves_list: List[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量预测多个状态的策略和价值
        
        参数:
            states: 批量棋盘状态 [batch_size, 119, 8, 8]
            valid_moves_list: 每个状态的有效移动列表，如果提供则使用normalize_encoding校验
            
        返回:
            policies: 策略网络输出的动作概率 [batch_size, 4672]
            values: 价值网络输出的局面评估 [batch_size]
        """
        self.model.eval()
        with torch.no_grad():
            # 转换为张量
            states_tensor = torch.FloatTensor(states).to(self.device)
            
            # 进行预测
            policy_logits, values = self.model(states_tensor)
            
            # 将策略转换为概率分布
            policies = F.softmax(policy_logits, dim=1).cpu().numpy()
            
            # 如果提供了合法移动列表，对每个样本进行校验和归一化
            if valid_moves_list is not None and len(valid_moves_list) == len(states):
                for i, valid_moves in enumerate(valid_moves_list):
                    if valid_moves and len(valid_moves) > 0:
                        policies[i] = normalize_encoding(policies[i], valid_moves, self.config.move_lookup)
            
            values = values.squeeze(-1).cpu().numpy()
            
        return policies, values
    
    def save_model(self, path: str):
        """
        保存模型到指定路径
        
        参数:
            path: 保存路径
        """
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        torch.save({
            'model_state_dict': self.model.state_dict(),
        }, path)
        print(f"模型已保存到 {path}")
        
    def load_model(self, path: str) -> bool:
        """
        从指定路径加载模型
        
        参数:
            path: 模型路径
            
        返回:
            是否成功加载
        """
        if os.path.exists(path):
            try:
                checkpoint = torch.load(path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"模型已从 {path} 加载")
                return True
            except Exception as e:
                print(f"加载模型时出错: {e}")
                return False
        else:
            print(f"没有找到模型文件: {path}")
            return False
    
    def train(self):
        """将模型设置为训练模式"""
        self.model.train()
    
    def eval(self):
        """将模型设置为评估模式"""
        self.model.eval()
