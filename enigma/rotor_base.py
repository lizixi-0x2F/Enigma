from abc import ABC, abstractmethod
import torch

class RotorBase(ABC):
    """
    转子基类，定义了所有转子类型必须实现的接口
    
    所有实现此接口的类都应该提供以下方法:
    - permute: 前向置换变换
    - inverse_permute: 逆向置换变换
    - step: 更新转子状态
    - at_notch: 检查转子是否处于缺口位置
    """
    
    @abstractmethod
    def permute(self, x: torch.Tensor) -> torch.Tensor:
        """
        执行前向置换变换
        
        参数:
            x (Tensor): 输入张量
            
        返回:
            Tensor: 置换变换后的张量
        """
        pass

    @abstractmethod
    def inverse_permute(self, x: torch.Tensor) -> torch.Tensor:
        """
        执行逆向置换变换
        
        参数:
            x (Tensor): 输入张量
            
        返回:
            Tensor: 逆置换变换后的张量
        """
        pass

    @abstractmethod
    def step(self) -> None:
        """
        更新转子状态，通常是移动转子位置
        """
        pass

    @abstractmethod
    def at_notch(self) -> bool:
        """
        检查转子是否处于缺口位置
        
        返回:
            bool: 如果转子处于缺口位置则为True，否则为False
        """
        pass 