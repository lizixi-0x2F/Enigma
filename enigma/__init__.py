"""
Enigma - 可逆动态置换网络

一个基于神经网络的可逆动态置换网络，其设计灵感来源于历史上著名的Enigma密码机。
"""

from enigma.model import Enigma, EnigmaLM
from enigma.plugboard import Plugboard
from enigma.reflector import Reflector
from enigma.rev_block import RevBlock
from enigma.rotor import Rotor, RotorStack
from enigma.invertible_conv1x1 import InvertibleConv1x1, InvertibleConv1x1Stack, DynamicInvertibleConv1x1Stack
from enigma.modeling_enigma import EnigmaConfig, EnigmaForCausalLM

__version__ = "0.1.0"

__all__ = [
    'Enigma',
    'EnigmaLM',
    'Plugboard',
    'Reflector',
    'RevBlock',
    'Rotor',
    'RotorStack',
    'InvertibleConv1x1',
    'InvertibleConv1x1Stack',
    'DynamicInvertibleConv1x1Stack',
    'EnigmaConfig',
    'EnigmaForCausalLM',
] 