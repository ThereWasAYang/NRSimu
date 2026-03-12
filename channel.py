"""
NR Channel Module
5G NR 信道模块：包括AWGN、多径衰落、多普勒效应
"""

import numpy as np
from typing import Optional, Tuple
from scipy import signal as scipy_signal


class NRChannel:
    """5G NR 无线信道仿真"""
    
    def __init__(self,
                 snr_db: float = 20.0,
                 channel_type: str = 'awgn',
                 delay_spread: float = 100e-9,  # 100 ns
                 doppler_freq: float = 100.0,   # 100 Hz
                 sampling_rate: float = 30.72e6):  # 30.72 MHz
        """
        初始化信道参数
        
        Args:
            snr_db: 信噪比 (dB)
            channel_type: 信道类型 ('awgn', 'rayleigh', 'rician', 'tdl')
            delay_spread: 时延扩展 (秒)
            doppler_freq: 最大多普勒频移 (Hz)
            sampling_rate: 采样率 (Hz)
        """
        self.snr_db = snr_db
        self.channel_type = channel_type
        self.delay_spread = delay_spread
        self.doppler_freq = doppler_freq
        self.sampling_rate = sampling_rate
        
        # 计算噪声功率
        self.noise_power = 10**(-snr_db/10)
        
    def awgn(self, signal: np.ndarray) -> np.ndarray:
        """
        添加加性高斯白噪声 (AWGN)
        
        Args:
            signal: 输入信号
            
        Returns:
            加噪后的信号
        """
        signal_power = np.mean(np.abs(signal)**2)
        noise_power_linear = signal_power * self.noise_power
        
        # 生成复高斯噪声
        noise = np.sqrt(noise_power_linear/2) * (np.random.randn(len(signal)) + 
                                                  1j*np.random.randn(len(signal)))
        
        return signal + noise
    
    def rayleigh_fading(self, num_samples: int) -> np.ndarray:
        """
        生成瑞利衰落信道响应
        
        Args:
            num_samples: 采样点数
            
        Returns:
            信道冲激响应
        """
        # Jakes模型简化实现
        num_paths = 16
        
        h = np.zeros(num_samples, dtype=complex)
        
        for i in range(num_paths):
            # 随机相位
            phi = 2*np.pi*np.random.rand()
            # 多普勒频移
            fd = self.doppler_freq * np.cos(2*np.pi*i/num_paths)
            # 叠加
            h += np.exp(1j*phi) * np.exp(1j*2*np.pi*fd*np.arange(num_samples)/self.sampling_rate)
        
        # 归一化
        h = h / np.sqrt(num_paths)
        
        return h
    
    def tdl_channel(self, signal: np.ndarray) -> np.ndarray:
        """
        TDL (Tapped Delay Line) 信道模型
        基于3GPP TR 38.901的TDL-A模型
        
        Args:
            signal: 输入信号
            
        Returns:
            经过多径衰落的信号
        """
        # TDL-A模型参数 (简化版)
        delays = np.array([0, 10, 20, 30, 40, 50]) * 1e-9  # 延迟 (秒)
        powers_db = np.array([0, -3, -6, -9, -12, -15])    # 功率 (dB)
        
        # 转换为采样点
        delay_samples = (delays * self.sampling_rate).astype(int)
        powers_linear = 10**(powers_db/10)
        
        max_delay = np.max(delay_samples)
        output = np.zeros(len(signal) + max_delay, dtype=complex)
        
        # 对每个路径应用衰落并叠加
        for delay, power in zip(delay_samples, powers_linear):
            # 生成衰落系数
            fading = (np.random.randn(len(signal)) + 1j*np.random.randn(len(signal))) / np.sqrt(2)
            faded_signal = signal * fading * np.sqrt(power)
            output[delay:delay+len(signal)] += faded_signal
        
        return output[:len(signal)]
    
    def apply_channel(self, tx_signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        应用信道效应
        
        Args:
            tx_signal: 发射信号
            
        Returns:
            (接收信号, 信道响应)
        """
        rx_signal = tx_signal.copy()
        h = np.ones(len(tx_signal), dtype=complex)  # 默认信道响应
        
        if self.channel_type == 'awgn':
            # 仅AWGN
            pass
            
        elif self.channel_type == 'rayleigh':
            # 瑞利衰落
            h = self.rayleigh_fading(len(tx_signal))
            rx_signal = rx_signal * h
            
        elif self.channel_type == 'tdl':
            # TDL多径信道
            rx_signal = self.tdl_channel(rx_signal)
            
        elif self.channel_type == 'rician':
            # 莱斯衰落 (简化实现)
            k_factor = 4  # K因子 (dB)
            k_linear = 10**(k_factor/10)
            
            # LOS分量
            los = np.sqrt(k_linear/(k_linear+1)) * np.ones(len(tx_signal))
            # NLOS分量
            nlos = np.sqrt(1/(k_linear+1)) * self.rayleigh_fading(len(tx_signal))
            
            h = los + nlos
            rx_signal = rx_signal * h
        
        # 添加AWGN噪声
        rx_signal = self.awgn(rx_signal)
        
        return rx_signal, h
    
    def set_snr(self, snr_db: float):
        """动态设置信噪比"""
        self.snr_db = snr_db
        self.noise_power = 10**(-snr_db/10)
    
    def calculate_ber(self, tx_bits: np.ndarray, rx_bits: np.ndarray) -> float:
        """计算误码率"""
        errors = np.sum(tx_bits != rx_bits)
        return errors / len(tx_bits)
    
    def calculate_snr(self, signal: np.ndarray, noise: np.ndarray) -> float:
        """计算信噪比 (dB)"""
        signal_power = np.mean(np.abs(signal)**2)
        noise_power = np.mean(np.abs(noise)**2)
        return 10 * np.log10(signal_power / noise_power)


if __name__ == '__main__':
    # 测试信道
    channel = NRChannel(snr_db=20, channel_type='awgn')
    
    # 生成测试信号
    tx_signal = np.random.randn(10000) + 1j*np.random.randn(10000)
    tx_signal = tx_signal / np.sqrt(2)
    
    # 通过信道
    rx_signal, h = channel.apply_channel(tx_signal)
    
    print(f"发射信号功率: {np.mean(np.abs(tx_signal)**2):.4f}")
    print(f"接收信号功率: {np.mean(np.abs(rx_signal)**2):.4f}")
    print(f"信道响应均值: {np.mean(np.abs(h)):.4f}")
