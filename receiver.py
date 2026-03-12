"""
NR Receiver Module
5G NR 接收机模块：包括同步、信道估计、均衡、解调、解码
"""

import numpy as np
from typing import Tuple, Optional
from scipy import signal as scipy_signal


class NRReceiver:
    """5G NR 接收机"""
    
    def __init__(self,
                 num_subcarriers: int = 1200,
                 cp_length: int = 72,
                 modulation: str = 'QPSK'):
        """
        初始化接收机参数
        
        Args:
            num_subcarriers: 子载波数量
            cp_length: 循环前缀长度
            modulation: 调制方式
        """
        self.num_subcarriers = num_subcarriers
        self.cp_length = cp_length
        self.modulation = modulation
        
        self.bits_per_symbol = {
            'QPSK': 2,
            '16QAM': 4,
            '64QAM': 6,
            '256QAM': 8
        }.get(modulation, 2)
        
        # 同步相关
        self.sync_threshold = 0.8
        
    def time_sync(self, rx_signal: np.ndarray, preamble: np.ndarray) -> int:
        """
        时间同步：基于前导序列的互相关
        
        Args:
            rx_signal: 接收信号
            preamble: 已知前导序列
            
        Returns:
            同步位置
        """
        # 计算互相关
        correlation = np.correlate(rx_signal, preamble, mode='valid')
        correlation_mag = np.abs(correlation)
        
        # 找到峰值位置
        sync_idx = np.argmax(correlation_mag)
        
        return sync_idx
    
    def freq_sync(self, rx_signal: np.ndarray, pilot_indices: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        频率同步：基于循环前缀的相位差
        
        Args:
            rx_signal: 接收信号
            pilot_indices: 导频位置
            
        Returns:
            (频率校正后的信号, 估计的频率偏移)
        """
        # 使用循环前缀进行粗频率同步
        fft_size = self.num_subcarriers
        
        # 提取多个OFDM符号的CP部分
        freq_offset_estimates = []
        
        for i in range(min(5, len(rx_signal) // (fft_size + self.cp_length))):
            start = i * (fft_size + self.cp_length)
            cp_part = rx_signal[start:start+self.cp_length]
            data_part = rx_signal[start+fft_size:start+fft_size+self.cp_length]
            
            if len(cp_part) == self.cp_length and len(data_part) == self.cp_length:
                # 计算相位差
                phase_diff = np.angle(np.mean(cp_part * np.conj(data_part)))
                freq_offset = phase_diff / (2 * np.pi * fft_size)
                freq_offset_estimates.append(freq_offset)
        
        if freq_offset_estimates:
            freq_offset = np.mean(freq_offset_estimates)
        else:
            freq_offset = 0
        
        # 频率校正
        n = np.arange(len(rx_signal))
        corrected_signal = rx_signal * np.exp(-1j * 2 * np.pi * freq_offset * n)
        
        return corrected_signal, freq_offset
    
    def ofdm_demodulate(self, rx_signal: np.ndarray) -> np.ndarray:
        """
        OFDM解调
        
        Args:
            rx_signal: 时域接收信号
            
        Returns:
            频域符号
        """
        fft_size = self.num_subcarriers
        symbol_length = fft_size + self.cp_length
        
        num_symbols = len(rx_signal) // symbol_length
        freq_symbols = []
        
        for i in range(num_symbols):
            start = i * symbol_length
            # 去除循环前缀
            ofdm_symbol = rx_signal[start+self.cp_length:start+symbol_length]
            
            if len(ofdm_symbol) < fft_size:
                break
            
            # FFT变换到频域
            freq_data = np.fft.fft(ofdm_symbol) / np.sqrt(fft_size)
            
            # 提取有效子载波
            start_idx = (fft_size - self.num_subcarriers) // 2
            freq_symbols.extend(freq_data[start_idx:start_idx+self.num_subcarriers])
        
        return np.array(freq_symbols)
    
    def channel_estimate(self, rx_pilots: np.ndarray, tx_pilots: np.ndarray) -> np.ndarray:
        """
        基于导频的信道估计 (LS估计)
        
        Args:
            rx_pilots: 接收导频
            tx_pilots: 发送导频
            
        Returns:
            信道估计值
        """
        # LS估计
        h_est = rx_pilots / tx_pilots
        return h_est
    
    def channel_estimate_mmse(self, rx_pilots: np.ndarray, tx_pilots: np.ndarray, 
                               noise_var: float = 0.01) -> np.ndarray:
        """
        MMSE信道估计
        
        Args:
            rx_pilots: 接收导频
            tx_pilots: 发送导频
            noise_var: 噪声方差
            
        Returns:
            MMSE信道估计值
        """
        # LS估计
        h_ls = rx_pilots / tx_pilots
        
        # 简化的MMSE估计 (假设已知信道统计特性)
        # H_mmse = R_hh * (R_hh + sigma^2*I)^-1 * H_ls
        # 这里使用简化的滤波器
        h_mmse = h_ls * (1 / (1 + noise_var))
        
        return h_mmse
    
    def equalize(self, rx_symbols: np.ndarray, channel_est: np.ndarray) -> np.ndarray:
        """
        信道均衡 (ZF均衡)
        
        Args:
            rx_symbols: 接收符号
            channel_est: 信道估计
            
        Returns:
            均衡后的符号
        """
        # 零迫均衡 (ZF)
        equalized = rx_symbols / channel_est
        return equalized
    
    def equalize_mmse(self, rx_symbols: np.ndarray, channel_est: np.ndarray, 
                      noise_var: float = 0.01) -> np.ndarray:
        """
        MMSE信道均衡
        
        Args:
            rx_symbols: 接收符号
            channel_est: 信道估计
            noise_var: 噪声方差
            
        Returns:
            MMSE均衡后的符号
        """
        # MMSE均衡
        h_conj = np.conj(channel_est)
        equalized = rx_symbols * h_conj / (np.abs(channel_est)**2 + noise_var)
        return equalized
    
    def qpsk_demodulate(self, symbols: np.ndarray) -> np.ndarray:
        """QPSK解调"""
        bits = np.zeros(len(symbols) * 2, dtype=int)
        
        for i, symbol in enumerate(symbols):
            # 判决
            bits[i*2] = 1 if np.real(symbol) > 0 else 0
            bits[i*2+1] = 1 if np.imag(symbol) > 0 else 0
        
        return bits
    
    def qam16_demodulate(self, symbols: np.ndarray) -> np.ndarray:
        """16QAM解调 - 与调制端的格雷编码对应"""
        bits = np.zeros(len(symbols) * 4, dtype=int)
        
        for i, symbol in enumerate(symbols):
            # 反归一化
            real_part = np.real(symbol) * np.sqrt(10)
            imag_part = np.imag(symbol) * np.sqrt(10)
            
            # I路判决 (阈值在 -2, 0, 2)
            if real_part < -2:
                i_val = -3
                i_bits = 0  # 00
            elif real_part < 0:
                i_val = -1
                i_bits = 1  # 01
            elif real_part < 2:
                i_val = 1
                i_bits = 3  # 11
            else:
                i_val = 3
                i_bits = 2  # 10
            
            # Q路判决
            if imag_part < -2:
                q_val = -3
                q_bits = 0
            elif imag_part < 0:
                q_val = -1
                q_bits = 1
            elif imag_part < 2:
                q_val = 1
                q_bits = 3
            else:
                q_val = 3
                q_bits = 2
            
            # 解码格雷编码
            # i_bits = b0*2 + b2
            bits[i*4] = (i_bits >> 1) & 1    # b0
            bits[i*4+2] = i_bits & 1          # b2
            
            # q_bits = b1*2 + b3
            bits[i*4+1] = (q_bits >> 1) & 1  # b1
            bits[i*4+3] = q_bits & 1          # b3
        
        return bits
    
    def demodulate(self, symbols: np.ndarray) -> np.ndarray:
        """根据调制方式进行解调"""
        if self.modulation == 'QPSK':
            return self.qpsk_demodulate(symbols)
        elif self.modulation == '16QAM':
            return self.qam16_demodulate(symbols)
        else:
            return self.qpsk_demodulate(symbols)
    
    def receive(self, rx_signal: np.ndarray, preamble: Optional[np.ndarray] = None) -> np.ndarray:
        """
        完整的接收流程
        
        Args:
            rx_signal: 接收信号
            preamble: 前导序列 (用于同步)
            
        Returns:
            解调后的比特
        """
        # 1. 时间同步 (如果有前导)
        if preamble is not None:
            sync_idx = self.time_sync(rx_signal, preamble)
            rx_signal = rx_signal[sync_idx:]
        
        # 2. 频率同步
        rx_signal, _ = self.freq_sync(rx_signal, np.array([]))
        
        # 3. OFDM解调
        freq_symbols = self.ofdm_demodulate(rx_signal)
        
        # 4. 解调
        bits = self.demodulate(freq_symbols)
        
        return bits
    
    def calculate_ber(self, tx_bits: np.ndarray, rx_bits: np.ndarray) -> float:
        """计算误码率"""
        min_len = min(len(tx_bits), len(rx_bits))
        errors = np.sum(tx_bits[:min_len] != rx_bits[:min_len])
        return errors / min_len
    
    def calculate_evm(self, tx_symbols: np.ndarray, rx_symbols: np.ndarray) -> float:
        """计算误差向量幅度 (EVM)"""
        min_len = min(len(tx_symbols), len(rx_symbols))
        error = tx_symbols[:min_len] - rx_symbols[:min_len]
        evm = np.sqrt(np.mean(np.abs(error)**2)) / np.sqrt(np.mean(np.abs(tx_symbols[:min_len])**2))
        return evm * 100  # 百分比


if __name__ == '__main__':
    # 测试接收机
    rx = NRReceiver(modulation='QPSK')
    
    # 生成测试信号
    tx_symbols = np.random.randn(1000) + 1j*np.random.randn(1000)
    tx_symbols = tx_symbols / np.sqrt(2)
    tx_bits = rx.demodulate(tx_symbols)
    
    print(f"解调比特数: {len(tx_bits)}")
    print(f"比特样本: {tx_bits[:20]}")
