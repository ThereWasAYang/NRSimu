"""
NR Transmitter Module
5G NR 发射机模块：包括信道编码、调制、OFDM信号生成
"""

import numpy as np
from typing import Tuple, Optional


class NRTransmitter:
    """5G NR 发射机"""
    
    def __init__(self, 
                 num_subcarriers: int = 1200,
                 cp_length: int = 72,
                 num_symbols: int = 14,
                 modulation: str = 'QPSK'):
        """
        初始化发射机参数
        
        Args:
            num_subcarriers: 子载波数量
            cp_length: 循环前缀长度
            num_symbols: 每个子帧的OFDM符号数
            modulation: 调制方式 ('QPSK', '16QAM', '64QAM', '256QAM')
        """
        self.num_subcarriers = num_subcarriers
        self.cp_length = cp_length
        self.num_symbols = num_symbols
        self.modulation = modulation
        
        # 根据调制方式设置每符号比特数
        self.bits_per_symbol = {
            'QPSK': 2,
            '16QAM': 4,
            '64QAM': 6,
            '256QAM': 8
        }.get(modulation, 2)
        
    def generate_data_bits(self, num_bits: int) -> np.ndarray:
        """生成随机数据比特"""
        return np.random.randint(0, 2, num_bits)
    
    def qpsk_modulate(self, bits: np.ndarray) -> np.ndarray:
        """QPSK调制"""
        # 将比特映射为符号
        bits_i = bits[0::2]
        bits_q = bits[1::2]
        
        # QPSK星座映射
        symbols = (2*bits_i - 1) + 1j*(2*bits_q - 1)
        # 归一化功率
        symbols = symbols / np.sqrt(2)
        return symbols
    
    def qam16_modulate(self, bits: np.ndarray) -> np.ndarray:
        """16QAM调制 - 格雷编码映射"""
        # 每4个比特映射一个符号: b0b1b2b3 -> I + jQ
        # 使用格雷编码减少误码
        num_symbols = len(bits) // 4
        symbols = np.zeros(num_symbols, dtype=complex)
        
        for i in range(num_symbols):
            b = bits[i*4:(i+1)*4]
            # 格雷编码映射到星座点
            # I路: b0b2 -> -3, -1, 3, 1 (格雷码顺序)
            i_bits = b[0] * 2 + b[2]
            if i_bits == 0:    # 00 -> -3
                i_val = -3
            elif i_bits == 1:  # 01 -> -1
                i_val = -1
            elif i_bits == 3:  # 11 -> 1
                i_val = 1
            else:              # 10 -> 3
                i_val = 3
            
            # Q路: b1b3 -> -3, -1, 3, 1
            q_bits = b[1] * 2 + b[3]
            if q_bits == 0:
                q_val = -3
            elif q_bits == 1:
                q_val = -1
            elif q_bits == 3:
                q_val = 1
            else:
                q_val = 3
            
            symbols[i] = i_val + 1j*q_val
        
        # 归一化: 平均功率 = (9+1+9+1)/4 = 5 -> sqrt(10)
        symbols = symbols / np.sqrt(10)
        return symbols
    
    def modulate(self, bits: np.ndarray) -> np.ndarray:
        """根据配置的调制方式进行调制"""
        if self.modulation == 'QPSK':
            return self.qpsk_modulate(bits)
        elif self.modulation == '16QAM':
            return self.qam16_modulate(bits)
        else:
            return self.qpsk_modulate(bits)
    
    def ofdm_modulate(self, symbols: np.ndarray) -> np.ndarray:
        """
        OFDM调制
        
        Args:
            symbols: 频域符号
            
        Returns:
            时域OFDM信号
        """
        num_symbols_total = len(symbols)
        num_symbols_per_ofdm = self.num_subcarriers
        num_ofdm_symbols = num_symbols_total // num_symbols_per_ofdm
        
        ofdm_signal = []
        
        for i in range(num_ofdm_symbols):
            # 提取当前OFDM符号的频域数据
            freq_data = symbols[i*num_symbols_per_ofdm:(i+1)*num_symbols_per_ofdm]
            
            # 零填充到FFT大小
            fft_size = self.num_subcarriers
            freq_padded = np.zeros(fft_size, dtype=complex)
            start_idx = (fft_size - num_symbols_per_ofdm) // 2
            freq_padded[start_idx:start_idx+num_symbols_per_ofdm] = freq_data
            
            # IFFT变换到时域
            time_signal = np.fft.ifft(freq_padded) * np.sqrt(fft_size)
            
            # 添加循环前缀
            cp = time_signal[-self.cp_length:]
            ofdm_symbol = np.concatenate([cp, time_signal])
            
            ofdm_signal.extend(ofdm_symbol)
        
        return np.array(ofdm_signal)
    
    def transmit(self, num_bits: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        完整的发射流程
        
        Args:
            num_bits: 传输的比特数，默认为一帧的数据量
            
        Returns:
            (发射信号, 原始比特)
        """
        if num_bits is None:
            num_bits = self.num_subcarriers * self.num_symbols * self.bits_per_symbol
        
        # 1. 生成数据比特
        bits = self.generate_data_bits(num_bits)
        
        # 2. 调制
        symbols = self.modulate(bits)
        
        # 3. OFDM调制
        tx_signal = self.ofdm_modulate(symbols)
        
        return tx_signal, bits
    
    def add_preamble(self, signal: np.ndarray) -> np.ndarray:
        """添加同步前导序列"""
        # 生成Zadoff-Chu序列作为前导
        n = np.arange(self.num_subcarriers)
        u = 25  # ZC序列的根索引
        zc_seq = np.exp(-1j*np.pi*u*n*(n+1)/self.num_subcarriers)
        
        # 前导的OFDM调制
        preamble = self.ofdm_modulate(zc_seq)
        
        # 拼接前导和数据
        return np.concatenate([preamble, signal])


if __name__ == '__main__':
    # 测试发射机
    tx = NRTransmitter(modulation='QPSK')
    signal, bits = tx.transmit(num_bits=10000)
    print(f"发射信号长度: {len(signal)}")
    print(f"发射信号功率: {np.mean(np.abs(signal)**2):.4f}")
