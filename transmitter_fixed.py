"""
NR Transmitter Fixed Module
修复导频插入导致数据丢失的问题
"""

import numpy as np
from typing import Tuple


class NRTransmitterFixed:
    """5G NR 发射机（修复版）"""
    
    def __init__(self, 
                 num_subcarriers: int = 1200,
                 cp_length: int = 72,
                 modulation: str = 'QPSK'):
        self.num_subcarriers = num_subcarriers
        self.cp_length = cp_length
        self.modulation = modulation
        
        self.bits_per_symbol = {
            'QPSK': 2,
            '16QAM': 4,
            '64QAM': 6,
            '256QAM': 8
        }.get(modulation, 2)
        
        # 导频配置
        self.pilot_spacing = 4
        self.pilot_indices = np.arange(0, num_subcarriers, self.pilot_spacing)
        self.num_pilots = len(self.pilot_indices)
        self.num_data_subcarriers = num_subcarriers - self.num_pilots
        
        # 生成导频（固定种子）
        np.random.seed(42)
        self.pilot_values = (2 * np.random.randint(0, 2, self.num_pilots) - 1).astype(complex)
    
    def generate_data_bits(self, num_bits: int) -> np.ndarray:
        """生成随机数据比特"""
        return np.random.randint(0, 2, num_bits)
    
    def qpsk_modulate(self, bits: np.ndarray) -> np.ndarray:
        """QPSK调制"""
        bits_i = bits[0::2]
        bits_q = bits[1::2]
        symbols = (2*bits_i - 1) + 1j*(2*bits_q - 1)
        symbols = symbols / np.sqrt(2)
        return symbols
    
    def qam16_modulate(self, bits: np.ndarray) -> np.ndarray:
        """16QAM调制 - 格雷编码"""
        num_symbols = len(bits) // 4
        symbols = np.zeros(num_symbols, dtype=complex)
        
        for i in range(num_symbols):
            b = bits[i*4:(i+1)*4]
            
            # I路: b0b2 -> -3, -1, 3, 1
            i_bits = b[0] * 2 + b[2]
            if i_bits == 0:
                i_val = -3
            elif i_bits == 1:
                i_val = -1
            elif i_bits == 3:
                i_val = 1
            else:
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
        
        symbols = symbols / np.sqrt(10)
        return symbols
    
    def modulate(self, bits: np.ndarray) -> np.ndarray:
        """根据调制方式进行调制"""
        if self.modulation == 'QPSK':
            return self.qpsk_modulate(bits)
        elif self.modulation == '16QAM':
            return self.qam16_modulate(bits)
        else:
            return self.qpsk_modulate(bits)
    
    def create_ofdm_symbol(self, data_symbols: np.ndarray) -> np.ndarray:
        """
        创建OFDM符号：将数据符号和导频组合成完整的频域符号
        
        Args:
            data_symbols: 数据符号 (num_data_subcarriers,)
            
        Returns:
            完整的OFDM频域符号 (num_subcarriers,)
        """
        ofdm_symbol = np.zeros(self.num_subcarriers, dtype=complex)
        
        # 数据子载波索引（排除导频位置）
        data_indices = np.setdiff1d(np.arange(self.num_subcarriers), self.pilot_indices)
        
        # 填充数据
        ofdm_symbol[data_indices] = data_symbols
        
        # 插入导频
        ofdm_symbol[self.pilot_indices] = self.pilot_values
        
        return ofdm_symbol
    
    def ofdm_modulate(self, symbols: np.ndarray) -> np.ndarray:
        """
        OFDM调制
        
        Args:
            symbols: 数据符号（已包含导频位置的数据，会被覆盖）
            
        Returns:
            时域OFDM信号
        """
        num_data_symbols = len(symbols)
        num_ofdm_symbols = num_data_symbols // self.num_data_subcarriers
        
        ofdm_signal = []
        
        for i in range(num_ofdm_symbols):
            # 提取数据符号
            start = i * self.num_data_subcarriers
            end = start + self.num_data_subcarriers
            data_symbols = symbols[start:end]
            
            # 创建完整的OFDM符号（数据+导频）
            freq_data = self.create_ofdm_symbol(data_symbols)
            
            # IFFT变换到时域
            time_signal = np.fft.ifft(freq_data) * np.sqrt(self.num_subcarriers)
            
            # 添加循环前缀
            cp = time_signal[-self.cp_length:]
            ofdm_symbol = np.concatenate([cp, time_signal])
            
            ofdm_signal.extend(ofdm_symbol)
        
        return np.array(ofdm_signal)
    
    def transmit(self, num_bits: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        完整的发射流程
        
        Args:
            num_bits: 传输的比特数
            
        Returns:
            (发射信号, 原始比特)
        """
        if num_bits is None:
            # 默认传输一帧的数据量
            num_ofdm_symbols = 10
            num_bits = self.num_data_subcarriers * num_ofdm_symbols * self.bits_per_symbol
        
        # 1. 生成数据比特
        bits = self.generate_data_bits(num_bits)
        
        # 2. 调制（只调制数据，不生成导频）
        symbols = self.modulate(bits)
        
        # 3. OFDM调制（自动插入导频）
        tx_signal = self.ofdm_modulate(symbols)
        
        return tx_signal, bits


if __name__ == '__main__':
    # 测试发射机
    tx = NRTransmitterFixed(modulation='16QAM')
    signal, bits = tx.transmit(num_bits=10000)
    print(f"发射信号长度: {len(signal)}")
    print(f"数据子载波数: {tx.num_data_subcarriers}")
    print(f"导频数: {tx.num_pilots}")
    print(f"总子载波数: {tx.num_subcarriers}")
