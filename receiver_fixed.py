"""
NR Receiver Fixed Module
修复版接收机，与发射机匹配
"""

import numpy as np
from typing import Tuple, List
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class NRReceiverFixed:
    """5G NR 接收机（修复版）"""
    
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
        
        # 导频配置（与发射机一致）
        self.pilot_spacing = 4
        self.pilot_indices = np.arange(0, num_subcarriers, self.pilot_spacing)
        self.num_pilots = len(self.pilot_indices)
        self.num_data_subcarriers = num_subcarriers - self.num_pilots
        
        # 生成导频（与发射机一致）
        np.random.seed(42)
        self.pilot_values = (2 * np.random.randint(0, 2, self.num_pilots) - 1).astype(complex)
        
        # 存储中间结果
        self.constellation_before_eq = []
        self.constellation_after_eq = []
    
    def ofdm_demodulate(self, rx_signal: np.ndarray) -> np.ndarray:
        """OFDM解调"""
        symbol_length = self.num_subcarriers + self.cp_length
        num_symbols = len(rx_signal) // symbol_length
        
        freq_symbols = []
        
        for i in range(num_symbols):
            start = i * symbol_length
            ofdm_symbol = rx_signal[start+self.cp_length:start+symbol_length]
            
            if len(ofdm_symbol) < self.num_subcarriers:
                break
            
            # FFT变换到频域
            freq_data = np.fft.fft(ofdm_symbol) / np.sqrt(self.num_subcarriers)
            freq_symbols.extend(freq_data)
        
        return np.array(freq_symbols)
    
    def channel_estimate_ls(self, ofdm_symbol: np.ndarray) -> np.ndarray:
        """LS信道估计"""
        # 提取接收导频
        rx_pilots = ofdm_symbol[self.pilot_indices]
        
        # LS估计
        h_pilots = rx_pilots / self.pilot_values
        
        # 线性插值到所有子载波
        all_indices = np.arange(self.num_subcarriers)
        h_est_real = np.interp(all_indices, self.pilot_indices, np.real(h_pilots))
        h_est_imag = np.interp(all_indices, self.pilot_indices, np.imag(h_pilots))
        h_est = h_est_real + 1j * h_est_imag
        
        return h_est
    
    def equalize_mmse(self, rx_symbols: np.ndarray, h_est: np.ndarray, 
                      noise_var: float) -> np.ndarray:
        """MMSE均衡 - 使用正则化参数防止噪声放大"""
        h_conj = np.conj(h_est)
        h_power = np.abs(h_est)**2
        
        # 添加正则化项，防止深衰落时的噪声放大
        reg_noise_var = noise_var * 1.5  # 稍微高估噪声，更保守
        
        equalized = rx_symbols * h_conj / (h_power + reg_noise_var)
        return equalized
    
    def extract_data_symbols(self, ofdm_symbol: np.ndarray) -> np.ndarray:
        """提取数据子载波（排除导频）"""
        data_indices = np.setdiff1d(np.arange(self.num_subcarriers), self.pilot_indices)
        return ofdm_symbol[data_indices]
    
    def qpsk_demodulate(self, symbols: np.ndarray) -> np.ndarray:
        """QPSK解调"""
        bits = np.zeros(len(symbols) * 2, dtype=int)
        for i, symbol in enumerate(symbols):
            bits[i*2] = 1 if np.real(symbol) > 0 else 0
            bits[i*2+1] = 1 if np.imag(symbol) > 0 else 0
        return bits
    
    def qam16_demodulate(self, symbols: np.ndarray) -> np.ndarray:
        """16QAM解调"""
        bits = np.zeros(len(symbols) * 4, dtype=int)
        
        for i, symbol in enumerate(symbols):
            real_part = np.real(symbol) * np.sqrt(10)
            imag_part = np.imag(symbol) * np.sqrt(10)
            
            # I路判决
            if real_part < -2:
                i_bits = 0
            elif real_part < 0:
                i_bits = 1
            elif real_part < 2:
                i_bits = 3
            else:
                i_bits = 2
            
            # Q路判决
            if imag_part < -2:
                q_bits = 0
            elif imag_part < 0:
                q_bits = 1
            elif imag_part < 2:
                q_bits = 3
            else:
                q_bits = 2
            
            bits[i*4] = (i_bits >> 1) & 1
            bits[i*4+2] = i_bits & 1
            bits[i*4+1] = (q_bits >> 1) & 1
            bits[i*4+3] = q_bits & 1
        
        return bits
    
    def demodulate(self, symbols: np.ndarray) -> np.ndarray:
        """解调"""
        if self.modulation == 'QPSK':
            return self.qpsk_demodulate(symbols)
        elif self.modulation == '16QAM':
            return self.qam16_demodulate(symbols)
        else:
            return self.qpsk_demodulate(symbols)
    
    def receive(self, rx_signal: np.ndarray, snr_db: float) -> dict:
        """完整接收流程"""
        results = {}
        
        # 1. OFDM解调
        ofdm_symbols = self.ofdm_demodulate(rx_signal)
        num_ofdm = len(ofdm_symbols) // self.num_subcarriers
        
        # 2. 处理每个OFDM符号
        data_symbols_all = []
        
        for i in range(num_ofdm):
            start = i * self.num_subcarriers
            ofdm_symbol = ofdm_symbols[start:start+self.num_subcarriers]
            
            # 保存均衡前的数据子载波
            data_before = self.extract_data_symbols(ofdm_symbol)
            self.constellation_before_eq.extend(data_before)
            
            # 信道估计
            h_est = self.channel_estimate_ls(ofdm_symbol)
            
            # 估计噪声方差（从导频）
            rx_pilots = ofdm_symbol[self.pilot_indices]
            noise_var = np.mean(np.abs(rx_pilots - self.pilot_values * h_est[self.pilot_indices])**2)
            noise_var = max(noise_var, 1e-10)
            
            # 均衡
            eq_symbol = self.equalize_mmse(ofdm_symbol, h_est, noise_var)
            
            # 提取数据
            data_after = self.extract_data_symbols(eq_symbol)
            
            # 功率归一化
            data_power = np.mean(np.abs(data_after)**2)
            if data_power > 0:
                data_after = data_after / np.sqrt(data_power)
            
            self.constellation_after_eq.extend(data_after)
            data_symbols_all.extend(data_after)
        
        # 3. 解调
        data_symbols_all = np.array(data_symbols_all)
        bits = self.demodulate(data_symbols_all)
        
        results['bits'] = bits
        results['data_symbols'] = data_symbols_all
        
        return results
    
    def calculate_ber(self, tx_bits: np.ndarray, rx_bits: np.ndarray) -> float:
        """计算BER"""
        min_len = min(len(tx_bits), len(rx_bits))
        if min_len == 0:
            return 1.0
        errors = np.sum(tx_bits[:min_len] != rx_bits[:min_len])
        return errors / min_len
    
    def calculate_evm(self, symbols: np.ndarray) -> float:
        """计算EVM"""
        if len(symbols) == 0:
            return 0
        
        # 根据调制方式确定参考点
        if self.modulation == 'QPSK':
            ref_points = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
        elif self.modulation == '16QAM':
            ref_points = np.array([i+1j*q for i in [-3,-1,1,3] for q in [-3,-1,1,3]]) / np.sqrt(10)
        else:
            ref_points = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
        
        # 计算每个符号到最近参考点的误差
        errors = []
        for s in symbols:
            min_err = min(np.abs(s - r)**2 for r in ref_points)
            errors.append(min_err)
        
        evm = np.sqrt(np.mean(errors))
        ref_power = np.sqrt(np.mean(np.abs(ref_points)**2))
        
        return (evm / ref_power) * 100
    
    def plot_constellation(self, save_path: str = 'constellation.png'):
        """绘制星座图"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        before = np.array(self.constellation_before_eq[:1000])
        after = np.array(self.constellation_after_eq[:1000])
        
        evm_before = self.calculate_evm(before)
        evm_after = self.calculate_evm(after)
        
        # 参考点
        if self.modulation == 'QPSK':
            ref_points = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
        elif self.modulation == '16QAM':
            ref_points = np.array([i+1j*q for i in [-3,-1,1,3] for q in [-3,-1,1,3]]) / np.sqrt(10)
        else:
            ref_points = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
        
        # 均衡前
        ax1 = axes[0]
        ax1.scatter(np.real(before), np.imag(before), c='blue', alpha=0.5, s=10)
        ax1.scatter(np.real(ref_points), np.imag(ref_points), c='red', s=150, marker='*', edgecolors='black')
        ax1.set_title(f'Before Equalization\nEVM = {evm_before:.2f}%', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax1.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        
        # 均衡后
        ax2 = axes[1]
        ax2.scatter(np.real(after), np.imag(after), c='green', alpha=0.5, s=10)
        ax2.scatter(np.real(ref_points), np.imag(ref_points), c='red', s=150, marker='*', edgecolors='black')
        ax2.set_title(f'After Equalization\nEVM = {evm_after:.2f}%', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax2.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"EVM Before: {evm_before:.2f}%")
        print(f"EVM After:  {evm_after:.2f}%")
        print(f"Improvement: {evm_before - evm_after:.2f}%")


if __name__ == '__main__':
    print("Receiver Fixed Module")
