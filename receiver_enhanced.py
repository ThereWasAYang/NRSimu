"""
NR Receiver Enhanced Module
5G NR 增强型接收机模块：完整流程包括同步、信道估计、均衡、解调
包含星座图可视化功能
"""

import numpy as np
from typing import Tuple, Optional, List
from scipy import signal as scipy_signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class NRReceiverEnhanced:
    """5G NR 增强型接收机 - 包含完整信号处理流程"""
    
    def __init__(self,
                 num_subcarriers: int = 1200,
                 cp_length: int = 72,
                 modulation: str = 'QPSK',
                 num_pilot_symbols: int = 4):
        """
        初始化接收机参数
        
        Args:
            num_subcarriers: 子载波数量
            cp_length: 循环前缀长度
            modulation: 调制方式
            num_pilot_symbols: 每帧导频符号数
        """
        self.num_subcarriers = num_subcarriers
        self.cp_length = cp_length
        self.modulation = modulation
        self.num_pilot_symbols = num_pilot_symbols
        
        self.bits_per_symbol = {
            'QPSK': 2,
            '16QAM': 4,
            '64QAM': 6,
            '256QAM': 8
        }.get(modulation, 2)
        
        # 存储信号处理中间结果
        self.rx_signal = None
        self.sync_signal = None
        self.freq_corrected_signal = None
        self.ofdm_symbols = None
        self.channel_estimates = None
        self.equalized_symbols = None
        self.demodulated_bits = None
        
        # 星座图数据
        self.constellation_before_eq = []  # 均衡前
        self.constellation_after_eq = []   # 均衡后
        
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
    
    def freq_sync(self, rx_signal: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        频率同步：基于循环前缀的相位差
        
        Args:
            rx_signal: 接收信号
            
        Returns:
            (频率校正后的信号, 估计的频率偏移)
        """
        fft_size = self.num_subcarriers
        symbol_length = fft_size + self.cp_length
        
        freq_offset_estimates = []
        
        # 使用多个OFDM符号进行估计
        for i in range(min(5, len(rx_signal) // symbol_length)):
            start = i * symbol_length
            cp_part = rx_signal[start:start+self.cp_length]
            data_part = rx_signal[start+fft_size:start+fft_size+self.cp_length]
            
            if len(cp_part) == self.cp_length and len(data_part) == self.cp_length:
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
            频域符号 (包含导频和数据)
        """
        fft_size = self.num_subcarriers
        symbol_length = fft_size + self.cp_length
        
        num_symbols = len(rx_signal) // symbol_length
        freq_symbols = []
        
        for i in range(num_symbols):
            start = i * symbol_length
            ofdm_symbol = rx_signal[start+self.cp_length:start+symbol_length]
            
            if len(ofdm_symbol) < fft_size:
                break
            
            # FFT变换到频域
            freq_data = np.fft.fft(ofdm_symbol) / np.sqrt(fft_size)
            
            # 提取有效子载波
            start_idx = (fft_size - self.num_subcarriers) // 2
            freq_symbols.extend(freq_data[start_idx:start_idx+self.num_subcarriers])
        
        return np.array(freq_symbols)
    
    def generate_pilot_pattern(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成导频图案
        
        Returns:
            (导频位置索引, 导频值)
        """
        # 导频间隔：每4个子载波放一个导频
        pilot_spacing = 4
        pilot_indices = np.arange(0, self.num_subcarriers, pilot_spacing)
        
        # 生成已知导频序列（BPSK）
        np.random.seed(42)  # 固定种子，确保收发一致
        pilot_values = 2 * np.random.randint(0, 2, len(pilot_indices)) - 1
        
        return pilot_indices, pilot_values
    
    def channel_estimate_ls(self, rx_symbols: np.ndarray, pilot_indices: np.ndarray, 
                           pilot_values: np.ndarray) -> np.ndarray:
        """
        LS信道估计（最小二乘）
        
        Args:
            rx_symbols: 接收的频域符号
            pilot_indices: 导频位置
            pilot_values: 发送的导频值
            
        Returns:
            全子载波信道估计
        """
        # 提取接收导频
        rx_pilots = rx_symbols[pilot_indices]
        
        # LS估计：H = Y/X
        h_pilots = rx_pilots / pilot_values
        
        # 线性插值到所有子载波
        h_est = np.zeros(self.num_subcarriers, dtype=complex)
        all_indices = np.arange(self.num_subcarriers)
        
        # 分别对实部和虚部进行插值
        h_est_real = np.interp(all_indices, pilot_indices, np.real(h_pilots))
        h_est_imag = np.interp(all_indices, pilot_indices, np.imag(h_pilots))
        h_est = h_est_real + 1j * h_est_imag
        
        return h_est
    
    def channel_estimate_mmse(self, rx_symbols: np.ndarray, pilot_indices: np.ndarray,
                              pilot_values: np.ndarray, snr_db: float = 20) -> np.ndarray:
        """
        MMSE信道估计（最小均方误差）
        
        Args:
            rx_symbols: 接收的频域符号
            pilot_indices: 导频位置
            pilot_values: 发送的导频值
            snr_db: 估计的信噪比
            
        Returns:
            MMSE信道估计
        """
        # 先用LS估计
        h_ls = self.channel_estimate_ls(rx_symbols, pilot_indices, pilot_values)
        
        # MMSE加权（简化版）
        snr_linear = 10 ** (snr_db / 10)
        mmse_factor = snr_linear / (snr_linear + 1)
        
        h_mmse = h_ls * mmse_factor
        
        return h_mmse
    
    def equalize_zf(self, rx_symbols: np.ndarray, channel_est: np.ndarray) -> np.ndarray:
        """
        零迫均衡（ZF）
        
        Args:
            rx_symbols: 接收符号
            channel_est: 信道估计
            
        Returns:
            均衡后的符号
        """
        return rx_symbols / channel_est
    
    def equalize_mmse(self, rx_symbols: np.ndarray, channel_est: np.ndarray, 
                      noise_var: float = 0.01) -> np.ndarray:
        """
        MMSE均衡
        
        Args:
            rx_symbols: 接收符号
            channel_est: 信道估计
            noise_var: 噪声方差
            
        Returns:
            MMSE均衡后的符号
        """
        h_conj = np.conj(channel_est)
        equalized = rx_symbols * h_conj / (np.abs(channel_est)**2 + noise_var)
        return equalized
    
    def demodulate(self, symbols: np.ndarray) -> np.ndarray:
        """根据调制方式进行解调"""
        if self.modulation == 'QPSK':
            return self.qpsk_demodulate(symbols)
        elif self.modulation == '16QAM':
            return self.qam16_demodulate(symbols)
        else:
            return self.qpsk_demodulate(symbols)
    
    def qpsk_demodulate(self, symbols: np.ndarray) -> np.ndarray:
        """QPSK解调"""
        bits = np.zeros(len(symbols) * 2, dtype=int)
        
        for i, symbol in enumerate(symbols):
            bits[i*2] = 1 if np.real(symbol) > 0 else 0
            bits[i*2+1] = 1 if np.imag(symbol) > 0 else 0
        
        return bits
    
    def qam16_demodulate(self, symbols: np.ndarray) -> np.ndarray:
        """16QAM解调 - 与调制端的格雷编码对应"""
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
    
    def receive_with_channel_processing(self, rx_signal: np.ndarray, 
                                       tx_signal: np.ndarray = None,
                                       snr_db: float = 20) -> dict:
        """
        完整的接收流程，包含信道估计和均衡
        
        Args:
            rx_signal: 接收信号
            tx_signal: 发射信号（用于生成导频）
            snr_db: 估计的信噪比
            
        Returns:
            包含所有中间结果的字典
        """
        results = {}
        
        # 1. 时间同步（如果有前导）
        if tx_signal is not None:
            # 简化的同步：直接使用已知信号
            self.sync_signal = rx_signal
        else:
            self.sync_signal = rx_signal
        
        # 2. 频率同步
        self.freq_corrected_signal, freq_offset = self.freq_sync(self.sync_signal)
        results['freq_offset'] = freq_offset
        
        # 3. OFDM解调
        self.ofdm_symbols = self.ofdm_demodulate(self.freq_corrected_signal)
        results['num_ofdm_symbols'] = len(self.ofdm_symbols) // self.num_subcarriers
        
        # 4. 信道估计
        pilot_indices, pilot_values = self.generate_pilot_pattern()
        
        # 对每个OFDM符号进行信道估计和均衡
        equalized_all = []
        channel_ests_all = []
        
        symbols_per_ofdm = self.num_subcarriers
        num_ofdm = len(self.ofdm_symbols) // symbols_per_ofdm
        
        for i in range(num_ofdm):
            start = i * symbols_per_ofdm
            end = start + symbols_per_ofdm
            ofdm_symbol = self.ofdm_symbols[start:end]
            
            # 保存均衡前的星座点
            self.constellation_before_eq.extend(ofdm_symbol)
            
            # LS信道估计
            h_est = self.channel_estimate_ls(ofdm_symbol, pilot_indices, pilot_values)
            channel_ests_all.extend(h_est)
            
            # MMSE均衡
            eq_symbol = self.equalize_mmse(ofdm_symbol, h_est, noise_var=10**(-snr_db/10))
            equalized_all.extend(eq_symbol)
            
            # 保存均衡后的星座点
            self.constellation_after_eq.extend(eq_symbol)
        
        self.equalized_symbols = np.array(equalized_all)
        self.channel_estimates = np.array(channel_ests_all)
        
        results['channel_estimates'] = self.channel_estimates
        results['equalized_symbols'] = self.equalized_symbols
        
        # 5. 解调
        self.demodulated_bits = self.demodulate(self.equalized_symbols)
        results['bits'] = self.demodulated_bits
        
        return results
    
    def plot_constellation(self, save_path: str = 'constellation.png'):
        """
        绘制星座图
        
        包括：
        - 均衡前的星座点（接收信号）
        - 均衡后的星座点
        - 理想的参考星座点
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 1. 均衡前的星座图
        ax1 = axes[0]
        before_eq = np.array(self.constellation_before_eq[:1000])  # 取前1000个点
        ax1.scatter(np.real(before_eq), np.imag(before_eq), 
                   c='blue', alpha=0.5, s=10, label='Received')
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax1.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        ax1.set_xlabel('In-Phase', fontsize=11)
        ax1.set_ylabel('Quadrature', fontsize=11)
        ax1.set_title('Before Equalization', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # 2. 均衡后的星座图
        ax2 = axes[1]
        after_eq = np.array(self.constellation_after_eq[:1000])
        ax2.scatter(np.real(after_eq), np.imag(after_eq), 
                   c='green', alpha=0.5, s=10, label='Equalized')
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax2.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        ax2.set_xlabel('In-Phase', fontsize=11)
        ax2.set_ylabel('Quadrature', fontsize=11)
        ax2.set_title('After Equalization', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        
        # 3. 理想的参考星座图
        ax3 = axes[2]
        if self.modulation == 'QPSK':
            ref_points = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
        elif self.modulation == '16QAM':
            ref_points = []
            for i in [-3, -1, 1, 3]:
                for q in [-3, -1, 1, 3]:
                    ref_points.append(i + 1j*q)
            ref_points = np.array(ref_points) / np.sqrt(10)
        else:
            ref_points = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
        
        ax3.scatter(np.real(ref_points), np.imag(ref_points), 
                   c='red', s=100, marker='*', label='Ideal', zorder=5)
        ax3.scatter(np.real(after_eq[:100]), np.imag(after_eq[:100]), 
                   c='green', alpha=0.3, s=10, label='Equalized')
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax3.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        ax3.set_xlabel('In-Phase', fontsize=11)
        ax3.set_ylabel('Quadrature', fontsize=11)
        ax3.set_title(f'Ideal {self.modulation} Constellation', fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.set_aspect('equal')
        ax3.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"星座图已保存到: {save_path}")
        plt.close()
    
    def plot_channel_response(self, save_path: str = 'channel_response.png'):
        """绘制信道响应"""
        if self.channel_estimates is None:
            print("请先运行接收流程")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 信道幅度响应
        ax1 = axes[0]
        h_mag = np.abs(self.channel_estimates[:self.num_subcarriers])
        ax1.plot(h_mag, 'b-', linewidth=1.5)
        ax1.set_xlabel('Subcarrier Index', fontsize=11)
        ax1.set_ylabel('Magnitude', fontsize=11)
        ax1.set_title('Channel Frequency Response (Magnitude)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # 信道相位响应
        ax2 = axes[1]
        h_phase = np.angle(self.channel_estimates[:self.num_subcarriers])
        ax2.plot(h_phase, 'r-', linewidth=1.5)
        ax2.set_xlabel('Subcarrier Index', fontsize=11)
        ax2.set_ylabel('Phase (rad)', fontsize=11)
        ax2.set_title('Channel Frequency Response (Phase)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"信道响应图已保存到: {save_path}")
        plt.close()
    
    def calculate_ber(self, tx_bits: np.ndarray, rx_bits: np.ndarray) -> float:
        """计算误码率"""
        min_len = min(len(tx_bits), len(rx_bits))
        errors = np.sum(tx_bits[:min_len] != rx_bits[:min_len])
        return errors / min_len if min_len > 0 else 0
    
    def calculate_evm(self, tx_symbols: np.ndarray, rx_symbols: np.ndarray) -> float:
        """计算误差向量幅度 (EVM)"""
        min_len = min(len(tx_symbols), len(rx_symbols))
        if min_len == 0:
            return 0
        error = tx_symbols[:min_len] - rx_symbols[:min_len]
        evm = np.sqrt(np.mean(np.abs(error)**2)) / np.sqrt(np.mean(np.abs(tx_symbols[:min_len])**2))
        return evm * 100


if __name__ == '__main__':
    # 测试增强型接收机
    from transmitter import NRTransmitter
    from channel import NRChannel
    
    print("测试增强型接收机...")
    
    # 初始化
    tx = NRTransmitter(modulation='16QAM')
    channel = NRChannel(snr_db=20, channel_type='rayleigh')
    rx = NRReceiverEnhanced(modulation='16QAM')
    
    # 发射
    tx_signal, tx_bits = tx.transmit(num_bits=4000)
    
    # 通过信道
    rx_signal, _ = channel.apply_channel(tx_signal)
    
    # 接收（完整流程）
    results = rx.receive_with_channel_processing(rx_signal, tx_signal, snr_db=20)
    
    # 计算性能
    ber = rx.calculate_ber(tx_bits, results['bits'])
    print(f"BER: {ber:.2e}")
    
    # 绘制星座图
    rx.plot_constellation('test_constellation.png')
    rx.plot_channel_response('test_channel.png')
    
    print("测试完成！")
