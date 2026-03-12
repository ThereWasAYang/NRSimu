"""
NR Simulation Main Script
5G NR 接收机性能仿真主程序
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互后端，避免Windows下卡死
import matplotlib.pyplot as plt
from typing import List, Tuple
import time
import os

from transmitter import NRTransmitter
from channel import NRChannel
from receiver import NRReceiver


class NRSimulation:
    """NR系统仿真主类"""
    
    def __init__(self,
                 num_subcarriers: int = 1200,
                 cp_length: int = 72,
                 modulation: str = 'QPSK',
                 channel_type: str = 'awgn'):
        """
        初始化仿真参数
        
        Args:
            num_subcarriers: 子载波数
            cp_length: 循环前缀长度
            modulation: 调制方式
            channel_type: 信道类型
        """
        self.num_subcarriers = num_subcarriers
        self.cp_length = cp_length
        self.modulation = modulation
        self.channel_type = channel_type
        
        # 初始化组件
        self.tx = NRTransmitter(num_subcarriers, cp_length, modulation=modulation)
        self.rx = NRReceiver(num_subcarriers, cp_length, modulation=modulation)
        
    def run_single_simulation(self, snr_db: float, num_bits: int = 10000) -> Tuple[float, float]:
        """
        运行单次仿真
        
        Args:
            snr_db: 信噪比 (dB)
            num_bits: 传输比特数
            
        Returns:
            (误码率, EVM)
        """
        # 创建信道
        channel = NRChannel(snr_db=snr_db, channel_type=self.channel_type)
        
        # 1. 发射
        tx_signal, tx_bits = self.tx.transmit(num_bits)
        
        # 2. 通过信道
        rx_signal, _ = channel.apply_channel(tx_signal)
        
        # 3. 接收
        rx_bits = self.rx.receive(rx_signal)
        
        # 4. 计算性能指标
        min_len = min(len(tx_bits), len(rx_bits))
        ber = self.rx.calculate_ber(tx_bits, rx_bits)
        
        # 计算EVM (需要重新获取符号)
        tx_symbols = self.tx.modulate(tx_bits[:min_len])
        rx_symbols = self.rx.ofdm_demodulate(rx_signal)
        if len(rx_symbols) >= len(tx_symbols):
            evm = self.rx.calculate_evm(tx_symbols, rx_symbols[:len(tx_symbols)])
        else:
            evm = 0
        
        return ber, evm
    
    def run_snr_sweep(self, 
                      snr_range: np.ndarray,
                      num_bits: int = 100000,
                      num_trials: int = 10) -> Tuple[List[float], List[float]]:
        """
        SNR扫描仿真
        
        Args:
            snr_range: SNR范围 (dB)
            num_bits: 每次试验的比特数
            num_trials: 每个SNR点的试验次数
            
        Returns:
            (BER列表, EVM列表)
        """
        ber_list = []
        evm_list = []
        
        print(f"开始SNR扫描: {snr_range[0]}dB ~ {snr_range[-1]}dB")
        print(f"调制方式: {self.modulation}, 信道: {self.channel_type}")
        print("-" * 60)
        
        for snr_db in snr_range:
            ber_trials = []
            evm_trials = []
            
            for trial in range(num_trials):
                ber, evm = self.run_single_simulation(snr_db, num_bits)
                ber_trials.append(ber)
                evm_trials.append(evm)
            
            avg_ber = np.mean(ber_trials)
            avg_evm = np.mean(evm_trials)
            
            ber_list.append(avg_ber)
            evm_list.append(avg_evm)
            
            print(f"SNR = {snr_db:4.1f} dB | BER = {avg_ber:.2e} | EVM = {avg_evm:.2f}%")
        
        print("-" * 60)
        
        return ber_list, evm_list
    
    def plot_results(self, snr_range: np.ndarray, ber_list: List[float], evm_list: List[float]):
        """绘制仿真结果"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # BER曲线
        ax1 = axes[0]
        ax1.semilogy(snr_range, ber_list, 'b-o', linewidth=2, markersize=6)
        ax1.grid(True, which='both', linestyle='--', alpha=0.7)
        ax1.set_xlabel('SNR (dB)', fontsize=12)
        ax1.set_ylabel('Bit Error Rate (BER)', fontsize=12)
        ax1.set_title(f'BER Performance - {self.modulation}', fontsize=14)
        ax1.set_ylim([1e-5, 1])
        
        # EVM曲线
        ax2 = axes[1]
        ax2.plot(snr_range, evm_list, 'r-s', linewidth=2, markersize=6)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.set_xlabel('SNR (dB)', fontsize=12)
        ax2.set_ylabel('EVM (%)', fontsize=12)
        ax2.set_title(f'EVM Performance - {self.modulation}', fontsize=14)
        
        plt.tight_layout()
        plt.savefig('simulation_results.png', dpi=150, bbox_inches='tight')
        print("\n结果已保存到 simulation_results.png")
        plt.close()  # 关闭图形，避免内存泄漏
    
    def compare_modulations(self, 
                           snr_range: np.ndarray,
                           modulations: List[str] = ['QPSK', '16QAM'],
                           num_bits: int = 50000):
        """比较不同调制方式的性能"""
        plt.figure(figsize=(10, 6))
        
        colors = ['b', 'r', 'g', 'm']
        markers = ['o', 's', '^', 'd']
        
        for idx, mod in enumerate(modulations):
            print(f"\n正在仿真 {mod}...")
            
            # 临时更换调制方式
            self.tx.modulation = mod
            self.rx.modulation = mod
            self.tx.bits_per_symbol = {'QPSK': 2, '16QAM': 4, '64QAM': 6, '256QAM': 8}.get(mod, 2)
            self.rx.bits_per_symbol = self.tx.bits_per_symbol
            
            ber_list = []
            for snr_db in snr_range:
                ber_trials = []
                for _ in range(5):
                    ber, _ = self.run_single_simulation(snr_db, num_bits)
                    ber_trials.append(ber)
                ber_list.append(np.mean(ber_trials))
            
            plt.semilogy(snr_range, ber_list, 
                        color=colors[idx % len(colors)],
                        marker=markers[idx % len(markers)],
                        linewidth=2, markersize=6,
                        label=mod)
        
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        plt.xlabel('SNR (dB)', fontsize=12)
        plt.ylabel('Bit Error Rate (BER)', fontsize=12)
        plt.title('Modulation Comparison', fontsize=14)
        plt.legend(fontsize=11)
        plt.ylim([1e-5, 1])
        plt.tight_layout()
        plt.savefig('modulation_comparison.png', dpi=150, bbox_inches='tight')
        print("\n对比结果已保存到 modulation_comparison.png")
        plt.close()  # 关闭图形


def main():
    """主函数"""
    print("=" * 60)
    print("5G NR 接收机性能仿真")
    print("=" * 60)
    
    # 创建仿真实例
    sim = NRSimulation(
        num_subcarriers=1200,
        cp_length=72,
        modulation='QPSK',
        channel_type='awgn'
    )
    
    # 运行SNR扫描
    snr_range = np.arange(0, 25, 2)  # 0 to 24 dB, step 2
    ber_list, evm_list = sim.run_snr_sweep(snr_range, num_bits=50000, num_trials=5)
    
    # 绘制结果
    sim.plot_results(snr_range, ber_list, evm_list)
    
    # 调制方式对比
    print("\n" + "=" * 60)
    print("调制方式性能对比")
    print("=" * 60)
    sim.compare_modulations(snr_range, ['QPSK', '16QAM'], num_bits=30000)
    
    print("\n仿真完成！")


if __name__ == '__main__':
    main()
