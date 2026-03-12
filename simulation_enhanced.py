"""
NR Enhanced Simulation
5G NR 增强型仿真 - 包含信道估计、均衡和星座图可视化
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Tuple

from transmitter import NRTransmitter
from channel import NRChannel
from receiver_enhanced import NRReceiverEnhanced


class NREnhancedSimulation:
    """NR增强型系统仿真"""
    
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
        self.rx = NRReceiverEnhanced(num_subcarriers, cp_length, modulation=modulation)
        
    def run_single_simulation(self, snr_db: float, num_bits: int = 10000) -> dict:
        """
        运行单次仿真，返回详细结果
        
        Args:
            snr_db: 信噪比 (dB)
            num_bits: 传输比特数
            
        Returns:
            包含BER、EVM、星座图数据的字典
        """
        # 创建信道
        channel = NRChannel(snr_db=snr_db, channel_type=self.channel_type)
        
        # 1. 发射（增加比特数以补偿导频开销）
        # 导频占用 1/4 的子载波
        num_bits_with_overhead = int(num_bits * 4 / 3)
        tx_signal, tx_bits = self.tx.transmit(num_bits_with_overhead)
        
        # 2. 通过信道
        rx_signal, h_true = channel.apply_channel(tx_signal)
        
        # 3. 接收（完整流程，包含信道估计和均衡）
        results = self.rx.receive_with_channel_processing(rx_signal, tx_signal, snr_db)
        
        # 4. 计算性能指标
        min_len = min(len(tx_bits), len(results['bits']))
        ber = self.rx.calculate_ber(tx_bits, results['bits'])
        
        # 计算EVM（使用均衡后的符号）
        tx_symbols = self.tx.modulate(tx_bits[:min_len])
        if len(self.rx.equalized_symbols) >= len(tx_symbols):
            evm = self.rx.calculate_evm(tx_symbols, self.rx.equalized_symbols[:len(tx_symbols)])
        else:
            evm = 0
        
        results['ber'] = ber
        results['evm'] = evm
        results['snr_db'] = snr_db
        
        return results
    
    def run_snr_sweep(self, 
                      snr_range: np.ndarray,
                      num_bits: int = 100000,
                      num_trials: int = 5) -> Tuple[List[float], List[float]]:
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
                results = self.run_single_simulation(snr_db, num_bits)
                ber_trials.append(results['ber'])
                evm_trials.append(results['evm'])
            
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
        plt.savefig('simulation_results_enhanced.png', dpi=150, bbox_inches='tight')
        print("\n结果已保存到 simulation_results_enhanced.png")
        plt.close()
    
    def compare_equalization(self, snr_db: float = 20, num_bits: int = 50000):
        """
        比较均衡前后的性能
        """
        print(f"\n{'='*60}")
        print(f"均衡性能对比 - SNR={snr_db}dB")
        print(f"{'='*60}")
        
        # 运行仿真
        results = self.run_single_simulation(snr_db, num_bits)
        
        print(f"调制方式: {self.modulation}")
        print(f"信道类型: {self.channel_type}")
        print(f"OFDM符号数: {results['num_ofdm_symbols']}")
        print(f"频偏估计: {results['freq_offset']:.6f}")
        print(f"BER: {results['ber']:.2e}")
        print(f"EVM: {results['evm']:.2f}%")
        
        # 绘制星座图
        self.rx.plot_constellation(f'constellation_{self.modulation}_SNR{snr_db}.png')
        
        # 绘制信道响应
        if self.channel_type != 'awgn':
            self.rx.plot_channel_response(f'channel_response_{self.channel_type}.png')
        
        print(f"{'='*60}\n")


def main():
    """主函数"""
    print("=" * 60)
    print("5G NR 增强型接收机性能仿真")
    print("包含信道估计、均衡和星座图可视化")
    print("=" * 60)
    
    # 测试1: QPSK with AWGN
    print("\n[测试1] QPSK - AWGN信道")
    sim1 = NREnhancedSimulation(
        num_subcarriers=1200,
        cp_length=72,
        modulation='QPSK',
        channel_type='awgn'
    )
    sim1.compare_equalization(snr_db=15, num_bits=20000)
    
    # 测试2: 16QAM with Rayleigh
    print("\n[测试2] 16QAM - 瑞利衰落信道")
    sim2 = NREnhancedSimulation(
        num_subcarriers=1200,
        cp_length=72,
        modulation='16QAM',
        channel_type='rayleigh'
    )
    sim2.compare_equalization(snr_db=25, num_bits=20000)
    
    # 测试3: SNR扫描
    print("\n[测试3] SNR扫描 - QPSK")
    sim3 = NREnhancedSimulation(
        num_subcarriers=1200,
        cp_length=72,
        modulation='QPSK',
        channel_type='awgn'
    )
    snr_range = np.arange(0, 25, 2)
    ber_list, evm_list = sim3.run_snr_sweep(snr_range, num_bits=30000, num_trials=3)
    sim3.plot_results(snr_range, ber_list, evm_list)
    
    # 测试4: 16QAM SNR扫描
    print("\n[测试4] SNR扫描 - 16QAM")
    sim4 = NREnhancedSimulation(
        num_subcarriers=1200,
        cp_length=72,
        modulation='16QAM',
        channel_type='awgn'
    )
    ber_list_16qam, evm_list_16qam = sim4.run_snr_sweep(snr_range, num_bits=30000, num_trials=3)
    sim4.plot_results(snr_range, ber_list_16qam, evm_list_16qam)
    
    print("\n" + "=" * 60)
    print("仿真完成！生成的文件：")
    print("  - constellation_QPSK_SNR15.png")
    print("  - constellation_16QAM_SNR25.png")
    print("  - channel_response_rayleigh.png")
    print("  - simulation_results_enhanced.png")
    print("=" * 60)


if __name__ == '__main__':
    main()
