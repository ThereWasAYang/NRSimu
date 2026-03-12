"""
NR Simulation Visualization
完整的可视化功能：星座图、BER/EVM曲线、信道估计
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import os

from transmitter_fixed import NRTransmitterFixed
from receiver_fixed import NRReceiverFixed
from channel import NRChannel


class NRVisualizer:
    """NR仿真可视化类"""
    
    def __init__(self, modulation: str = '16QAM', channel_type: str = 'awgn'):
        self.modulation = modulation
        self.channel_type = channel_type
        self.tx = NRTransmitterFixed(modulation=modulation)
        self.rx = NRReceiverFixed(modulation=modulation)
        
        # 存储仿真结果
        self.results = {}
        
    def run_snr_sweep(self, snr_range: np.ndarray, num_bits: int = 16000) -> Dict:
        """
        SNR扫描仿真
        
        Returns:
            包含BER、EVM、信道估计结果的字典
        """
        print(f"\n{'='*60}")
        print(f"SNR扫描: {self.modulation} - {self.channel_type}")
        print(f"{'='*60}")
        
        ber_list = []
        evm_before_list = []
        evm_after_list = []
        channel_ests = []
        
        for snr_db in snr_range:
            channel = NRChannel(snr_db=snr_db, channel_type=self.channel_type)
            
            # 发射
            tx_signal, tx_bits = self.tx.transmit(num_bits)
            
            # 通过信道
            rx_signal, _ = channel.apply_channel(tx_signal)
            
            # 接收
            results = self.rx.receive(rx_signal, snr_db)
            
            # 计算BER
            ber = self.rx.calculate_ber(tx_bits, results['bits'])
            ber_list.append(ber)
            
            # 计算EVM
            evm_before = self.rx.calculate_evm(np.array(self.rx.constellation_before_eq))
            evm_after = self.rx.calculate_evm(np.array(self.rx.constellation_after_eq))
            evm_before_list.append(evm_before)
            evm_after_list.append(evm_after)
            
            print(f"SNR={snr_db:2d}dB | BER={ber:.2e} | "
                  f"EVM_before={evm_before:5.2f}% | EVM_after={evm_after:5.2f}%")
        
        self.results = {
            'snr_range': snr_range,
            'ber': ber_list,
            'evm_before': evm_before_list,
            'evm_after': evm_after_list,
            'channel_type': self.channel_type,
            'modulation': self.modulation
        }
        
        print(f"{'='*60}\n")
        return self.results
    
    def plot_constellation_comparison(self, save_path: str = None):
        """
        绘制均衡前后的星座图对比
        """
        if save_path is None:
            save_path = f'constellation_{self.modulation}_{self.channel_type}.png'
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        before = np.array(self.rx.constellation_before_eq[:1000])
        after = np.array(self.rx.constellation_after_eq[:1000])
        
        evm_before = self.rx.calculate_evm(before)
        evm_after = self.rx.calculate_evm(after)
        
        # 参考点
        if self.modulation == 'QPSK':
            ref_points = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
        elif self.modulation == '16QAM':
            ref_points = np.array([i+1j*q for i in [-3,-1,1,3] for q in [-3,-1,1,3]]) / np.sqrt(10)
        else:
            ref_points = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
        
        # 均衡前
        ax1 = axes[0]
        ax1.scatter(np.real(before), np.imag(before), 
                   c='blue', alpha=0.5, s=15, label='Received')
        ax1.scatter(np.real(ref_points), np.imag(ref_points), 
                   c='red', s=200, marker='*', edgecolors='black', 
                   linewidths=1.5, label='Ideal', zorder=5)
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=0.8)
        ax1.axvline(x=0, color='k', linestyle='--', alpha=0.3, linewidth=0.8)
        ax1.set_xlabel('In-Phase', fontsize=12)
        ax1.set_ylabel('Quadrature', fontsize=12)
        ax1.set_title(f'Before Equalization\nEVM = {evm_before:.2f}%', fontsize=13)
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        ax1.legend(fontsize=10)
        
        # 均衡后
        ax2 = axes[1]
        ax2.scatter(np.real(after), np.imag(after), 
                   c='green', alpha=0.5, s=15, label='Equalized')
        ax2.scatter(np.real(ref_points), np.imag(ref_points), 
                   c='red', s=200, marker='*', edgecolors='black', 
                   linewidths=1.5, label='Ideal', zorder=5)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=0.8)
        ax2.axvline(x=0, color='k', linestyle='--', alpha=0.3, linewidth=0.8)
        ax2.set_xlabel('In-Phase', fontsize=12)
        ax2.set_ylabel('Quadrature', fontsize=12)
        ax2.set_title(f'After Equalization\nEVM = {evm_after:.2f}%', fontsize=13)
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        ax2.legend(fontsize=10)
        
        plt.suptitle(f'{self.modulation} Constellation - {self.channel_type.upper()} Channel', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"星座图已保存: {save_path}")
    
    def plot_ber_evm_curves(self, save_path: str = None):
        """
        绘制BER和EVM随SNR变化的曲线
        """
        if save_path is None:
            save_path = f'ber_evm_curves_{self.modulation}_{self.channel_type}.png'
        
        if not self.results:
            print("请先运行SNR扫描")
            return
        
        snr_range = self.results['snr_range']
        ber = self.results['ber']
        evm_before = self.results['evm_before']
        evm_after = self.results['evm_after']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # BER曲线
        ax1 = axes[0]
        ax1.semilogy(snr_range, ber, 'b-o', linewidth=2.5, markersize=8, 
                    label=f'{self.modulation}')
        ax1.grid(True, which='both', linestyle='--', alpha=0.6)
        ax1.set_xlabel('SNR (dB)', fontsize=12)
        ax1.set_ylabel('Bit Error Rate (BER)', fontsize=12)
        ax1.set_title('BER vs SNR', fontsize=13, fontweight='bold')
        ax1.set_ylim([1e-6, 1])
        ax1.legend(fontsize=11)
        
        # 添加BER=10^-3和10^-5参考线
        ax1.axhline(y=1e-3, color='r', linestyle='--', alpha=0.5, label='BER=10^-3')
        ax1.axhline(y=1e-5, color='g', linestyle='--', alpha=0.5, label='BER=10^-5')
        
        # EVM曲线
        ax2 = axes[1]
        ax2.plot(snr_range, evm_before, 'b-s', linewidth=2.5, markersize=8, 
                label='Before Equalization', alpha=0.7)
        ax2.plot(snr_range, evm_after, 'g-^', linewidth=2.5, markersize=8, 
                label='After Equalization')
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.set_xlabel('SNR (dB)', fontsize=12)
        ax2.set_ylabel('EVM (%)', fontsize=12)
        ax2.set_title('EVM vs SNR', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=11)
        
        plt.suptitle(f'Performance Curves - {self.modulation} over {self.channel_type.upper()}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"BER/EVM曲线已保存: {save_path}")
    
    def plot_channel_estimation(self, snr_db: float = 20, save_path: str = None):
        """
        绘制导频符号信道估计结果的幅度和相位
        """
        if save_path is None:
            save_path = f'channel_estimation_{self.channel_type}_SNR{snr_db}.png'
        
        # 运行一次仿真获取信道估计
        channel = NRChannel(snr_db=snr_db, channel_type=self.channel_type)
        tx_signal, _ = self.tx.transmit(num_bits=8000)
        rx_signal, _ = channel.apply_channel(tx_signal)
        
        # 解调
        ofdm_symbols = self.rx.ofdm_demodulate(rx_signal)
        
        # 收集每个OFDM符号的信道估计
        num_ofdm = len(ofdm_symbols) // self.rx.num_subcarriers
        all_h_est = []
        
        for i in range(min(num_ofdm, 10)):  # 取前10个符号
            start = i * self.rx.num_subcarriers
            ofdm_symbol = ofdm_symbols[start:start+self.rx.num_subcarriers]
            h_est = self.rx.channel_estimate_ls(ofdm_symbol)
            all_h_est.append(h_est)
        
        all_h_est = np.array(all_h_est)
        h_mean = np.mean(all_h_est, axis=0)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 幅度响应 - 所有OFDM符号
        ax1 = axes[0, 0]
        for i, h_est in enumerate(all_h_est):
            ax1.plot(np.abs(h_est), alpha=0.5, linewidth=1, label=f'Symbol {i+1}' if i < 3 else '')
        ax1.plot(np.abs(h_mean), 'r-', linewidth=2.5, label='Average')
        ax1.set_xlabel('Subcarrier Index', fontsize=11)
        ax1.set_ylabel('Magnitude', fontsize=11)
        ax1.set_title('Channel Magnitude Response', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=9)
        
        # 2. 相位响应 - 所有OFDM符号
        ax2 = axes[0, 1]
        for i, h_est in enumerate(all_h_est):
            ax2.plot(np.angle(h_est), alpha=0.5, linewidth=1)
        ax2.plot(np.angle(h_mean), 'r-', linewidth=2.5, label='Average')
        ax2.set_xlabel('Subcarrier Index', fontsize=11)
        ax2.set_ylabel('Phase (rad)', fontsize=11)
        ax2.set_title('Channel Phase Response', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. 导频位置的幅度
        ax3 = axes[1, 0]
        pilot_indices = self.rx.pilot_indices
        pilot_mag = np.abs(h_mean[pilot_indices])
        ax3.stem(pilot_indices, pilot_mag, basefmt=' ')
        ax3.set_xlabel('Subcarrier Index', fontsize=11)
        ax3.set_ylabel('Magnitude', fontsize=11)
        ax3.set_title('Pilot Subcarrier Magnitude', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. 导频位置的相位
        ax4 = axes[1, 1]
        pilot_phase = np.angle(h_mean[pilot_indices])
        ax4.stem(pilot_indices, pilot_phase, basefmt=' ', linefmt='g-', markerfmt='go')
        ax4.set_xlabel('Subcarrier Index', fontsize=11)
        ax4.set_ylabel('Phase (rad)', fontsize=11)
        ax4.set_title('Pilot Subcarrier Phase', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'Channel Estimation - {self.channel_type.upper()} Channel at {snr_db}dB SNR', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"信道估计图已保存: {save_path}")
    
    def generate_all_plots(self, snr_range: np.ndarray = None):
        """
        生成所有图表
        """
        if snr_range is None:
            snr_range = np.arange(0, 26, 5)
        
        print("\n" + "="*60)
        print("生成所有可视化图表")
        print("="*60)
        
        # 1. 运行SNR扫描
        self.run_snr_sweep(snr_range)
        
        # 2. 绘制BER/EVM曲线
        self.plot_ber_evm_curves()
        
        # 3. 绘制星座图
        self.plot_constellation_comparison()
        
        # 4. 绘制信道估计（不同SNR）
        for snr in [10, 20]:
            self.plot_channel_estimation(snr_db=snr)
        
        print("="*60)
        print("所有图表生成完成！")
        print("="*60)


def main():
    """主函数 - 生成完整可视化"""
    
    # 场景1: 16QAM - AWGN
    print("\n" + "="*60)
    print("场景1: 16QAM - AWGN信道")
    print("="*60)
    vis1 = NRVisualizer(modulation='16QAM', channel_type='awgn')
    vis1.generate_all_plots(snr_range=np.arange(5, 26, 5))
    
    # 场景2: 16QAM - 瑞利衰落
    print("\n" + "="*60)
    print("场景2: 16QAM - 瑞利衰落信道")
    print("="*60)
    vis2 = NRVisualizer(modulation='16QAM', channel_type='rayleigh')
    vis2.generate_all_plots(snr_range=np.arange(5, 31, 5))
    
    # 场景3: QPSK - 瑞利衰落
    print("\n" + "="*60)
    print("场景3: QPSK - 瑞利衰落信道")
    print("="*60)
    vis3 = NRVisualizer(modulation='QPSK', channel_type='rayleigh')
    vis3.generate_all_plots(snr_range=np.arange(0, 26, 5))
    
    print("\n" + "="*60)
    print("生成的文件列表:")
    print("="*60)
    for f in os.listdir('.'):
        if f.endswith('.png'):
            print(f"  - {f}")
    print("="*60)


if __name__ == '__main__':
    main()
