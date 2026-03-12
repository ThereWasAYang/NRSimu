"""
Test Fixed Implementation
测试修复后的发射机和接收机
"""

import numpy as np
from transmitter_fixed import NRTransmitterFixed
from receiver_fixed import NRReceiverFixed
from channel import NRChannel


def test_16qam_awgn():
    """测试16QAM在AWGN信道下的性能"""
    print("=" * 60)
    print("测试 16QAM - AWGN信道")
    print("=" * 60)
    
    tx = NRTransmitterFixed(modulation='16QAM')
    rx = NRReceiverFixed(modulation='16QAM')
    
    for snr_db in [10, 15, 20, 25]:
        channel = NRChannel(snr_db=snr_db, channel_type='awgn')
        
        # 发射
        tx_signal, tx_bits = tx.transmit(num_bits=16000)
        
        # 通过信道
        rx_signal, _ = channel.apply_channel(tx_signal)
        
        # 接收
        results = rx.receive(rx_signal, snr_db)
        
        # 计算BER
        ber = rx.calculate_ber(tx_bits, results['bits'])
        
        print(f"SNR = {snr_db}dB | BER = {ber:.2e} | Bits: {len(tx_bits)}->{len(results['bits'])}")
    
    # 绘制星座图
    rx.plot_constellation('constellation_16qam_fixed.png')
    print("=" * 60)


def test_qpsk_rayleigh():
    """测试QPSK在瑞利衰落信道下的性能"""
    print("\n" + "=" * 60)
    print("测试 QPSK - 瑞利衰落信道")
    print("=" * 60)
    
    tx = NRTransmitterFixed(modulation='QPSK')
    rx = NRReceiverFixed(modulation='QPSK')
    
    for snr_db in [10, 15, 20, 25]:
        channel = NRChannel(snr_db=snr_db, channel_type='rayleigh')
        
        tx_signal, tx_bits = tx.transmit(num_bits=8000)
        rx_signal, _ = channel.apply_channel(tx_signal)
        results = rx.receive(rx_signal, snr_db)
        
        ber = rx.calculate_ber(tx_bits, results['bits'])
        print(f"SNR = {snr_db}dB | BER = {ber:.2e}")
    
    rx.plot_constellation('constellation_qpsk_rayleigh.png')
    print("=" * 60)


def test_evm_improvement():
    """测试EVM改善"""
    print("\n" + "=" * 60)
    print("测试 EVM 改善")
    print("=" * 60)
    
    tx = NRTransmitterFixed(modulation='16QAM')
    rx = NRReceiverFixed(modulation='16QAM')
    channel = NRChannel(snr_db=20, channel_type='rayleigh')
    
    tx_signal, tx_bits = tx.transmit(num_bits=16000)
    rx_signal, _ = channel.apply_channel(tx_signal)
    results = rx.receive(rx_signal, snr_db=20)
    
    # 计算EVM
    evm_before = rx.calculate_evm(np.array(rx.constellation_before_eq))
    evm_after = rx.calculate_evm(np.array(rx.constellation_after_eq))
    
    print(f"EVM Before Equalization: {evm_before:.2f}%")
    print(f"EVM After Equalization:  {evm_after:.2f}%")
    
    if evm_after < evm_before:
        print(f"✓ 均衡改善: {evm_before - evm_after:.2f}%")
    else:
        print(f"✗ 均衡恶化: {evm_after - evm_before:.2f}%")
    
    print("=" * 60)


if __name__ == '__main__':
    test_16qam_awgn()
    test_qpsk_rayleigh()
    test_evm_improvement()
    print("\n所有测试完成！")
