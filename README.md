# 5G NR 接收机性能仿真

基于Python的5G NR（New Radio）通信系统链路级仿真平台，包含完整的发射机、信道模型和接收机实现。

## 项目结构

```
NRsimu/
├── README.md           # 项目说明
├── requirements.txt    # Python依赖
├── transmitter.py      # 发射机模块
├── channel.py          # 信道模块
├── receiver.py         # 接收机模块
└── simulation.py       # 仿真主程序
```

## 功能特性

### 发射机 (transmitter.py)
- **信道编码**: 支持卷积码（可扩展LDPC/Polar码）
- **数字调制**: QPSK、16QAM、64QAM、256QAM
- **OFDM调制**: 
  - 可配置子载波数量
  - 循环前缀(CP)插入
  - IFFT/FFT变换
- **同步前导**: Zadoff-Chu序列生成

### 信道模型 (channel.py)
- **AWGN信道**: 加性高斯白噪声
- **瑞利衰落**: 基于Jakes模型的多径衰落
- **莱斯衰落**: 含LOS分量的衰落信道
- **TDL信道**: 3GPP TR 38.901 TDL-A模型
- **动态SNR**: 支持实时调整信噪比

### 接收机 (receiver.py)
- **时间同步**: 基于前导序列的互相关检测
- **频率同步**: 基于循环前缀的频偏估计与校正
- **信道估计**: 
  - LS (最小二乘)估计
  - MMSE (最小均方误差)估计
- **信道均衡**: 
  - ZF (零迫)均衡
  - MMSE均衡
- **解调**: QPSK、16QAM软/硬判决

### 仿真功能 (simulation.py)
- **SNR扫描**: 自动遍历SNR范围，收集BER/EVM数据
- **调制对比**: 比较不同调制方式的性能
- **结果可视化**: 自动生成BER曲线和EVM曲线

## 快速开始

### 1. 安装依赖

```bash
cd /data/Python/NRsimu
pip install -r requirements.txt
```

### 2. 运行仿真

```bash
python simulation.py
```

### 3. 单独测试模块

```bash
# 测试发射机
python transmitter.py

# 测试信道
python channel.py

# 测试接收机
python receiver.py
```

## 使用示例

### 基本仿真

```python
from simulation import NRSimulation
import numpy as np

# 创建仿真实例
sim = NRSimulation(
    num_subcarriers=1200,
    cp_length=72,
    modulation='QPSK',
    channel_type='awgn'
)

# SNR扫描
snr_range = np.arange(0, 25, 2)
ber_list, evm_list = sim.run_snr_sweep(snr_range, num_bits=50000, num_trials=5)

# 绘制结果
sim.plot_results(snr_range, ber_list, evm_list)
```

### 自定义仿真

```python
from transmitter import NRTransmitter
from channel import NRChannel
from receiver import NRReceiver

# 初始化组件
tx = NRTransmitter(modulation='16QAM')
channel = NRChannel(snr_db=20, channel_type='rayleigh')
rx = NRReceiver(modulation='16QAM')

# 发射
tx_signal, tx_bits = tx.transmit(num_bits=10000)

# 通过信道
rx_signal, h = channel.apply_channel(tx_signal)

# 接收
rx_bits = rx.receive(rx_signal)

# 计算BER
ber = rx.calculate_ber(tx_bits, rx_bits)
print(f"BER: {ber:.2e}")
```

## 配置参数

### 发射机参数
- `num_subcarriers`: 子载波数量 (默认: 1200)
- `cp_length`: 循环前缀长度 (默认: 72)
- `modulation`: 调制方式 ('QPSK', '16QAM', '64QAM', '256QAM')

### 信道参数
- `snr_db`: 信噪比 (dB)
- `channel_type`: 信道类型 ('awgn', 'rayleigh', 'rician', 'tdl')
- `delay_spread`: 时延扩展 (秒)
- `doppler_freq`: 多普勒频移 (Hz)

### 接收机参数
- `num_subcarriers`: 子载波数量
- `cp_length`: 循环前缀长度
- `modulation`: 调制方式

## 性能指标

- **BER (Bit Error Rate)**: 误码率
- **EVM (Error Vector Magnitude)**: 误差向量幅度
- **SNR (Signal-to-Noise Ratio)**: 信噪比

## 扩展计划

- [ ] MIMO多天线系统
- [ ] 完整的LDPC/Polar编解码
- [ ] 更精确的信道估计算法
- [ ] 链路自适应 (AMC)
- [ ] 更多3GPP标准信道模型

## 参考标准

- 3GPP TS 38.211: NR物理信道与调制
- 3GPP TS 38.212: NR复用与信道编码
- 3GPP TS 38.213: NR物理层控制过程
- 3GPP TR 38.901: 信道模型研究

## 作者

NR通信仿真团队

## 许可证

MIT License
