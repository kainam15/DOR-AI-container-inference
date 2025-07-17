'''
Silero TTS 演示脚本

前置条件：
  - Python 3.7+
  - 安装依赖：
      pip install torch soundfile

使用步骤：
  1. 在项目目录下创建文件 silero_tts_demo.py 并粘贴以下代码。
  2. 根据需要修改 language、speaker、device 和 output_dir。
  3. 运行脚本：
       python silero_tts_demo.py
  4. 脚本会在指定目录下生成 output.wav，使用任意音频播放器即可试听合成结果。

示例代码开始：
'''  
import torch
from pathlib import Path
import soundfile as sf

# ——————————————————————————————————
# 1. 配置区域
# ——————————————————————————————————
language = 'en'        # 语种: 'en' 英语, 'de' 德语, 'ru' 俄语 等
speaker = 'lj_16khz'   # 预训练说话人: 例如 lj_16khz, v3_en_jenny 等
device = torch.device('cpu')  # 计算设备: 'cpu' 或 'cuda' (有 GPU 时)

# 指定输出目录
output_dir = Path(r"E:\GitHubProjects\DOR：AI容器推理时画像组\DOR-AI-container-inference\data\output_audio")       # 自定义输出文件夹
output_dir.mkdir(parents=True, exist_ok=True)  # 如果不存在则创建

# ——————————————————————————————————
# 2. 从 Torch Hub 加载模型
# ——————————————————————————————————
model, symbols, sample_rate, example_text, apply_tts = torch.hub.load(
    repo_or_dir='snakers4/silero-models',
    model='silero_tts',
    language=language,
    speaker=speaker
)
model = model.to(device)

# ——————————————————————————————————
# 3. 准备文本输入
# ——————————————————————————————————
# 可以将 example_text 换成自定义字符串列表
texts = ["hello, my name is Wu Kai Nam, i am currently studying at Jinan University"]  # 例如: ["你好，这是一次 TTS 测试！"]

# ——————————————————————————————————
# 4. 合成语音
# ——————————————————————————————————
audio_list = apply_tts(
    texts=texts,
    model=model,
    sample_rate=sample_rate,
    symbols=symbols,
    device=device
)

# ——————————————————————————————————
# 5. 保存为 WAV 文件到指定目录
# ——————————————————————————————————
output_path = output_dir / 'output.wav'  # 在 output_audio 文件夹下
waveform = audio_list[0].cpu().numpy()  # 转成 NumPy
sf.write(str(output_path), waveform, sample_rate)
print(f"已将合成语音保存到: {output_path.resolve()}")
