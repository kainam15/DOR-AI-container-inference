import os
import torch, zipfile, torchaudio
from glob import glob

# 1. 路径设置
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
audio_file = os.path.join(BASE_DIR, 'data', '237-134500-0009.flac')

# 2. 拉取模型
device = torch.device('cpu')  # 如果想跑 GPU，可改为 'cuda'

model, decoder, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-models',
    model='silero_stt',
    language='en',      # 支持 'en' / 'de' / 'es'
    device=device
)

# 3. 拆包工具
read_batch, split_into_batches, read_audio, prepare_model_input = utils

# 4. 准备输入
test_files = [os.path.join(BASE_DIR, 'data', '237-134500-0009.flac')]      # 单文件
#test_files = glob(os.path.join(BASE_DIR, 'data', '*.wav'))                #处理目录里所有 WAV：
batches = split_into_batches(test_files, batch_size=10)
input = prepare_model_input(read_batch(batches[0]), device=device)

# 5. 模型推理 & 打印
output = model(input)
for example in output:
    print(decoder(example.cpu()))
