import time, csv, torch, soundfile as sf
from pathlib import Path

# —— 配置区 —— 
language = 'en'
speaker  = 'lj_16khz'
device   = torch.device('cpu')  # 或 'cuda'
sample_rate = 48000

# —— 加载模型 —— 
model = torch.hub.load(
    'snakers4/silero-models','silero_tts',
    language=language,speaker=speaker,
    trust_repo=True
)
model = model if not isinstance(model, (tuple,list)) else model[0]
model = model.to(device)

# —— 准备输出 & 读入测试文本 —— 
root       = Path(__file__).parent.parent
output_dir = root/'data'/'metrics'
output_dir.mkdir(exist_ok=True,parents=True)

texts = (root/'data'/'test_texts'/'sentences.txt').read_text().splitlines()

# —— 逐句测试并记录耗时 —— 
with open(output_dir/'tts_benchmark.csv','w',newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['idx','length','time_ms'])
    for i,txt in enumerate(texts,1):
        start = time.perf_counter()
        audio = model.apply_tts(txt, sample_rate, device)
        elapsed_ms = (time.perf_counter()-start)*1000
        writer.writerow([i, len(txt), f"{elapsed_ms:.2f}"])
        # 可选：把每条音频也存盘
        sf.write(str(root/'data'/'output_audio'/f"test_{i}.wav"),
                 audio[0].cpu().numpy(),sample_rate)
        print(f"[{i}/{len(texts)}] len={len(txt)} → {elapsed_ms:.1f}ms")
