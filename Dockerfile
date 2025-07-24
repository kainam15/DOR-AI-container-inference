#Cmd
#docker build -t silero-runtime-profiling .
#docker run --rm -v "%cd%\data:/app/data" silero-runtime-profiling



# ----------------------------
# 1. 基础镜像 & 环境变量
# ----------------------------
FROM python:3.12-slim AS base

# 模型缓存目录
ENV TORCH_HOME=/app/models

# 设定容器内工作目录
WORKDIR /app

# ----------------------------
# 2. 安装 Python 依赖
# ----------------------------

# 先拷贝 requirements.txt 并安装
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# ----------------------------
# 3. 拷贝项目脚本和数据目录
# ----------------------------
COPY . /app

# 创建输出目录
RUN mkdir -p data/output_audio

# ----------------------------
# 4. 预下载并信任 Silero 模型
# ----------------------------
RUN python - <<EOF
import torch
# 预下载支持英语的 TTS 模型（v3_en）
torch.hub.load(
    'snakers4/silero-models',
    'silero_tts',
    language='en',  # 使用英语 TTS
    speaker='v3_en',  # 使用支持英语的模型
    force_reload=True,
    trust_repo=True
)
# 预下载 STT 模型
torch.hub.load(
    'snakers4/silero-models',
    'silero_stt',
    trust_repo=True
)
EOF

# ----------------------------
# 5. 定义 Volume & Entrypoint
# ----------------------------
# 将 /app/data 设为可挂载点，宿主和容器共享 data/output_audio
VOLUME ["/app/data"]

# 默认命令：运行 TTS Demo；也可以在 `docker run` 时改为
#    python scripts/silero_stt_demo.py
ENTRYPOINT ["python"]
CMD ["scripts/silero_tts_demo.py"]
