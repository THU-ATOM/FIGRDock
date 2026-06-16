FROM dptechnology/unicore:0.0.1-pytorch1.11.0-cuda11.3

RUN pip install setuptools wheel twine

RUN pip install -U numpy>=1.23 scipy

RUN pip install biopandas networkx

RUN pip install rdkit==2024.3.5

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        linux-libc-dev \
        libc6-dev && \
    rm -rf /var/lib/apt/lists/*

# 2) 安装 torch-geometric 及 4 个扩展库（全部用官方预编译 wheel）
#    PyTorch 1.11.0 + CUDA 11.3 对应索引：
ARG TORCH_CUDA="cu113"
ARG PYG_VERSION="2.3.1"        # 可根据需要改版本号
RUN pip install --no-cache-dir torch-scatter torch-sparse==0.6.15 torch-cluster torch-spline-conv \
        -f https://data.pyg.org/whl/torch-1.11.0+${TORCH_CUDA}.html && \
    pip install --no-cache-dir torch-geometric==${PYG_VERSION}
# pip install torch-scatter==2.1.1 torch-sparse==0.6.15 torch-cluster==1.6.0 torch-spline-conv torch-geometric==2.0.4 -f https://data.pyg.org/whl/torch-1.11.0+cu117.html

RUN ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* && \
    pip cache purge
    
# 复制代码到工作目录
COPY . /app
WORKDIR /app
