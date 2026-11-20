# LiteOff-LLM reproducible environment (same image locally and on HPC via Apptainer).
#   docker build -t liteoff:latest -f env/Dockerfile .
#   apptainer build liteoff.sif docker-daemon://liteoff:latest     (or env/liteoff.def)
# Pinned: OMNeT++ 6.1.0, Veins 5.3.1 (+ env/veins-5.3.1-liteoff.patch), SUMO 1.25.0, Ollama 0.40.0.
FROM ubuntu:24.04
ARG DEBIAN_FRONTEND=noninteractive
ARG OMNETPP_VERSION=6.1.0
ARG VEINS_VERSION=5.3.1
ARG SUMO_VERSION=1.25.0
ARG OLLAMA_VERSION=0.40.0
ARG OLLAMA_SHA256=c94aa4156b3d13e64ebc2efe5ea53f015384c882be776e6695cfb37fb180d5ad
ARG WITH_TRAINING=0

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential clang lld bison flex perl python3 python3-pip python3-venv python3-dev \
      libxml2-dev zlib1g-dev libsqlite3-dev ca-certificates curl wget zstd git patch make \
      libgl1 libxrender1 libxext6 libfontconfig1 && rm -rf /var/lib/apt/lists/*

# Python (simulation side-car, analysis); SUMO from its official wheel (pinned version)
ENV VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH
RUN python3 -m venv /opt/venv && pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "eclipse-sumo==${SUMO_VERSION}" "traci==${SUMO_VERSION}" "sumolib==${SUMO_VERSION}" \
      numpy pandas scipy matplotlib pyyaml requests pytest
RUN if [ "$WITH_TRAINING" = "1" ]; then pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
      torch transformers peft accelerate datasets safetensors sentencepiece huggingface_hub; fi
RUN SH=$(python -c "import sumo,os;print(os.path.dirname(sumo.__file__))") && ln -s "$SH" /opt/sumo && \
    for b in sumo netconvert netgenerate duarouter; do ln -sf /opt/sumo/bin/$b /usr/local/bin/$b; done
ENV SUMO_HOME=/opt/sumo

# OMNeT++ (command line only)
WORKDIR /opt
RUN wget -q https://github.com/omnetpp/omnetpp/releases/download/omnetpp-${OMNETPP_VERSION}/omnetpp-${OMNETPP_VERSION}-linux-x86_64.tgz && \
    tar xzf omnetpp-${OMNETPP_VERSION}-linux-x86_64.tgz && rm omnetpp-*.tgz && mv omnetpp-6.1 omnetpp
WORKDIR /opt/omnetpp
RUN sed -i 's/^WITH_QTENV=yes/WITH_QTENV=no/; s/^WITH_OSG=yes/WITH_OSG=no/; s/^WITH_OSGEARTH=yes/WITH_OSGEARTH=no/; s/^WITH_SCAVE_PYTHON_BINDINGS=yes/WITH_SCAVE_PYTHON_BINDINGS=no/' configure.user && \
    bash -c "source setenv -f && ./configure WITH_QTENV=no WITH_OSG=no WITH_OSGEARTH=no WITH_LIBXML=yes && make -j$(nproc) MODE=release base"
ENV OMNETPP_HOME=/opt/omnetpp PATH=/opt/omnetpp/bin:$PATH LD_LIBRARY_PATH=/opt/omnetpp/lib

# Veins + LiteOff patch (TraCI API 22 for SUMO >= 1.24, tolerate late ACKs)
COPY env/veins-5.3.1-liteoff.patch /tmp/veins.patch
WORKDIR /opt
RUN wget -q https://github.com/sommer/veins/archive/refs/tags/veins-${VEINS_VERSION}.tar.gz && tar xzf veins-${VEINS_VERSION}.tar.gz && \
    rm veins-*.tar.gz && mv veins-veins-${VEINS_VERSION} veins && cd veins && patch -p1 < /tmp/veins.patch && \
    ./configure && make -j$(nproc) MODE=release
ENV VEINS_HOME=/opt/veins

# Ollama runtime (models are bind-mounted: OLLAMA_MODELS)
RUN wget -q https://github.com/ollama/ollama/releases/download/v${OLLAMA_VERSION}/ollama-linux-amd64.tar.zst && \
    echo "${OLLAMA_SHA256}  ollama-linux-amd64.tar.zst" | sha256sum -c - && \
    mkdir -p /opt/ollama && tar --zstd -xf ollama-linux-amd64.tar.zst -C /opt/ollama && rm ollama-linux-amd64.tar.zst && \
    ln -s /opt/ollama/bin/ollama /usr/local/bin/ollama
ENV OLLAMA_BIN=/opt/ollama/bin/ollama

# LiteOff code (the repository is also bind-mounted on HPC; this copy makes the image self-contained)
COPY . /opt/liteoff
WORKDIR /opt/liteoff
RUN rm -rf .venv sim/src/out sim/src/*.so sim/src/Makefile && make -C sim VEINS=/opt/veins && \
    make -C sim/tests && python3 scenarios/gen_scenarios.py nets
ENV LITEOFF_ROOT=/opt/liteoff PYTHONPATH=/opt/liteoff/llm
CMD ["bash"]
