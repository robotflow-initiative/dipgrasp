FROM rekcod.robotflow.ai/davidliyutong/idekube-container:coder-base-v0.2.2

# Install project dependencies
RUN apt-get update && \
    apt-get install -y minizip libc6-dev && \
    apt-get autoclean -y && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# Create conda environment
RUN . /opt/miniconda3/etc/profile.d/conda.sh && \
    # fix conda init issue
    conda create -n dipgrasp -y python=3.10

# Configure Python environment
COPY ./requirements.txt /tmp/requirements.txt
RUN . /opt/miniconda3/etc/profile.d/conda.sh && \
    conda activate dipgrasp && \
    pip install -U pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt --no-deps && \
    rm /tmp/requirements.txt

# copy the code
COPY . /opt/dipgrasp

RUN chown idekube:idekube /opt/dipgrasp && \
    echo "if [ ! -f /home/idekube/.bashrc ]; then sudo -u idekube /opt/miniconda3/bin/conda init bash; fi" >> /etc/idekube/startup.bash/00-conda-init.sh && \
    sudo -u idekube /opt/miniconda3/bin/conda init bash

