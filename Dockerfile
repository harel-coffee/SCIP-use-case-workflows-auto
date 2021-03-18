# https://pythonspeed.com/articles/activate-conda-dockerfile/

FROM frolvlad/alpine-miniconda3:python3.7

RUN apk add bash

WORKDIR /app

# install mamba
RUN conda install mamba -n base -c conda-forge

# Create the environment:
COPY environment.yml .
RUN mamba env create -f environment.yml

# flowkit related dependencies
RUN apk add alpine-sdk cmake

# Make RUN commands use the new environment:
SHELL ["mamba", "run", "-n", "env", "/bin/bash", "-c"]

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY README.md .
COPY setup.py .
COPY settings.ini .
COPY ehv/ ./ehv
RUN pip install -e .

COPY jupyter_lab_config.py .

EXPOSE 8888
EXPOSE 8787

CMD ["conda", "run", "-n", "env", "--cwd", "/app", "--no-capture-output", "jupyter", "lab", "--no-browser", "--allow-root", "--port=8888", "--ip=0.0.0.0", "--config=/app/jupyter_lab_config.py"]
