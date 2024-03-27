FROM jupyter/scipy-notebook:b72e40b2e3b1 as common

COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt

RUN mkdir /opt/conda/share/jupyter/lab/settings && echo '{ "@jupyterlab/apputils-extension:themes": { "theme": "JupyterLab Dark" } }' > /opt/conda/share/jupyter/lab/settings/overrides.json
