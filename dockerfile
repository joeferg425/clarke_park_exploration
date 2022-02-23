FROM python:3.9.5

RUN pip install dash==2.0.0 dash_bootstrap_components==1.0.3 dash_daq==0.5.0 numpy==1.20.3

RUN mkdir -p /opt/code/assets
COPY ./clarke_park_3d.py /opt/code/
COPY ./assets/mathjax.js /opt/code/assets/

EXPOSE 8050
ENTRYPOINT [ "python","/opt/code/clarke_park_3d.py" ]