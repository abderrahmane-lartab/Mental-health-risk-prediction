FROM apache/airflow:2.10.2-python3.11

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

COPY --chown=airflow:root requirements.txt /opt/airflow/requirements.txt
RUN pip install --no-cache-dir -r /opt/airflow/requirements.txt

COPY --chown=airflow:root src /opt/airflow/src
COPY --chown=airflow:root app /opt/airflow/app
COPY --chown=airflow:root airflow/dags /opt/airflow/dags
COPY --chown=airflow:root data /opt/airflow/data
COPY --chown=airflow:root models /opt/airflow/models
COPY --chown=airflow:root reports /opt/airflow/reports

ENV PYTHONPATH="/opt/airflow"
WORKDIR /opt/airflow
