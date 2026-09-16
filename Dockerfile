FROM python:3.11-slim

WORKDIR /app

COPY requirements-runner.txt .
RUN pip install --no-cache-dir -r requirements-runner.txt

COPY personas/ ./personas/
COPY scenarios/ ./scenarios/
COPY runner/ ./runner/

RUN mkdir -p results

WORKDIR /app/runner
