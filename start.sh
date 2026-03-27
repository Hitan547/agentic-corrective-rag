#!/usr/bin/env bash

python ingestion.py
uvicorn main:app --host 0.0.0.0 --port 10000