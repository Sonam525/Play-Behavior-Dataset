#!/bin/bash
# Cluster init script for notebooks 02 and 09 (OCR of the burned-in camera clock): installs the OCR toolchain on every node.
#
# Versions: the OCR runs that produced the dataset printed "opencv=4.11.0  pytesseract=4.1.1  tesseract_bin=tesseract 4.1.1".
# `apt-get install tesseract-ocr` gives Tesseract 4.1.1 on Ubuntu 22.04 (Databricks Runtime 15.4 LTS ML); newer images
# ship Tesseract 5.x. opencv-python is pinned because unpinned installs now resolve to 5.x, whose import aborted on
# Databricks. Also set OMP_THREAD_LIMIT=1 in the cluster's environment variables.
set -e
echo "Installing OCR toolchain..."
apt-get update -qq
apt-get install -y --no-install-recommends tesseract-ocr
pip install --no-cache-dir opencv-python==4.11.0.86 pillow pytesseract
echo "Done OCR install"
