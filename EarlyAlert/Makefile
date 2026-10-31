# EarlyAlert: Makefile
# Author: Vasanthkumar A | Knowledge Institute of Technology

PYTHON = python
SEED = 42

.PHONY: all setup download audit train test predict clean

all: setup download audit train test

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

download:
	$(PYTHON) src/load_data.py

audit:
	$(PYTHON) src/validate.py

train:
	$(PYTHON) src/train.py --seed $(SEED)

test:
	$(PYTHON) -m pytest -v

predict:
	$(PYTHON) src/predict.py --input data/sample_students/high_risk_student.json

clean:
	rm -rf .pytest_cache __pycache__ src/__pycache__ tests/__pycache__
