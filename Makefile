#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = credit-risk-xai-thesis
PYTHON_VERSION = 3.14
PYTHON_INTERPRETER = python
FORCE ?= false
RAW ?= false
MACRO ?= false
FEATURES ?= true

FORCE_FLAG = $(if $(filter $(FORCE),true),--force,)
RAW_FLAG = $(if $(filter $(RAW),true),--raw,)
MACRO_FLAG = $(if $(filter $(MACRO),true),--macro,)
FEATURES_FLAG = $(if $(filter $(FEATURES),false),--no-features,--features)

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format





## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	@bash -c "if [ ! -z `which virtualenvwrapper.sh` ]; then source `which virtualenvwrapper.sh`; mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); else mkvirtualenv.bat $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); fi"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Build interim Serrano dataset (raw -> interim cache)
.PHONY: data-raw
data-raw:
	$(PYTHON_INTERPRETER) -m credit_risk_xai.data.make_dataset serrano $(FORCE_FLAG)

## Build macroeconomic summary (external -> interim cache)
.PHONY: data-macro
data-macro:
	$(PYTHON_INTERPRETER) -m credit_risk_xai.data.make_macro macro $(FORCE_FLAG)

## Engineer feature matrix (interim -> processed)
.PHONY: features
features:
	$(PYTHON_INTERPRETER) -m credit_risk_xai.features.engineer engineer $(FORCE_FLAG)

## Orchestrate pipeline (select stages via RAW=true MACRO=true FEATURES=false, etc.)
.PHONY: build
build:
	$(PYTHON_INTERPRETER) -m credit_risk_xai.pipelines.build_features run $(RAW_FLAG) $(MACRO_FLAG) $(FEATURES_FLAG) $(FORCE_FLAG)


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
