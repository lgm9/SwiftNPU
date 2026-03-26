.DEFAULT_GOAL := all

BUILD_DIR := build
PROFILE_BUILD_DIR := build_tracy
SOURCE_DIR := source

TARGET_PROJECT1 := $(SOURCE_DIR)/SwiftNPU/SwiftNPU

#venv
VENV_DIR:=./.venv
PYTHON := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip
REQUIREMENTS := requirements.txt
venv: $(VENV_DIR)/bin/activate

$(VENV_DIR)/bin/activate: $(REQUIREMENTS)
	python3 -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip
	$(PIP) install -r $(REQUIREMENTS)
	touch $(VENV_DIR)/bin/activate

all: configure build

configure:
	cmake -S . -B $(BUILD_DIR)

build:
	cmake --build $(BUILD_DIR) -j

clean:
	rm -rf $(BUILD_DIR) $(PROFILE_BUILD_DIR)
	
distclean: clean
	rm -rf .cpmcache

rebuild: clean all

run_SwiftNPU: venv
	./$(BUILD_DIR)/$(TARGET_PROJECT1) \
	--out_dir results/swiftNPU \
	--out results.txt \
	--visualize_option 0 \
	--debug_option 0 \
	--alloc_algorithm 0

.PHONY: all configure build clean distclean rebuild run_SwiftNPU