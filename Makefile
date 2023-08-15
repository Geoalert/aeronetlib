# Makefile

# Variables
PROJECT_NAME = aeronet
LIBRARIES = aeronet_raster aeronet_vector aeronet_convert

.PHONY: build upload clean

# Build all libraries
build:
	@for lib in $(LIBRARIES); do \
		echo "Building $$lib"; \
		cd $$lib; \
		python3 setup.py build sdist bdist_wheel --universal; \
		cd ..;\
	done
	@echo "Building $(PROJECT_NAME) library"
	python3 setup.py build sdist bdist_wheel

# Upload all packages to PyPI using twine
upload:
	@for lib in $(LIBRARIES); do \
		echo "Uploading $$lib"; \
		twine upload --skip-existing $$lib/dist/*; \
	done
	@echo "Uploading $(PROJECT_NAME) library"
	twine upload --skip-existing dist/*

# Clean up build artifacts for all libraries
clean:
	@for lib in $(LIBRARIES); do \
		echo "Cleaning $$lib"; \
		cd $$lib; \
		rm -rf build dist *.egg-info; \
		cd ..; \
	done
	@echo "Cleaning $(PROJECT_NAME) library"
	rm -rf build dist *.egg-info

# Install all the requirements (activate venv first!)
prepare:
	@for lib in $(LIBRARIES); do \
		echo "Installing requirements for $$lib"; \
		cd $$lib; \
		pip install -r requirements.txt; \
		cd ..; \
	done

# Install all the requirements (activate venv first!)
test:
	@for lib in $(LIBRARIES); do \
		echo "Testing $$lib"; \
		python3 -m pytest $$lib/test; \
	done
