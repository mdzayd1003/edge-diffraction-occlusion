.PHONY: install test quick spectrum study validate matlab-test matlab-study clean

install:
	pip install -r requirements.txt

test:
	python -m pytest -q

# Everything: one spectrum, the full 126-configuration study, the listening analysis.
# Seconds, no downloads, no MATLAB.
quick:
	python -m diffraction --results results/smoke spectrum --height 3 --receiver-distance 10
	python -m diffraction --results results/smoke study
	python -m diffraction --results results/smoke validate --ratings ../data/listening_test.csv

spectrum:
	python -m diffraction --results results/run spectrum

study:
	python -m diffraction --results results/run study

validate:
	python -m diffraction --results results/run validate --ratings ../data/listening_test.csv

# The MATLAB port, under Octave.
matlab-test:
	cd ../matlab/tests && octave --no-gui --quiet --eval "exit(run_tests())"

matlab-study:
	cd ../matlab && octave --no-gui --quiet --eval "run_parameter_study('../data/matlab_run');"

clean:
	rm -rf results/run .pytest_cache
