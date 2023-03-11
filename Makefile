install:
	python setup.py install

uninstall:
	pip uninstall pynance -y

update:
	make uninstall
	make install

techus_run:
	python run.py -n techus_forecast -k train
	python run.py -n techus_regr -k train
	python run.py -n techus_coint -k coint
	python run.py -n techus_infer -k infer

luxefr_run:
	python run.py -n luxefr_forecast -k train
	python run.py -n luxefr_regr -k train
	python run.py -n luxefr_coint -k coint
	python run.py -n luxefr_infer -k infer

deffr_run:
	python run.py -n deffr_forecast -k train
	python run.py -n deffr_regr -k train
	python run.py -n deffr_coint -k coint
	python run.py -n deffr_infer -k infer

runs:
	make techus_run
	make luxefr_run
	make deffr_run

clean:
	rm -rf pynance.egg-info/
	rm -rf **__pycache__/
	rm -rf dist/
	rm -rf build/

make crypto_live:
	python run.py -n crypto -k crypto