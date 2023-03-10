install:
	python setup.py install

uninstall:
	pip uninstall pynance -y

update:
	make uninstall
	make install

basic_run:
	python run.py -n basic_pred -k train
	python run.py -n basic_regr -k train
	python run.py -n coint -k coint
	python run.py -n basic_infer -k infer

clean:
	rm -rf pynance.egg-info/
	rm -rf **__pycache__/
	rm -rf dist/
	rm -rf build/