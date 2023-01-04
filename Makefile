install:
	python setup.py install

uninstall:
	pip uninstall pynance -y

update:
	make uninstall
	make install