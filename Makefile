collect:
	mkdir -p data/trajectories
	python3 data_collectionINVP.py

train:
	mkdir -p data/models
	python3 train_WM.py

train_dynamics:
	mkdir -p data/models
	python3 train_dynamics.py

visualize:
	python3 visualize.py

visualize_dynamics:
	python3 visualize_dynamics.py

all: collect train train_dynamics visualize
