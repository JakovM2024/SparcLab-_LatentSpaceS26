collect:
	mkdir -p data/trajectories
	python3 data_collectionINVP.py

collect_demo:
	python3 -c "from data_collectionINVP import get_data; get_data('demo')"

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
