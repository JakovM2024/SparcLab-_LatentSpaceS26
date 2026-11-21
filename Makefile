collect:
	mkdir -p data/trajectories
	python3 data_collectionINVP.py

# small held-out noisy-PD set in data/demo (separate seed, does not touch training data)
collect_demo:
	python3 -c "import numpy as np; from data_collectionINVP import get_data; get_data('noisy_pd', np.random.default_rng(1), n_episodes=10, out_dir='data/demo')"

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

train_margin:
	mkdir -p data/models
	python3 train_margin.py

visualize_margin:
	python3 visualize_margin.py

train_value:
	mkdir -p data/models
	python3 train_value.py

visualize_value:
	python3 visualize_value.py

all: collect train train_dynamics visualize
