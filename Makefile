run:
	venv/bin/python3 data_collectionINVP.py

collect:
	mkdir -p data/trajectories
	venv/bin/python3 -c "from data_collectionINVP import get_data; get_data('safe'); get_data('unsafe')"
