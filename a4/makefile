# SCRIPT ENTRY POINT
all: p1/sample_output.txt

# MISC CONFIG
python=python

# GETTING DATA
# 	downloads data to a 'dataset' directory
# 	and provides a rule to link it into each subpart's directory
snapshot_url=http://www.cs.toronto.edu/~guerzhoy/321/proj4/char-rnn-snapshot.npz
local_file=dataset/char-rnn-snapshot.npz

$(local_file):
	mkdir -p dataset
	wget $(snapshot_url) $(local_file)

# P1 OUTPUT
p1/sample_output.txt: p1/p1.py
	$(python) $< > $@



