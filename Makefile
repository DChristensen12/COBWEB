SYSTEMS = hubbard modes
MOLECULES = h4 lih h2o

hamiltonians:
	for s in $(SYSTEMS); do python src/hamiltonians.py -s $$s; done

molecules:
	for m in $(MOLECULES); do python src/molecules.py -m $$m; done

compare: hamiltonians
	for s in $(SYSTEMS); do python src/compare.py -s $$s; done

compare-molecules: molecules
	for m in $(MOLECULES); do python src/compare.py -s $$m; done

test:
	pytest -q src

format:
	isort --profile black --known-local-folder utils --known-local-folder hamiltonians --known-local-folder molecules src
	black src

clean:
	rm -rf results/hamiltonians results/compare

.PHONY: hamiltonians molecules compare compare-molecules test format clean
