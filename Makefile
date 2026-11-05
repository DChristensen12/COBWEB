SYSTEMS = hubbard modes
MOLECULES = h4 lih h2o
VIBRATIONS = water formic_acid

hamiltonians:
	for s in $(SYSTEMS); do python src/hamiltonians.py -s $$s; done

molecules:
	for m in $(MOLECULES); do python src/molecules.py -m $$m; done

circuits: molecules
	for m in $(MOLECULES); do python src/circuits.py -m $$m; done

compare: hamiltonians
	for s in $(SYSTEMS); do python src/compare.py -s $$s; done

compare-molecules: circuits
	for m in $(MOLECULES); do python src/compare.py -s $$m; done

vibrations:
	for v in $(VIBRATIONS); do python src/vibrations.py -m $$v; done

compare-vibrations: vibrations
	for v in $(VIBRATIONS); do python src/compare.py -s $$v; done

test:
	pytest -q src

format:
	isort --profile black --known-local-folder utils --known-local-folder hamiltonians --known-local-folder molecules --known-local-folder circuits --known-local-folder vibrations src
	black src

# Force fields take minutes to compute, so clean leaves results/forcefields alone
clean:
	rm -rf results/hamiltonians results/compare results/circuits

.PHONY: hamiltonians molecules circuits compare compare-molecules vibrations compare-vibrations test format clean
