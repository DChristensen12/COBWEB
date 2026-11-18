SYSTEMS = hubbard modes
MOLECULES = h4 lih h2o h6
VIBRATIONS = water formic_acid

hamiltonians:
	for s in $(SYSTEMS); do python src/hamiltonians.py -s $$s; done

molecules:
	for m in $(MOLECULES); do python src/molecules.py -m $$m; done

circuits: molecules
	for m in $(MOLECULES); do python src/circuits.py -m $$m; done

circuits-heavy-hex: molecules
	for m in $(MOLECULES); do python src/circuits.py -m $$m -p heavy-hex; done

# A noisy simulator of ibm_kingston, on this computer. It slows down fast with
# qubits, so only the two smallest molecules run. Real QPU runs go through
# src/hardware.py with --backend and --submit, one molecule at a time
FAKE_MOLECULES = lih h2o

hardware-fake:
	for m in $(FAKE_MOLECULES); do python src/hardware.py -m $$m -b fake_kingston; done

compare: hamiltonians
	for s in $(SYSTEMS); do python src/compare.py -s $$s; done

compare-molecules: circuits
	for m in $(MOLECULES); do python src/compare.py -s $$m; done

vibrations:
	for v in $(VIBRATIONS); do python src/vibrations.py -m $$v; done

compare-vibrations: vibrations
	for v in $(VIBRATIONS); do python src/compare.py -s $$v; done

# Stage 5 reruns every comparison so it picks up the heavy-hex circuits, the
# noisy simulator shots and the fundamentals
compare-real: compare circuits circuits-heavy-hex hardware-fake vibrations
	for m in $(MOLECULES); do python src/compare.py -s $$m --overwrite; done
	for v in $(VIBRATIONS); do python src/compare.py -s $$v --overwrite; done

test:
	pytest -q src

format:
	isort --profile black --known-local-folder utils --known-local-folder hamiltonians --known-local-folder molecules --known-local-folder circuits --known-local-folder vibrations --known-local-folder hardware src
	black src

# Force fields take minutes to compute, so clean leaves results/forcefields alone
clean:
	rm -rf results/hamiltonians results/compare results/circuits results/hardware

.PHONY: hamiltonians molecules circuits circuits-heavy-hex hardware-fake compare compare-molecules vibrations compare-vibrations compare-real test format clean
