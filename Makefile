SYSTEMS = hubbard modes

hamiltonians:
	for s in $(SYSTEMS); do python src/hamiltonians.py -s $$s; done

compare: hamiltonians
	for s in $(SYSTEMS); do python src/compare.py -s $$s; done

test:
	pytest -q src

format:
	isort --profile black --known-local-folder utils --known-local-folder hamiltonians src
	black src

clean:
	rm -rf results/hamiltonians results/compare

.PHONY: hamiltonians compare test format clean
