# COBWEB

Here, we introduce COBWEB, which combines two directions in quantum computational chemistry: (1) subspace sampling, in which configurations sampled from a state prepared on a quantum computer define the subspace for a selected configuration interaction (CI) calculation, and (2) infrared (IR) spectra, in which the same calculation applied to a molecule’s vibrations predicts the frequencies measured by an IR spectrometer. Full CI diagonalizes the Hamiltonian over every configuration, which is only feasible for small systems, so selected CI keeps a subset, and the result depends on how that subset is chosen. We compare subspace sampling against classical selection by second-order perturbation theory on both electronic (fermionic) and vibrational (bosonic) Hamiltonians. Because both are written in occupation-number bases, a single selection engine handles both, which lets us test whether the comparison carries over from electrons to vibrations. The current code uses two model systems small enough to check every result against exact diagonalization: a 6-site Hubbard chain (400 configurations) and three coupled Morse modes (512 configurations). Applying the comparison to real molecules, and checking predicted IR spectra against measured data, is the next stage.


# Thank you to all of those who contributed!

<a href="https://github.com/DChristensen12/QCB_Quantum_Chemistry_Project/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=DChristensen12/QCB_Quantum_Chemistry_Project&max=100&columns=10&anon=1" width="60" alt="Contributors" />
</a>

And a special thank you to our advisor Wendy Billings!