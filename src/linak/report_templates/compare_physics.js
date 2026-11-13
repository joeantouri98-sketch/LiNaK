// Compare page physics: linak.physics.laser_cooling and the α(λ) sum of polarizability_core,
// same formulas, SI inside. tests/test_compare_page.py checks the two against each other in node.
const CP = (() => {
  const HBAR = 1.054571817e-34, H = 6.62607015e-34, KB = 1.380649e-23, AMU = 1.66053906660e-27, C = 2.99792458e8;
  const KB_EV = 8.617333262e-5, EV_J = 1.602176634e-19, NM_TO_AU = 1239.841984 / 27.211386245988;
  const k = wl => 2 * Math.PI / (wl * 1e-9);
  const rate = (G, s, d) => { const x = 2 * d / G; return 0.5 * G * s / (1 + s + x * x); };             // photons/s
  const force = (v, G, s, d, wl) => HBAR * k(wl) * (rate(G, s, d - k(wl) * v) - rate(G, s, d + k(wl) * v));   // N
  const temp = (G, s, d, n) => { if (!(d < 0)) return Infinity; const x = 2 * d / G; return HBAR * G / (4 * KB) * (1 + (n || 2) * s + x * x) / Math.abs(x); };  // K
  const TD = G => HBAR * G / (2 * KB);
  const vrec = (wl, m) => HBAR * k(wl) / (m * AMU);
  const Trec = (wl, m) => Math.pow(HBAR * k(wl), 2) / (m * AMU * KB);
  const amax = (G, wl, m) => HBAR * k(wl) * G / (2 * m * AMU);
  const vcap = (G, wl) => G / k(wl);
  const isatCyc = (G, wl) => Math.PI * H * C * G / (3 * Math.pow(wl * 1e-9, 3)) * 0.1;                     // mW/cm²
  const vp = (T, m) => Math.sqrt(2 * KB * T / (m * AMU));
  const slower = (v, G, wl, m, eta) => v * v / (2 * (eta || 0.5) * amax(G, wl, m));
  // α(λ) of a ground state from [ΔE_au, s·f] pairs (a.u.); NaN within 1e-4 ΔE of a pole, as alpha_real
  function alpha(wl, tr) { const w = NM_TO_AU / wl; let a = 0; for (const [dE, sf] of tr) { const den = dE * dE - w * w; if (Math.abs(den) < 1e-4 * Math.abs(dE)) return NaN; a += sf / den; } return a; }
  const omegaNm = wl => NM_TO_AU / wl;
  // D-line table at temperature T (the comparison table's columns)
  function dopplerFwhmHz(wl, T, m) { const nu = C / (wl * 1e-9), mc2 = m * AMU * C * C / EV_J; return nu * Math.sqrt(8 * Math.log(2) * KB_EV * T / mc2); }
  function isatFromSigma(wl, sig_m2, tau_s) { return H * C / (wl * 1e-9) / (2 * sig_m2 * tau_s) * 0.1; }       // mW/cm²
  return { HBAR, H, KB, AMU, C, NM_TO_AU, k, rate, force, temp, TD, vrec, Trec, amax, vcap, isatCyc, vp, slower, alpha, omegaNm, dopplerFwhmHz, isatFromSigma };
})();
if (typeof module !== 'undefined') module.exports = CP;
