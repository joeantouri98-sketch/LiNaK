// Feshbach page physics: the derived quantities of qali.physics.feshbach (same formulas, SI inside).
// a in a0, B in G, T in uK, densities in cm^-3. Mirrors feshbach.py; tests/test_feshbach_page.py
// checks the two against each other.
const FP = (() => {
  const HBAR = 1.054571817e-34, H = 6.62607015e-34, KB = 1.380649e-23, AMU = 1.66053906660e-27, A0 = 5.29177210903e-11;
  const MU_B = 9.2740100783e-24;
  // f(b) = int_0^inf u e^-u /(1+b u) du : trapezoid in ln u on 1e-10..60 (as thermal_factor)
  const NQ = 700, LU0 = Math.log(1e-10), LU1 = Math.log(60), DLU = (LU1 - LU0) / (NQ - 1);
  const QU = new Float64Array(NQ), QW = new Float64Array(NQ);
  for (let i = 0; i < NQ; i++) { const u = Math.exp(LU0 + i * DLU); QU[i] = u; QW[i] = u * u * Math.exp(-u) * DLU * (i === 0 || i === NQ - 1 ? 0.5 : 1); }
  function thermalFactor(b) { let s = 0; for (let i = 0; i < NQ; i++) s += QW[i] / (1 + b * QU[i]); return s; }
  const mu = m => 0.5 * m * AMU;
  const thermalK = (T, m) => Math.sqrt(2 * mu(m) * KB * T * 1e-6) / HBAR * A0;           // a0^-1
  const vRel = (T, m) => Math.sqrt(8 * KB * T * 1e-6 / (Math.PI * mu(m)));                // m/s
  function thermalSigma(a, T, m, bos) { const k = thermalK(T, m), C = (bos ? 8 : 4) * Math.PI; return C * a * a * thermalFactor(k * k * a * a); }  // a0^2
  const sigma0 = (a, bos) => (bos ? 8 : 4) * Math.PI * a * a;
  const unitarity = (T, m, bos) => { const k = thermalK(T, m); return (bos ? 8 : 4) * Math.PI / (k * k); };   // C/k_T^2, a0^2
  // elastic collision rate per atom (s^-1): n <sigma v_rel>, n in cm^-3
  const elasticRate = (a, T, m, bos, n) => n * 1e6 * thermalSigma(a, T, m, bos) * A0 * A0 * vRel(T, m);
  // E_b/h (Hz) of the shallow dimer; abar = 0 for the universal form; NaN when there is none
  function bindingHz(a, m, abar) { const e = a - (abar || 0); return e > 0 ? HBAR * HBAR / (m * AMU * Math.pow(e * A0, 2)) / H : NaN; }
  const kF = n => Math.cbrt(6 * Math.PI * Math.PI * n * 1e6) * A0;                          // a0^-1, n per spin state
  const TF = (n, m) => { const k = kF(n) / A0; return HBAR * HBAR * k * k / (2 * m * AMU * KB); };  // K
  // L3 = 3 C hbar a^4/m (cm^6/s), capped at 36 sqrt3 pi^2 hbar^5/(m^3 (kT)^2) when T is given
  function L3(a, m, C, T) {
    const M = m * AMU, aa = a * A0; let L = 3 * C * HBAR * aa * aa * aa * aa / M;
    if (T) L = Math.min(L, 36 * Math.sqrt(3) * Math.PI * Math.PI * Math.pow(HBAR, 5) / (M * M * M * Math.pow(KB * T * 1e-6, 2)));
    return L * 1e12;
  }
  const Ebar = (abar, m) => HBAR * HBAR / (m * AMU * Math.pow(abar * A0, 2));              // J
  function sRes(abg, D, dmu, abar, m) { return Math.abs(abg / abar) * Math.abs(dmu * MU_B * D * 1e-4) / Ebar(abar, m); }
  return { HBAR, H, KB, AMU, A0, MU_B, thermalFactor, thermalK, vRel, thermalSigma, sigma0, unitarity, elasticRate, bindingHz, kF, TF, L3, Ebar, sRes };
})();
if (typeof module !== 'undefined') module.exports = FP;
