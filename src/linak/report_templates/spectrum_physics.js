// Spectrum page physics. Twin of physics/spectrum_core.py: the tests load this file in node and
// compare every function with the Python one. Units: Hz, vacuum nm, m² (cross-sections per atom).
const SP = (function () {
  const C = 2.99792458e8, H = 6.62607015e-34, KB = 1.380649e-23, AMU = 1.66053906660e-27, EV = 1.602176634e-19;
  const RE = 2.8179403262e-15, LINE_K = Math.PI * RE * C, SQ2PI = Math.sqrt(2 * Math.PI), SQ2 = Math.SQRT2;
  const LN2 = Math.LN2, FWHM_G = 2 * Math.sqrt(2 * LN2);

  // ---- complex helpers [re, im]
  const cadd = (a, b) => [a[0] + b[0], a[1] + b[1]];
  const cmul = (a, b) => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];
  const cdiv = (a, b) => { const d = b[0] * b[0] + b[1] * b[1]; return [(a[0] * b[0] + a[1] * b[1]) / d, (a[1] * b[0] - a[0] * b[1]) / d]; };
  const cexp = a => { const e = Math.exp(a[0]); return [e * Math.cos(a[1]), e * Math.sin(a[1])]; };
  const cpoly = (t, c) => { let r = [c[c.length - 1], 0]; for (let k = c.length - 2; k >= 0; k--) r = cadd(cmul(r, t), [c[k], 0]); return r; };
  // Faddeeva function w(x + iy), y ≥ 0: Humlíček (1982) W4, relative accuracy ~1e-4.
  const R3N = [16.4955, 20.20933, 11.96482, 3.778987, 0.5642236], R3D = [16.4955, 38.82363, 39.27121, 21.69274, 6.699398, 1.0];
  const R4N = [36183.31, -3321.9905, 1540.787, -219.0313, 35.76683, -1.320522, 0.56419];
  const R4D = [32066.6, -24322.84, 9022.228, -2186.181, 364.2191, -61.57037, 1.841439, -1.0];
  function faddeeva(x, y) {
    const t = [y, -x], s = Math.abs(x) + y;
    if (s >= 15) return cdiv(cmul(t, [0.5641896, 0]), cadd([0.5, 0], cmul(t, t)));
    if (s >= 5.5) { const u = cmul(t, t); return cdiv(cmul(t, cadd([1.410474, 0], cmul(u, [0.5641896, 0]))), cadd([0.75, 0], cmul(u, cadd([3, 0], u)))); }
    if (y >= 0.195 * Math.abs(x) - 0.176) return cdiv(cpoly(t, R3N), cpoly(t, R3D));
    const u = cmul(t, t), e = cexp(u), q = cdiv(cmul(t, cpoly(u, R4N)), cpoly(u, R4D));
    return [e[0] - q[0], e[1] - q[1]];
  }
  // Area-normalised Voigt (1/Hz) and its dispersion partner: w(z)/(σ√2π).
  function voigtW(dnu, sigma, gamma) {
    const w = faddeeva(dnu / (sigma * SQ2), gamma / (sigma * SQ2)), k = 1 / (sigma * SQ2PI);
    return [w[0] * k, w[1] * k];
  }
  const voigt = (dnu, sigma, gamma) => voigtW(dnu, sigma, gamma)[0];
  // erf (Abramowitz & Stegun 7.1.26, |error| < 1.5e-7) for bin-averaged unresolved lines.
  function erf(x) { const s = x < 0 ? -1 : 1; x = Math.abs(x); const t = 1 / (1 + 0.3275911 * x);
    return s * (1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x)); }
  // Pseudo-Voigt (Thompson, Cox & Hastings 1987) CDF: puts a line's whole area into the bins it falls in.
  function pvParams(sigma, gamma) { const fg = FWHM_G * sigma, fl = 2 * gamma;
    const f = Math.pow(fg ** 5 + 2.69269 * fg ** 4 * fl + 2.42843 * fg ** 3 * fl ** 2 + 4.47163 * fg ** 2 * fl ** 3 + 0.07842 * fg * fl ** 4 + fl ** 5, 0.2);
    const r = fl / f, eta = 1.36603 * r - 0.47719 * r * r + 0.11116 * r ** 3; return { f, eta, sg: f / FWHM_G }; }
  const pvCdf = (d, p) => p.eta * (0.5 + Math.atan(2 * d / p.f) / Math.PI) + (1 - p.eta) * 0.5 * (1 + erf(d / (p.sg * SQ2)));

  const dopplerSigma = (nu, massAmu, T) => nu * Math.sqrt(KB * T / (massAmu * AMU * C * C));
  const instrumentSigma = (nu, fwhmNm) => { const lam = C / nu * 1e9; return fwhmNm / FWHM_G * C / (lam * lam) * 1e9; };
  const naturalFwhm = (tu, tl) => ((tu > 0 && isFinite(tu) ? 1 / tu : 0) + (tl > 0 && isFinite(tl) ? 1 / tl : 0)) / (2 * Math.PI);
  const stimulated = (nu, T) => T > 0 ? -Math.expm1(-H * nu / (KB * T)) : 1;
  function airIndex(lamVacNm) { if (lamVacNm < 200) return 1; const s2 = (1e3 / lamVacNm) ** 2;
    return 1 + 8.34254e-5 + 2.406147e-2 / (130 - s2) + 1.5998e-4 / (38.9 - s2); }
  const vacToAir = l => l / airIndex(l);
  function airToVac(la) { let l = la; for (let k = 0; k < 4; k++) l = la * airIndex(l); return l; }
  function vapourDensityCm3(vp, T) { if (!vp) return null; const [a, b] = T < vp.T_melt ? vp.solid : vp.liquid; return Math.pow(10, a - b / T) / (KB * T) * 1e-6; }
  function boltzmann(E, g, T) { const e0 = Math.min(...E), w = E.map((e, i) => isFinite(T) ? g[i] * Math.exp(-(e - e0) * EV / (KB * T)) : g[i]);
    const s = w.reduce((a, b) => a + b, 0); return w.map(v => v / s); }

  // ---- axis units: x ↔ ν (Hz), and |dν/dx| for spectral densities
  const UNITS = {
    nm: { to: x => C / (x * 1e-9), from: nu => C / nu * 1e9, jac: x => C / (x * x) * 1e9, label: 'Wavelength, vacuum (nm)' },
    air: { to: x => C / (airToVac(x) * 1e-9), from: nu => vacToAir(C / nu * 1e9), jac: x => { const l = airToVac(x); return C / (l * l) * 1e9 * airIndex(l); }, label: 'Wavelength, air (nm)' },
    cm: { to: x => x * 100 * C, from: nu => nu / (100 * C), jac: () => 100 * C, label: 'Wavenumber (cm⁻¹)' },
    eV: { to: x => x * EV / H, from: nu => nu * H / EV, jac: () => EV / H, label: 'Photon energy (eV)' },
    THz: { to: x => x * 1e12, from: nu => nu / 1e12, jac: () => 1e12, label: 'Frequency (THz)' },
  };

  // Sum Σ_k a_k · w(ν − ν_k)/(σ_k √2π) on an ascending ν grid. Lines resolved by the grid (FWHM over
  // two bins) are sampled exactly; narrower ones put their exact area into the bins they cover, so
  // integrals (equivalent widths, total power) are right at any zoom. Returns {re, im} (im only if asked).
  function sumProfiles(nu, lines, wantIm) {
    const n = nu.length, re = new Float64Array(n), im = wantIm ? new Float64Array(n) : null;
    if (n < 2) return { re, im };
    const edge = new Float64Array(n + 1);
    edge[0] = nu[0] - 0.5 * (nu[1] - nu[0]); edge[n] = nu[n - 1] + 0.5 * (nu[n - 1] - nu[n - 2]);
    for (let i = 1; i < n; i++) edge[i] = 0.5 * (nu[i - 1] + nu[i]);
    const find = v => { let lo = 0, hi = n; while (lo < hi) { const m = (lo + hi) >> 1; if (nu[m] < v) lo = m + 1; else hi = m; } return lo; };
    // Each line is kept out to where its Lorentzian wing a γ/(πΔ²) falls below 1e-12 of the highest
    // plotted point, so the wings reach any plotted floor while weak lines stay cheap.
    // (The plotted height of a line narrower than a bin is its area over the bin width.)
    const binAt = v => { const k = Math.min(n - 1, find(v)); return edge[Math.min(n, k + 1)] - edge[k]; };
    let top = 0;
    for (const L of lines) if (L.a > 0 && L.nu >= edge[0] && L.nu <= edge[n]) top = Math.max(top, L.a * Math.min(voigt(0, L.s, L.g), 1 / binAt(L.nu)));
    if (!(top > 0)) for (const L of lines) if (L.a > 0) top = Math.max(top, L.a * voigt(0, L.s, L.g));
    const floor = 1e-12 * top;
    for (const L of lines) {
      if (!(L.a > 0) && !(wantIm && L.a)) continue;
      const p = pvParams(L.s, L.g), core = 3 * p.f;
      const cut = L.g > 0 && floor > 0 ? Math.sqrt(Math.abs(L.a) * L.g / (Math.PI * floor)) : 0, half = Math.min(Math.max(14 * L.s, core, cut), 0.5 * L.nu);
      if (!wantIm && (L.nu + half < nu[0] || L.nu - half > nu[n - 1])) continue;
      const i0 = wantIm ? 0 : Math.max(0, find(L.nu - half) - 1), i1 = wantIm ? n - 1 : Math.min(n - 1, find(L.nu + half));
      const k = Math.min(n - 1, find(L.nu)), bw = edge[Math.min(n, k + 1)] - edge[k];
      const deposit = !wantIm && p.f < 2 * bw;
      for (let i = i0; i <= i1; i++) {
        if (deposit && edge[i + 1] > L.nu - core && edge[i] < L.nu + core) {
          const w = edge[i + 1] - edge[i];
          if (w > 0) re[i] += L.a * (pvCdf(edge[i + 1] - L.nu, p) - pvCdf(edge[i] - L.nu, p)) / w;
        } else {
          const w = voigtW(nu[i] - L.nu, L.s, L.g); re[i] += L.a * w[0]; if (im) im[i] += L.a * w[1];
        }
      }
    }
    return { re, im };
  }

  // Equivalent width W_ν (Hz) of one line at column density Nℓ (cm⁻²): ∫(1 − e^{−τ}) dν.
  function equivalentWidth(column, f, sigma, gamma, nPts) {
    nPts = nPts || 4001; const K = LINE_K * f * column * 1e4;
    const half = 12 * sigma + 40 * gamma + 12 * Math.sqrt(Math.max(K * gamma / Math.PI, 0)), U = 8, sh = Math.sinh(U);
    let W = 0, dPrev = null, yPrev = 0;
    for (let k = 0; k < nPts; k++) { const u = -1 + 2 * k / (nPts - 1), d = half * Math.sinh(U * u) / sh, y = -Math.expm1(-K * voigt(d, sigma, gamma));
      if (dPrev !== null) W += 0.5 * (y + yPrev) * (d - dPrev); dPrev = d; yPrev = y; }
    return W + 2 * K * gamma / (Math.PI * half);   // the thin Lorentzian tails beyond ±half
  }

  // Saturated absorption of one isotope's components: Lamb dips and crossovers (see sas_pairs in Python).
  function sasPairs(comps, sigmaD) {
    const out = [];
    for (let i = 0; i < comps.length; i++) for (let j = i; j < comps.length; j++) {
      const a = comps[i], b = comps[j];
      if (i !== j && a.F !== b.F && a.Fp !== b.Fp) continue;
      let k = i === j ? a.w * a.S : 0.5 * (a.w * b.S + b.w * a.S);
      const half = 0.5 * (a.d - b.d) * 1e6; k *= Math.exp(-half * half / (2 * sigmaD * sigmaD));
      out.push({ d: 0.5 * (a.d + b.d), k, i, j });
    }
    return out;
  }

  // Wavelength (nm) → sRGB (Bruton's piecewise approximation, with fall-off at the eye's limits).
  function wlRgb(l) {
    let r = 0, g = 0, b = 0;
    if (l >= 380 && l < 440) { r = -(l - 440) / 60; b = 1; } else if (l < 490) { g = (l - 440) / 50; b = 1; }
    else if (l < 510) { g = 1; b = -(l - 510) / 20; } else if (l < 580) { r = (l - 510) / 70; g = 1; }
    else if (l < 645) { r = 1; g = -(l - 645) / 65; } else if (l <= 780) { r = 1; }
    const f = l < 380 || l > 780 ? 0 : l < 420 ? 0.3 + 0.7 * (l - 380) / 40 : l > 700 ? 0.3 + 0.7 * (780 - l) / 80 : 1;
    return [r, g, b].map(v => Math.round(255 * Math.pow(v * f, 0.8)));
  }

  return { C, H, KB, AMU, EV, RE, LINE_K, faddeeva, voigt, voigtW, erf, pvParams, pvCdf, dopplerSigma, instrumentSigma,
    naturalFwhm, stimulated, airIndex, vacToAir, airToVac, vapourDensityCm3, boltzmann, UNITS, sumProfiles,
    equivalentWidth, sasPairs, wlRgb };
})();
if (typeof module !== 'undefined') module.exports = SP;
