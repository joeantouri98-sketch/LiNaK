
<script>
(function() {
  const ALL_LINES  = $!spec_lines_js;
  const MASS_AMU   = $!mass_amu;
  const kB         = $!kb_eV;   // eV/K (Python KB_EV, single source)
  const SRC_COLORS = $!SRC_COLORS_JS;

  // ── Source checkboxes ───────────────────────────────────────────────────
  const sources = [...new Set(ALL_LINES.map(l => l.src))].sort();
  const srcEl   = document.getElementById('src-checks');
  sources.forEach(src => {
    const lbl = document.createElement('label');
    const chk = document.createElement('input');
    chk.type = 'checkbox'; chk.checked = true;
    chk.dataset.value = src; chk.className = 'src-chk';
    const sw = document.createElement('span');
    sw.className = 'color-swatch'; sw.style.background = SRC_COLORS[src] || '#888';
    const cnt = ALL_LINES.filter(l => l.src === src).length;
    const badge = document.createElement('span');
    badge.className = 'badge'; badge.textContent = cnt;
    lbl.appendChild(sw); lbl.appendChild(chk);
    lbl.appendChild(document.createTextNode(' ' + src));
    lbl.appendChild(badge); srcEl.appendChild(lbl);
  });

  function getPlotDiv() { return document.querySelector('.plotly-graph-div'); }

  // ── Safe numeric parsers (0 is valid, not falsy) ──────────────────────
  // BUG FIX 1: parseFloat(x) || default fails when user types 0
  function parseNumMin(id, def) {
    const v = document.getElementById(id).value.trim();
    return v === '' ? def : parseFloat(v);
  }
  function parseNumMax(id, def) {
    const v = document.getElementById(id).value.trim();
    return v === '' ? def : parseFloat(v);
  }

  // ── Doppler σ in nm (standard deviation, not FWHM) ───────────────────
  // PHYSICS FIX 1: docstring clarification — σ is used in exp(-0.5*(x/σ)²)
  // FWHM_Doppler = 2√(2ln2)·σ ≈ 2.355·σ
  function dopplerSigmaNm(wl_nm, T_K) {
    const mc2 = MASS_AMU * 931.494e6;   // eV
    const E_eV = 1239.84 / wl_nm;
    const sigmaE = E_eV * Math.sqrt(kB * T_K / mc2);   // σ_E = E sqrt(kT/mc²)
    // σ_nm = wl²/hc · σ_E (same as the Python doppler_sigma_nm)
    return wl_nm * wl_nm / 1239.84 * sigmaE;
  }

  // ── Peak-normalised lineshapes ──────────────────────────────────────────
  // PHYSICS FIX 2: Gaussians are already peak=1 at x=x0 — no change needed.
  // PHYSICS FIX 3: Lorentzian should be used for natural broadening; the JS
  //                computed spectrum uses Gaussian as a proxy. Fix it here.
  // PHYSICS FIX 5: All lineshapes normalised to peak=1 before scaling by amp
  //                so relative heights correctly reflect oscillator strengths.
  function gaussian(x, x0, sigma) {
    // Peak = 1 at x=x0
    return Math.exp(-0.5 * ((x - x0) / sigma) ** 2);
  }
  function lorentzianNorm(x, x0, gamma) {
    // Peak-normalised Lorentzian: value=1 at x=x0
    // Raw Lorentz peak = (gamma/2pi)/(gamma/2)^2 = 2/(pi*gamma)
    // Dividing out: (gamma/2pi)/((x-x0)^2+(gamma/2)^2) / (2/(pi*gamma))
    //             = 0.5*(gamma/pi)^2 / ((x-x0)^2+(gamma/2)^2)   -- simplify:
    //             = 1 / (1 + (2*(x-x0)/gamma)^2)
    const u = 2 * (x - x0) / gamma;
    return 1.0 / (1.0 + u * u);
  }

  // ── Wavelength grid ─────────────────────────────────────────────────────
  const WL_MIN = $!wl_lo;
  const WL_MAX = $!wl_hi;
  const N_PTS  = 8000;   // increased from 5000 for better Doppler resolution
  const wlGrid = Array.from({length: N_PTS},
    (_, i) => WL_MIN + (WL_MAX - WL_MIN) * i / (N_PTS - 1));

  // ── Compute spectrum ────────────────────────────────────────────────────
  // PHYSICS FIX 2+3+5+6: peak-normalised shapes, correct Lorentzian for natural,
  // amplitude = fosc * n_degenerate.
  function computeSpectrum(lines, broadening, T_K, sigmaInst_nm) {
    const arr = new Float64Array(N_PTS);
    lines.forEach(r => {
      const wl0  = r.wl;
      // PHYSICS FIX 6: weight by degeneracy-corrected oscillator strength
      const nDeg = r.n_deg || 1;
      const amp  = r.fosc * nDeg;

      if (broadening === 'doppler') {
        const sig = Math.max(dopplerSigmaNm(wl0, T_K), 1e-6);
        for (let i = 0; i < N_PTS; i++)
          arr[i] += amp * gaussian(wlGrid[i], wl0, sig);

      } else if (broadening === 'instrumental') {
        const sig = sigmaInst_nm;
        for (let i = 0; i < N_PTS; i++)
          arr[i] += amp * gaussian(wlGrid[i], wl0, sig);

      } else {
        // natural — PHYSICS FIX 3: use peak-normalised Lorentzian
        // gamma from lifetimes.py pre-computed value stored in r.gamma_nm,
        // or fall back to a 0.01pm proxy (effectively a delta function)
        const gamma = (r.gamma_nm && r.gamma_nm > 0) ? r.gamma_nm : 1e-5;
        for (let i = 0; i < N_PTS; i++)
          arr[i] += amp * lorentzianNorm(wlGrid[i], wl0, gamma);
      }
    });
    let mx = 0;
    for (let i = 0; i < N_PTS; i++) if (arr[i] > mx) mx = arr[i];
    if (mx > 0) for (let i = 0; i < N_PTS; i++) arr[i] /= mx;
    return arr;
  }

  // ── Pagination for line table ────────────────────────────────────────────
  // BUG FIX 4: full pagination instead of truncating at 300
  const PAGE_SIZE = 150;
  let _allSpecLines = [];
  let _specPage     = 0;

  function specTablePage(delta) {
    _specPage = Math.max(0,
      Math.min(_specPage + delta, Math.ceil(_allSpecLines.length / PAGE_SIZE) - 1));
    renderSpecTable();
  }
  window.specTablePage = specTablePage;

  function renderSpecTable() {
    const tbody  = document.getElementById('line-tbody');
    const pager  = document.getElementById('spec-pager');
    const pgInfo = document.getElementById('spec-page-info');
    if (!tbody) return;
    const total  = _allSpecLines.length;
    const nPages = Math.ceil(total / PAGE_SIZE) || 1;
    _specPage    = Math.max(0, Math.min(_specPage, nPages - 1));
    const rows   = _allSpecLines.slice(_specPage * PAGE_SIZE, (_specPage + 1) * PAGE_SIZE);

    tbody.innerHTML = '';
    rows.forEach(r => {
      const tr  = document.createElement('tr');
      const col = SRC_COLORS[r.src] || '#888';
      // BUG FIX 4 IMPROVE 3+4: show lower label; adaptive dE units
      const dE_meV = r.dE;
      const dEstr  = dE_meV >= 0.01
        ? dE_meV.toFixed(2) + ' meV'
        : (dE_meV * 1000).toFixed(2) + ' μeV';

      // Build cells as DOM nodes (no innerHTML += pattern)
      const cells = [
        [r.ulp || r.ul,          `color:${col};font-weight:bold`],
        [r.llp || r.ll || '–', ''],
        [r.wl.toFixed(3), ''],
        [dEstr,          ''],
        [r.fosc.toFixed(4), ''],
        [r.src,          `color:${col}`],
      ];
      cells.forEach(([text, style]) => {
        const td = document.createElement('td');
        td.textContent = text;
        if (style) td.style.cssText = style;
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });

    if (total > PAGE_SIZE) {
      pager.style.display = 'flex';
      pgInfo.textContent  = `Page ${_specPage+1}/${nPages} (${total} total)`;
    } else {
      pager.style.display = 'none';
    }
  }

  // Apply filter 
  function applySpecFilter() {
    const gd = getPlotDiv();
    if (!gd || !gd.data) { alert('Plot not ready.'); return; }

    const selSrc    = new Set([...document.querySelectorAll('.src-chk:checked')]
                               .map(c => c.dataset.value));
    const broadening = document.querySelector('input[name="broad"]:checked').value;
    const T_K        = parseFloat(document.getElementById('temp-slider').value) || 300;
    const sigmaInst  = parseFloat(document.getElementById('inst-sigma').value) || 0.3;
    // BUG FIX 1: safe parsers
    const wlMin      = parseNumMin('wl-min-sp', -Infinity);
    const wlMax      = parseNumMax('wl-max-sp',  Infinity);
    const foscMin    = parseNumMin('fosc-min', 0);
    const lblSearch  = document.getElementById('lbl-search-sp').value.trim().toLowerCase();

    const matched = ALL_LINES.filter(l => {
      if (!selSrc.has(l.src)) return false;
      if (l.wl < wlMin || l.wl > wlMax) return false;
      if (l.fosc < foscMin) return false;
      if (lblSearch && !l.ul.toLowerCase().includes(lblSearch) &&
                       !(l.ll || '').toLowerCase().includes(lblSearch) &&
                       !(l.ulp || '').toLowerCase().includes(lblSearch) &&
                       !(l.llp || '').toLowerCase().includes(lblSearch)) return false;
      return true;
    });

    document.getElementById('spec-result').textContent =
      matched.length + ' / ' + ALL_LINES.length + ' lines match';

    // Compute new spectrum
    const newSpec = computeSpectrum(matched, broadening, T_K, sigmaInst);
    const xArr    = Array.from(wlGrid);
    const yArr    = Array.from(newSpec);

    // BUG FIX 2: always target or create a dedicated 'broad_filtered' trace.
    // Never overwrite a pre-rendered trace (broad_All_*) — those are fixed
    // at their build-time broadening. Only overwrite the user-filtered trace.
    let filteredIdx = -1;
    gd.data.forEach((tr, i) => {
      if ((tr.legendgroup || '') === 'broad_filtered') filteredIdx = i;
    });

    const broadeningLabel = {
      doppler:      `Doppler T=${T_K.toFixed(0)}K`,
      instrumental: `Instrumental σ=${sigmaInst}nm`,
      natural:      'Natural (Lorentzian)',
    }[broadening] || broadening;

    if (filteredIdx >= 0) {
      Plotly.restyle(gd, {
        visible: true,
        x: [xArr], y: [yArr],
        name: ['Filtered — ' + broadeningLabel],
      }, [filteredIdx]);
    } else {
      Plotly.addTraces(gd, {
        x: xArr, y: yArr, mode: 'lines',
        fill: 'tozeroy', line: {color:'#8e44ad', width:2},
        fillcolor: 'rgba(142,68,173,0.15)',
        name: 'Filtered — ' + broadeningLabel,
        hovertemplate: 'λ = %{x:.3f} nm<br>I = %{y:.4f}<extra></extra>',
        legendgroup: 'broad_filtered',
      });
    }

    _allSpecLines = matched;
    _specPage     = 0;
    renderSpecTable();
  }

  // Reset 
  function resetSpecFilter() {
    document.querySelectorAll('#spec-filter input[type=checkbox]')
            .forEach(c => c.checked = true);
    document.querySelector('input[name="broad"][value="instrumental"]').checked = true;
    document.getElementById('temp-slider').value = 300;
    document.getElementById('temp-val').textContent = '300K';
    document.getElementById('inst-sigma').value = 0.3;
    ['wl-min-sp','wl-max-sp','fosc-min','lbl-search-sp']
      .forEach(id => { const el = document.getElementById(id); if(el) el.value=''; });
    document.getElementById('spec-result').textContent = '';

    // BUG FIX 3: delete the filtered trace if it exists, then restore originals
    const gd = getPlotDiv();
    if (gd && gd.data) {
      const filteredIdx = gd.data.findIndex(tr =>
        (tr.legendgroup || '') === 'broad_filtered');
      if (filteredIdx >= 0) {
        Plotly.deleteTraces(gd, filteredIdx).then(() => {
          const vis = gd.data.map(tr =>
            (tr.legendgroup || '') === 'broad_All_instrumental' ? true : 'legendonly');
          Plotly.restyle(gd, { visible: vis }, gd.data.map((_,i) => i));
        });
      } else {
        const vis = gd.data.map(tr =>
          (tr.legendgroup || '') === 'broad_All_instrumental' ? true : 'legendonly');
        Plotly.restyle(gd, { visible: vis }, gd.data.map((_,i) => i));
      }
    }

    _allSpecLines = []; _specPage = 0;
    renderSpecTable();
  }

  window.applySpecFilter  = applySpecFilter;
  window.resetSpecFilter  = resetSpecFilter;
  renderSpecTable();
})();
</script>
