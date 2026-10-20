
<script>
(function() {
  const ALL_TRANS     = $!ALL_TRANS_JS;
  const EXTRA_SOURCES = $!EXTRA_SOURCES_JS;
  const EXTRA_SOURCE_NAMES = new Set(EXTRA_SOURCES.map(e => e.name));
  const REGION_COLORS = $!REGION_COLORS_JS;
  const GROT_X_LO     = $!GROT_X_LO;
  const GROT_X_HI     = $!GROT_X_HI;
  const GROT_X_INNER_L = $!GROT_X_INNER_L;
  const GROT_X_INNER_R = $!GROT_X_INNER_R;
  const GROT_EMIS_MAX = $!GROT_EMIS_MAX;
  const GROT_INITIAL  = $!grot_initial_count;

  function fnvHash(s) {
    let h = 2166136261;
    for (let i = 0; i < s.length; i++) {
      h ^= s.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    return h >>> 0;
  }

  function mulberry32(seed) {
    let s = seed >>> 0;
    return () => {
      s = (s + 0x6D2B79F5) >>> 0;
      let t = Math.imul(s ^ (s >>> 15), 1 | s);
      t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function grotHoverHtml(t, direction) {
    const ulShow = t.ulp || t.ul;
    const llShow = t.llp || t.ll;
    const lbl = direction === 'absorption'
      ? `${llShow} → ${ulShow}` : `${ulShow} → ${llShow}`;
    const tag = direction === 'absorption' ? 'abs' : 'emis';
    const foscStr = (t.fos != null && t.fos > 0)
      ? `<br>fosc = ${t.fos.toFixed(5)}` : '<br>fosc = —';
    const dEeV = (t.dE / 1000).toFixed(4);
    return (`<b>${lbl}</b> (${tag})<br>λ = ${t.wl.toFixed(1)} nm`
      + `<br>ΔE = ${dEeV} eV${foscStr}`
      + `<br>Region: ${t.reg}<br>[${t.us}/${t.ls}]<extra></extra>`);
  }

  // Spread arrows horizontally by shared lower level (matches Python layout).
  function assignGrotrianX(transList) {
    const buckets = {};
    transList.forEach(t => {
      const ylKey = (t.yL != null ? Number(t.yL).toFixed(4) : '0');
      const key = t.ll + '\x00' + ylKey;
      if (!buckets[key]) buckets[key] = { ll: t.ll, items: [] };
      buckets[key].items.push(t);
    });
    const xMap = new Map();
    const emisMap = new Map();
    Object.values(buckets).forEach(bucket => {
      const items = bucket.items;
      items.sort((a, b) =>
        (a.yU - b.yU) || (a.wl - b.wl) || a.ul.localeCompare(b.ul));
      const n = items.length;
      const emisOff = Math.min(GROT_EMIS_MAX, 0.010 + 0.0008 * Math.min(n, 12));
      const absLo = GROT_X_LO + GROT_X_INNER_L;
      const absHi = GROT_X_HI - GROT_X_INNER_R - emisOff;
      const span = Math.max(absHi - absLo, 1e-6);
      const softGap = span / Math.max(n * 1.6, 5.0);
      const placed = [];
      items.forEach(t => {
        const sk = t.ul + '\x00' + t.ll;
        const rng = mulberry32(fnvHash(t.ul + '\x00' + t.ll + '\x00' + bucket.ll));
        let x = null;
        for (let attempt = 0; attempt < 10; attempt++) {
          const cand = absLo + rng() * span;
          if (attempt >= 7) { x = cand; break; }
          if (placed.every(p => Math.abs(cand - p) >= softGap * 0.35)) {
            x = cand; break;
          }
        }
        if (x == null) x = absLo + rng() * span;
        x = Math.max(absLo, Math.min(absHi, x));
        placed.push(x);
        xMap.set(sk, x);
        emisMap.set(sk, emisOff);
      });
    });
    return { xMap, emisMap };
  }

  // Re-position pre-rendered arrows to match the current matched set layout.
  function relayoutArrowPositions(gd, matched, direction) {
    if (!matched.length || !TRACE_META) return Promise.resolve();
    const { xMap, emisMap } = assignGrotrianX(matched);
    const byKey = new Map(matched.map(t => [t.ul + '\x00' + t.ll, t]));
    const idxs = [], xs = [], ys = [];
    Object.entries(TRACE_META).forEach(([iStr, meta]) => {
      if (!meta.ul || !meta.ll) return;
      if (direction !== 'both' && direction !== meta.direction) return;
      const sk = meta.ul + '\x00' + meta.ll;
      if (!xMap.has(sk)) return;
      const t = byKey.get(sk);
      if (!t || t.yL === undefined || t.yU === undefined) return;
      const xA = xMap.get(sk);
      const dx = emisMap.get(sk) ?? 0.014;
      idxs.push(parseInt(iStr));
      if (meta.direction === 'absorption') {
        xs.push([xA, xA]); ys.push([t.yL, t.yU]);
      } else {
        xs.push([xA + dx, xA + dx]); ys.push([t.yU, t.yL]);
      }
    });
    if (!idxs.length) return Promise.resolve();
    return Plotly.restyle(gd, { x: xs, y: ys }, idxs);
  }

  function getPlotDiv() { return document.querySelector('.plotly-graph-div'); }
  function getChecked(cls) {
    return new Set([...document.querySelectorAll('.' + cls + ':checked')]
                   .map(c => c.dataset.value));
  }

  // ── Safe numeric parsers (0 is valid, not falsy-ignored) ────────────────
  function parseMin(id, def) {
    const v = document.getElementById(id).value.trim();
    return v === '' ? def : parseFloat(v);
  }
  function parseMax(id, def) {
    const v = document.getElementById(id).value.trim();
    return v === '' ? def : parseFloat(v);
  }
  function parseIntOpt(id, def) {
    const v = document.getElementById(id).value.trim();
    return v === '' ? def : parseInt(v);
  }

  // ── Build filter checkboxes ──────────────────────────────────────────────
  const regions = [...new Set(ALL_TRANS.map(t => t.reg))]
    .sort((a,b) => {
      const o = $!region_order_js;
      return o.indexOf(a) - o.indexOf(b);
    });
  const ltypes  = [...new Set(ALL_TRANS.map(t => t.ll_l + '→' + t.ul_l))].sort();
  // Individual endpoint sources (NIST, QDT model) — not us/ls compound pairs
  const sources = [...new Set(ALL_TRANS.flatMap(t => [t.us, t.ls].filter(Boolean)))].sort();
  const dns     = [...new Set(ALL_TRANS.map(t => t.dn))].sort((a,b)=>a-b);

  function makeCheckboxes(containerId, items, colorFn, keyFn) {
    const el = document.getElementById(containerId);
    if (!el) return;
    items.forEach(item => {
      const lbl = document.createElement('label');
      const chk = document.createElement('input');
      chk.type = 'checkbox'; chk.checked = true;
      chk.dataset.value = item;
      chk.className = containerId + '-chk';
      if (colorFn) {
        const dot = document.createElement('span');
        dot.className = 'color-dot';
        dot.style.background = colorFn(item);
        lbl.appendChild(dot);
      }
      lbl.appendChild(chk);
      lbl.appendChild(document.createTextNode(' ' + item));
      const cnt = ALL_TRANS.filter(t => (keyFn || (x => x.reg))(t) === item).length;
      const badge = document.createElement('span');
      badge.className = 'count-badge'; badge.textContent = cnt;
      lbl.appendChild(badge); el.appendChild(lbl);
    });
  }

  makeCheckboxes('region-checks',  regions,  r => REGION_COLORS[r] || '#999', t => t.reg);
  makeCheckboxes('ltype-checks',   ltypes,   null, t => t.ll_l + '→' + t.ul_l);
  // Per-source checkboxes: NIST/QDT from transitions + --extra overlays
  (function() {
    const el = document.getElementById('source-checks');
    if (!el) return;
    sources.forEach(src => {
      const lbl = document.createElement('label');
      const chk = document.createElement('input');
      chk.type = 'checkbox'; chk.checked = true;
      chk.dataset.value = src;
      chk.className = 'source-checks-chk';
      lbl.appendChild(chk);
      lbl.appendChild(document.createTextNode(' ' + src));
      const cnt = ALL_TRANS.filter(t => t.us === src || t.ls === src).length;
      const badge = document.createElement('span');
      badge.className = 'count-badge'; badge.textContent = cnt;
      lbl.appendChild(badge); el.appendChild(lbl);
    });
    EXTRA_SOURCES.forEach(ex => {
      const lbl = document.createElement('label');
      const chk = document.createElement('input');
      chk.type = 'checkbox'; chk.checked = false;  // default hidden
      chk.dataset.value = ex.name;
      chk.dataset.extra = '1';
      chk.className = 'source-checks-chk';
      const dot = document.createElement('span');
      dot.className = 'color-dot';
      dot.style.background = ex.color || '#888';
      lbl.appendChild(dot);
      lbl.appendChild(chk);
      lbl.appendChild(document.createTextNode(' ' + ex.name));
      const badge = document.createElement('span');
      badge.className = 'count-badge'; badge.textContent = String(ex.count);
      badge.title = 'Extra calculation overlay (--extra)';
      lbl.appendChild(badge); el.appendChild(lbl);
    });
  })();

  // Show/hide --extra overlay traces (levels + arrows) by legendgroup
  function applyExtraOverlayVisibility(gd, selSources) {
    if (!gd || !gd.data || !EXTRA_SOURCES.length) return;
    const idxs = [], viss = [];
    gd.data.forEach((tr, i) => {
      const lg = tr.legendgroup || '';
      for (let k = 0; k < EXTRA_SOURCES.length; k++) {
        const ex = EXTRA_SOURCES[k];
        const isLvl = (lg === ex.legendgroup);
        const isArr = lg.startsWith(ex.slug + '_');
        if (!isLvl && !isArr) continue;
        idxs.push(i);
        viss.push(selSources.has(ex.name) ? true : 'legendonly');
        break;
      }
    });
    if (idxs.length) Plotly.restyle(gd, { visible: viss }, idxs);
  }

  const dnEl = document.getElementById('dn-checks');
  if (dnEl) dns.forEach(dn => {
    const lbl = document.createElement('label');
    lbl.style.cssText = 'margin:1px 3px 1px 0;cursor:pointer;font-size:11px;';
    const chk = document.createElement('input');
    chk.type = 'checkbox'; chk.checked = true;
    chk.dataset.value = String(dn); chk.className = 'dn-checks-chk';
    lbl.appendChild(chk);
    lbl.appendChild(document.createTextNode(' ' + dn));
    dnEl.appendChild(lbl);
  });

  // ── Trace metadata ───────────────────────────────────────────────────────
  let TRACE_META = null;   // idx -> {region,direction,ul,ll,origVis}
  let LEVEL_META = null;   // idx -> {label,isGs,legendgroup,origVis}
  let _applyActive = false;  // true after Apply — Reset clears it
  // Per region_direction legend toggle: undefined = never touched, true/false = user set
  const _legendGroupState = {};

  // Track Rydberg (and any level) live visibility ourselves —
  // Plotly doesn't update data[i].visible on legend click,
  // so we listen for plotly_legendclick and maintain a shadow map.
  // _liveVis[i] = true | false | 'legendonly'
  let _liveVis = {};

  function buildTraceMeta() {
    const gd = getPlotDiv();
    if (!gd || !gd.data) return false;
    TRACE_META = {}; LEVEL_META = {};
    gd.data.forEach((trace, i) => {
      const lg  = trace.legendgroup || '';
      const vis = (trace.visible === undefined) ? true : trace.visible;
      // Only initialise _liveVis for entries not yet tracked
      // (preserve values set by legend-click listener across rebuilds)
      if (!(_liveVis[i] !== undefined)) _liveVis[i] = vis;

      if (lg.startsWith('transition_')) {
        const lgBody    = lg.slice('transition_'.length);
        const lastUs    = lgBody.lastIndexOf('_');
        const region    = lgBody.slice(0, lastUs);
        const direction = lgBody.slice(lastUs + 1);
        const cd = trace.customdata;
        const ul = cd && cd[0] ? cd[0][0] : '';
        const ll = cd && cd[0] ? cd[0][1] : '';
        TRACE_META[i] = { region, direction, ul, ll, origVis: vis };
      }
      // Level lines: ORCA TD-DFT (main), Rydberg NIST/QDT/theo (rydberg_*),
      // ORCA SOC overlay (orca_soc), and --extra overlays (extra_*).
      const isLevel = (lg === 'main'
                    || lg === 'rydberg' || lg.startsWith('rydberg_')
                    || lg === 'orca_soc' || lg.startsWith('orca_soc')
                    || lg.startsWith('extra_'));
      if (isLevel) {
        const cd    = trace.customdata;
        const label = (cd && cd[0] && cd[0][0] !== undefined) ? String(cd[0][0]) : null;
        const isGs  = (cd && cd[0] && cd[0][1] === true);
        if (label !== null)
          LEVEL_META[i] = { label, isGs, legendgroup: lg, origVis: vis };
      }
    });
    return true;
  }

  // ── Pagination ───────────────────────────────────────────────────────────
  const PAGE_SIZE = 100;
  let _allMatched  = [];
  let _viewMatched = [];     // _allMatched narrowed by the search box (what the table pages through)
  let _matchQuery  = '';
  let _currentPage = 0;

  const _REG_SHORT = {'X-ray':'X','Vacuum UV':'VUV','UV':'UV','Visible':'Vis',
                      'Near-IR':'NIR','Mid/Far-IR':'MIR','Microwave':'μW'};

  // Every whitespace-separated term must appear in the row text (upper, lower,
  // wavelength, energy gap, region name or its short tag), case-insensitive.
  function matchHaystack(t) {
    return (t.ul + ' ' + t.ll + ' ' + t.wl.toFixed(1) + ' ' + t.wl.toFixed(4) + ' '
            + t.dE.toFixed(1) + ' ' + t.reg + ' ' + (_REG_SHORT[t.reg] || '')).toLowerCase();
  }

  function refreshViewMatched() {
    const terms = _matchQuery.toLowerCase().split(/\s+/).filter(Boolean);
    _viewMatched = terms.length === 0
      ? _allMatched
      : _allMatched.filter(t => {
          const h = matchHaystack(t);
          return terms.every(w => h.indexOf(w) !== -1);
        });
  }

  function tablePage(delta) {
    _currentPage = Math.max(0,
      Math.min(_currentPage + delta, Math.ceil(_viewMatched.length / PAGE_SIZE) - 1));
    renderTablePage();
  }
  window.tablePage = tablePage;

  // ── Multi-selection ──────────────────────────────────────────────────────
  let _selectedKeys = new Set();
  let _lastClickIdx = null;

  function selKey(t) { return t.ul + '\x00' + t.ll; }

  function updateSelCount() {
    const el = document.getElementById('sel-count');
    if (!el) return;
    el.textContent = _selectedKeys.size > 0
      ? `${_selectedKeys.size} transition${_selectedKeys.size > 1 ? 's' : ''} selected`
      : '';
  }

  // ── Table rendering ──────────────────────────────────────────────────────
  // BUG FIX: build ALL cells as DOM nodes — never use innerHTML +=
  // after appending a node with an event listener, as it destroys the node.
  function renderTablePage() {
    const tbody  = document.getElementById('trans-tbody');
    const pager  = document.getElementById('table-pager');
    const pgInfo = document.getElementById('page-info');
    if (!tbody) return;

    refreshViewMatched();
    const total  = _viewMatched.length;
    const nPages = Math.ceil(total / PAGE_SIZE) || 1;
    _currentPage = Math.max(0, Math.min(_currentPage, nPages - 1));
    const start  = _currentPage * PAGE_SIZE;
    const rows   = _viewMatched.slice(start, start + PAGE_SIZE);
    const qCount = document.getElementById('match-search-count');
    if (qCount) qCount.textContent = _matchQuery.trim()
      ? total + ' of ' + _allMatched.length : '';

    const SHORT = {'X-ray':'X','Vacuum UV':'VUV','UV':'UV','Visible':'Vis',
                    'Near-IR':'NIR','Mid/Far-IR':'MIR','Microwave':'μW'};
    tbody.innerHTML = '';

    rows.forEach((t, pageIdx) => {
      const globalIdx = start + pageIdx;
      const key   = selKey(t);
      const isSel = _selectedKeys.has(key);
      const col   = REGION_COLORS[t.reg] || '#999';
      const wlFmt = t.wl >= 1e6 ? (t.wl/1e6).toFixed(2)+'M'
                  : t.wl >= 1e3 ? (t.wl/1e3).toFixed(1)+'k'
                  : t.wl.toFixed(1);

      const tr = document.createElement('tr');
      tr.style.cursor     = 'pointer';
      tr.style.background = isSel ? '#c8e0ff' : '';
      tr.title = 'Click / checkbox to select · Shift+click range · Dbl-click isolate';

      // ── Checkbox cell — built as DOM node, never overwritten ──
      const chkTd = document.createElement('td');
      const chk   = document.createElement('input');
      chk.type    = 'checkbox';
      chk.checked = isSel;
      chk.style.cssText = 'cursor:pointer;accent-color:#4a7ec7;';

      // Checkbox change: toggle selection, update row highlight, recount
      chk.addEventListener('change', e => {
        e.stopPropagation();
        if (chk.checked) _selectedKeys.add(key);
        else             _selectedKeys.delete(key);
        tr.style.background = chk.checked ? '#c8e0ff' : '';
        _lastClickIdx = globalIdx;
        updateSelCount();
      });
      chkTd.appendChild(chk);
      tr.appendChild(chkTd);

      // ── Data cells — built as DOM nodes too ──
      const cells = [
        [t.ul,  `color:${col};font-weight:bold`],
        [t.ll,  ''],
        [wlFmt, ''],
        [t.dE.toFixed(1), ''],
        [SHORT[t.reg] || t.reg, `color:${col}`],
      ];
      cells.forEach(([text, style]) => {
        const td = document.createElement('td');
        td.textContent = text;
        if (style) td.style.cssText = style;
        tr.appendChild(td);
      });

      // ── Row click: toggle / shift-range ──
      tr.addEventListener('click', e => {
        if (e.target === chk) return;   // checkbox handles its own click
        if (e.shiftKey && _lastClickIdx !== null) {
          const lo = Math.min(_lastClickIdx, globalIdx);
          const hi = Math.max(_lastClickIdx, globalIdx);
          for (let i = lo; i <= hi; i++) {
            if (_viewMatched[i]) _selectedKeys.add(selKey(_viewMatched[i]));
          }
          _lastClickIdx = globalIdx;
          renderTablePage();
        } else {
          if (_selectedKeys.has(key)) {
            _selectedKeys.delete(key);
            chk.checked = false;
            tr.style.background = '';
          } else {
            _selectedKeys.add(key);
            chk.checked = true;
            tr.style.background = '#c8e0ff';
          }
          _lastClickIdx = globalIdx;
        }
        updateSelCount();
      });

      // ── Double-click: isolate immediately ──
      tr.addEventListener('dblclick', e => {
        e.preventDefault();
        isolateTransitions([t]);
      });

      tbody.appendChild(tr);
    });

    if (total > PAGE_SIZE) {
      pager.style.display = 'flex';
      pgInfo.textContent  = `Page ${_currentPage+1}/${nPages} (${total} total)`;
    } else {
      pager.style.display = 'none';
    }
    updateSelCount();
  }

  // ── Search box above the table ───────────────────────────────────────────
  (function() {
    const box = document.getElementById('match-search');
    if (!box) return;
    box.addEventListener('input', () => {
      _matchQuery = box.value; _currentPage = 0; renderTablePage();
    });
    box.addEventListener('keydown', e => {
      if (e.key === 'Escape') { box.value = ''; _matchQuery = ''; _currentPage = 0; renderTablePage(); }
      e.stopPropagation();   // typing must not trigger plot keyboard shortcuts
    });
  })();

  // ── Isolate selected (button) ────────────────────────────────────────────
  function isolateSelected() {
    if (_selectedKeys.size === 0) {
      alert('Select at least one transition first.');
      return;
    }
    const targets = _allMatched.filter(t => _selectedKeys.has(selKey(t)));
    isolateTransitions(targets);
  }
  window.isolateSelected = isolateSelected;

  function clearSelection() {
    _selectedKeys.clear(); _lastClickIdx = null;
    renderTablePage(); updateSelCount();
  }
  window.clearSelection = clearSelection;

  // ── Injected trace registry ──────────────────────────────────────────────
  let _isolatedTraces = [];   // indices of Plotly.addTraces injected traces

  function cleanupInjected(gd) {
    if (_isolatedTraces.length === 0) return Promise.resolve();
    const sorted = [..._isolatedTraces].sort((a,b) => b-a);
    _isolatedTraces = [];
    return Plotly.deleteTraces(gd, sorted).then(() => buildTraceMeta());
  }

  // Inject arrows for transitions not in the initial max_show sample.
  // Uses precomputed yU/yL from Python — labelToY fails for ground state
  // (stored as energy 0, not "3s") and for levels not yet enabled in legend.
  function injectTransitionTraces(gd, transList, direction, layoutList) {
    if (!transList || transList.length === 0) return Promise.resolve();
    const forLayout = (layoutList && layoutList.length) ? layoutList : transList;
    const { xMap, emisMap } = assignGrotrianX(forLayout);
    const newTraces = [];
    transList.forEach((t) => {
      let yU = t.yU, yL = t.yL;
      if (yU === undefined || yL === undefined) {
        const labelToY = {};
        Object.keys(LEVEL_META || {}).forEach(iStr => {
          const i  = parseInt(iStr);
          const td = gd.data[i];
          if (td && td.y && td.y[0] !== undefined)
            labelToY[LEVEL_META[i].label] = td.y[0];
        });
        yU = yU ?? labelToY[t.ul];
        yL = yL ?? labelToY[t.ll];
      }
      if (yU === undefined || yL === undefined) return;
      const sk = t.ul + '\x00' + t.ll;
      const xPos = xMap.get(sk) ?? (GROT_X_LO + GROT_X_HI) / 2;
      const emisDx = emisMap.get(sk) ?? 0.014;
      const col  = REGION_COLORS[t.reg] || '#f90';
      const showAbs  = direction === 'both' || direction === 'absorption';
      const showEmis = direction === 'both' || direction === 'emission';
      if (showAbs) newTraces.push({
        x: [xPos, xPos], y: [yL, yU], mode: 'lines+markers',
        line: { color: col, width: 3, dash: 'solid' },
        marker: { symbol: ['line-ew','arrow-up'], size: [0,12], color: col },
        name: `${t.ulp || t.ul}→${t.llp || t.ll}`, showlegend: false,
        legendgroup: `transition_${t.reg}_absorption`,
        customdata: [[t.ul, t.ll]],
        hovertemplate: grotHoverHtml(t, 'absorption'),
      });
      if (showEmis) newTraces.push({
        x: [xPos + emisDx, xPos + emisDx], y: [yU, yL], mode: 'lines+markers',
        line: { color: col, width: 2, dash: 'dot' },
        marker: { symbol: ['line-ew','arrow-down'], size: [0,9], color: col },
        name: '', showlegend: false,
        legendgroup: `transition_${t.reg}_emission`,
        customdata: [[t.ul, t.ll]],
        hovertemplate: grotHoverHtml(t, 'emission'),
      });
    });
    if (newTraces.length === 0) return Promise.resolve();
    return Plotly.addTraces(gd, newTraces).then(() => {
      const n = gd.data.length;
      for (let i = n - newTraces.length; i < n; i++) _isolatedTraces.push(i);
      buildTraceMeta();
    });
  }

  function existingTransitionKeys() {
    const keys = new Set();
    Object.values(TRACE_META || {}).forEach(m => {
      if (m.ul && m.ll) keys.add(m.ul + '\x00' + m.ll);
    });
    return keys;
  }

  // True when user turned a legend group off (all traces false, not legendonly).
  function transitionGroupLegendOn(rdKey) {
    return _legendGroupState[rdKey] === true;
  }

  function anyLegendGroupTouched() {
    return Object.keys(_legendGroupState).length > 0;
  }

  function missingFromPlot(matched) {
    const have = existingTransitionKeys();
    return matched.filter(t => !have.has(selKey(t)));
  }

  // ── Core isolation ───────────────────────────────────────────────────────
  // BUG FIX 1: direction radio is respected when showing existing traces.
  // BUG FIX 2: level lines for endpoints are forced visible (true) not
  //            origVis — so Rydberg levels show even if they started legendonly.
  // BUG FIX 3: addTraces is awaited before recording injected indices.
  function isolateTransitions(targets) {
    const gd = getPlotDiv();
    if (!gd || !gd.data) return;
    if (!TRACE_META) buildTraceMeta();

    const direction   = document.querySelector('input[name="direction"]:checked').value;
    const targetKeys  = new Set(targets.map(t => selKey(t)));

    cleanupInjected(gd).then(() => {
      // ── Show only matching transition traces ──────────────────────────
      const allTrIdx  = Object.keys(TRACE_META).map(Number);
      const matchedTr = allTrIdx.filter(i => {
        const m = TRACE_META[i];
        if (!m.ul || !targetKeys.has(m.ul + '\x00' + m.ll)) return false;
        // Respect the direction radio
        if (direction !== 'both' && direction !== m.direction) return false;
        return true;
      });
      const visArr = allTrIdx.map(i => matchedTr.includes(i));
      if (allTrIdx.length > 0) Plotly.restyle(gd, { visible: visArr }, allTrIdx);

      // ── Show endpoint levels — force true so Rydberg shows ───────────
      // We force true (not origVis) because origVis for Rydberg is
      // 'legendonly' by default. The user chose to isolate this transition
      // so we must make the levels visible regardless of prior state.
      const endpointLabels = new Set();
      targets.forEach(t => {
        endpointLabels.add(String(t.ul));
        endpointLabels.add(String(t.ll));
      });
      const lvlIdx = Object.keys(LEVEL_META).map(Number);
      const lvlVis = lvlIdx.map(i => {
        if (LEVEL_META[i].isGs) return true;
        return endpointLabels.has(LEVEL_META[i].label) ? true : false;
      });
      if (lvlIdx.length > 0) Plotly.restyle(gd, { visible: lvlVis }, lvlIdx);

      // Inject transitions missing from the max_show sample (e.g. 3p3/2 D-line)
      const missing = missingFromPlot(targets);
      relayoutArrowPositions(gd, targets, direction).then(() =>
        injectTransitionTraces(gd, missing, direction, targets)
      ).then(() => {
        const desc = targets.length === 1
          ? `${targets[0].ul} → ${targets[0].ll} (${targets[0].wl.toFixed(1)} nm)`
          : `${targets.length} transitions`;
        document.getElementById('result-count').textContent =
          `Isolated: ${desc}  — click ↺ Reset to restore`;
      });
    });
  }

  // ── Core filter ──────────────────────────────────────────────────────────
  function applyFilter() {
    const gd = getPlotDiv();
    if (!gd || !gd.data) { alert('Plot not ready yet.'); return; }
    if (!TRACE_META) { if (!buildTraceMeta()) return; }

    cleanupInjected(gd).then(() => {
      _applyActive = true;
      const selRegions = getChecked('region-checks-chk');
      const selLtypes  = getChecked('ltype-checks-chk');
      const selSources = getChecked('source-checks-chk');
      const selDns = new Set([...document.querySelectorAll('.dn-checks-chk:checked')]
                             .map(c => parseInt(c.dataset.value)));
      const wlMin = parseMin('wl-min', -Infinity);
      const wlMax = parseMax('wl-max',  Infinity);
      const deMin = parseMin('de-min', -Infinity);
      const deMax = parseMax('de-max',  Infinity);
      const unMin = parseIntOpt('un-min', 0);
      const unMax = parseIntOpt('un-max', 9999);
      const labelSearch = document.getElementById('label-search').value.trim().toLowerCase();
      const direction   = document.querySelector('input[name="direction"]:checked').value;

      _allMatched = ALL_TRANS.filter(t => {
        if (!selRegions.has(t.reg)) return false;
        if (!selLtypes.has(t.ll_l + '→' + t.ul_l)) return false;
        // Both endpoint sources must be selected (deselecting QDT hides QDT/*)
        if (selSources.size === 0) return false;
        if (!selSources.has(t.us) || !selSources.has(t.ls)) return false;
        if (!selDns.has(t.dn)) return false;
        if (t.wl < wlMin || t.wl > wlMax) return false;
        if (t.dE < deMin || t.dE > deMax) return false;
        if (t.un < unMin || t.un > unMax) return false;
        if (labelSearch &&
            !t.ul.toLowerCase().includes(labelSearch) &&
            !t.ll.toLowerCase().includes(labelSearch)) return false;
        return true;
      });

      // Legend groups (plot) + filter panel (fine control) work together:
      // - panel region checkboxes gate which regions Apply may touch
      // - legend toggles persist: groups explicitly turned off stay hidden
      // - Apply shows pre-rendered traces that match filters (Isolate for the rest)
      const matchULLL = new Set(_allMatched.map(t => t.ul + '\x00' + t.ll));
      const matchedRD = new Set();
      _allMatched.forEach(t => {
        if (direction === 'both' || direction === 'absorption')
          matchedRD.add(t.reg + '_absorption');
        if (direction === 'both' || direction === 'emission')
          matchedRD.add(t.reg + '_emission');
      });

      const firstPerRD = {};
      Object.entries(TRACE_META).forEach(([iStr, meta]) => {
        if (!meta.ul) return;
        const k = meta.region + '_' + meta.direction;
        if (!(k in firstPerRD)) firstPerRD[k] = parseInt(iStr);
      });

      const pairs = [];
      Object.entries(TRACE_META).forEach(([iStr, meta]) => {
        const i = parseInt(iStr);
        if (!meta.ul) return;
        if (!selRegions.has(meta.region)) { pairs.push({idx:i,vis:false}); return; }
        if (direction !== 'both' && direction !== meta.direction) {
          pairs.push({idx:i,vis:false}); return;
        }
        const rdKey = meta.region + '_' + meta.direction;
        const legendOn = transitionGroupLegendOn(rdKey);
        const inMatch = matchULLL.has(meta.ul + '\x00' + meta.ll);
        const showLegend = (firstPerRD[rdKey] === i) && matchedRD.has(rdKey);
        let vis = false;
        if (inMatch) {
          if (legendOn) vis = true;
          else if (!anyLegendGroupTouched()) vis = true;
          // else: legend used on other groups — leave this group hidden
        } else if (showLegend) vis = 'legendonly';
        pairs.push({idx:i, vis: vis});
      });
      if (pairs.length > 0)
        Plotly.restyle(gd, { visible: pairs.map(p=>p.vis) }, pairs.map(p=>p.idx));

      relayoutArrowPositions(gd, _allMatched, direction);

      document.getElementById('result-count').textContent =
        _allMatched.length + ' / ' + ALL_TRANS.length + ' transitions match, '
        + GROT_INITIAL + ' pre-rendered';

      // ── Level lines (all sources) ─────────────────────────────────────
      // When hideNoArrows is on, hide every registered level that is not an
      // endpoint of a matched visible transition. Applies to ORCA (main),
      // Rydberg NIST/QDT/theo (rydberg_*), and SOC overlay (orca_soc).
      // Ground state is always kept.
      const hideNoArrows = document.getElementById('hide-no-arrows').checked;
      if (hideNoArrows && LEVEL_META && Object.keys(LEVEL_META).length > 0) {
        const visLabels = new Set();
        _allMatched.forEach(t => {
          if (t.ul) visLabels.add(String(t.ul));
          if (t.ll) visLabels.add(String(t.ll));
        });
        pairs.forEach(p => {
          if (p.vis !== true) return;
          const cd = gd.data[p.idx] && gd.data[p.idx].customdata;
          if (cd && cd[0]) {
            if (cd[0][0]) visLabels.add(String(cd[0][0]));
            if (cd[0][1]) visLabels.add(String(cd[0][1]));
          }
        });
        const lvlIdx = Object.keys(LEVEL_META).map(Number);
        const lvlVis = lvlIdx.map(i => {
          const lm = LEVEL_META[i];
          if (lm.isGs) return true;
          return visLabels.has(lm.label) ? true : false;
        });
        Plotly.restyle(gd, { visible: lvlVis }, lvlIdx);
      }
      // When hideNoArrows is OFF: leave ALL level lines completely alone.

      // --extra overlays: gated only by their source checkbox (default off)
      applyExtraOverlayVisibility(gd, selSources);

      _currentPage = 0;
      _selectedKeys.clear();
      renderTablePage();
    });
  }

  // ── Reset ────────────────────────────────────────────────────────────────
  function resetFilter() {
    const gd = getPlotDiv();
    cleanupInjected(gd).then(() => {
      _applyActive = false;
      Object.keys(_legendGroupState).forEach(k => delete _legendGroupState[k]);
      document.querySelectorAll('#filter-panel input[type=checkbox]')
              .forEach(c => {
                if (c.id === 'hide-no-arrows') { c.checked = false; return; }
                if (c.classList.contains('source-checks-chk')
                    && EXTRA_SOURCE_NAMES.has(c.dataset.value)) {
                  c.checked = false; return;
                }
                c.checked = true;
              });
      document.getElementById('hide-no-arrows').checked = false;
      document.querySelector('input[name="direction"][value="both"]').checked = true;
      ['wl-min','wl-max','de-min','de-max','un-min','un-max','label-search']
        .forEach(id => { const el = document.getElementById(id); if(el) el.value=''; });
      if (gd && gd.data) {
        if (TRACE_META) {
          const idxArr = Object.keys(TRACE_META).map(Number);
          Plotly.restyle(gd, { visible: idxArr.map(i => TRACE_META[i].origVis) }, idxArr);
        }
        if (LEVEL_META) {
          const lvlIdx = Object.keys(LEVEL_META).map(Number);
          Plotly.restyle(gd, { visible: lvlIdx.map(i => LEVEL_META[i].origVis) }, lvlIdx);
        }
        applyExtraOverlayVisibility(gd, new Set());
      }
      _allMatched=[]; _currentPage=0;
      _selectedKeys.clear(); _lastClickIdx=null;
      document.getElementById('result-count').textContent =
        GROT_INITIAL + ' / ' + ALL_TRANS.length
        + ' transitions pre-rendered. Enable regions in the legend or use Isolate.';
      renderTablePage(); updateSelCount();
    });
  }

  function selectAllRegions(val) {
    document.querySelectorAll('.region-checks-chk').forEach(c => c.checked = val);
  }

  window.applyFilter      = applyFilter;
  window.resetFilter      = resetFilter;
  window.selectAllRegions = selectAllRegions;

  // ── Init + legend-click tracker ──────────────────────────────────────────
  // Listen for Plotly legend clicks to track live visibility of Rydberg levels.
  // Plotly does not update data[i].visible on legend click — we must track it.
  function attachLegendTracker(gd) {
    gd.on('plotly_legendclick', ev => {
      const i   = ev.curveNumber;
      const tr  = gd.data[i];
      const cur = tr && tr.visible;
      // After a legend click Plotly will TOGGLE the trace.
      // 'legendonly' or false -> toggled to true; true -> toggled to legendonly.
      _liveVis[i] = (cur === true) ? 'legendonly' : true;

      const lg = (tr && tr.legendgroup) || '';
      if (lg.startsWith('transition_')) {
        const body = lg.slice('transition_'.length);
        const lastUs = body.lastIndexOf('_');
        const rdKey = body;
        setTimeout(() => {
          const region = body.slice(0, lastUs);
          const direction = body.slice(lastUs + 1);
          let anyTrue = false;
          Object.entries(TRACE_META || {}).forEach(([iStr, meta]) => {
            if (!meta.ul || meta.region !== region || meta.direction !== direction)
              return;
            if (gd.data[parseInt(iStr)] && gd.data[parseInt(iStr)].visible === true)
              anyTrue = true;
          });
          _legendGroupState[rdKey] = anyTrue;
        }, 0);
      }
    });
    gd.on('plotly_legenddoubleclick', ev => {
      // Double-click isolates one trace — set all others to legendonly
      // We can't know the final state until after Plotly settles, so
      // re-read after a short delay.
      setTimeout(() => {
        if (!gd.data) return;
        gd.data.forEach((tr, i) => {
          _liveVis[i] = (tr.visible === undefined) ? true : tr.visible;
        });
      }, 100);
    });
  }

  function init() {
    const gd = getPlotDiv();
    if (!gd) { setTimeout(init, 100); return; }
    const setup = () => {
      // Initialise _liveVis from current trace state
      if (gd.data) gd.data.forEach((tr, i) => {
        _liveVis[i] = (tr.visible === undefined) ? true : tr.visible;
      });
      buildTraceMeta();
      attachLegendTracker(gd);
      _allMatched = []; renderTablePage();
      document.getElementById('result-count').textContent =
        GROT_INITIAL + ' / ' + ALL_TRANS.length
        + ' transitions pre-rendered. Enable regions in the legend or use Isolate.';
    };
    if (gd.data) setup();
    else gd.addEventListener('plotly_afterplot', setup, { once:true });
  }

  if (document.readyState === 'loading')
    document.addEventListener('DOMContentLoaded', init);
  else
    init();

})();
</script>
