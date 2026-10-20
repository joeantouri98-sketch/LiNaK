
<script>
(function() {
  const EDGES3     = $!EE_EDGES_JS;
  const REGION_COL = $!EE_RC_JS;
  const SRC_COL    = $!EE_SRC_COLORS_JS;
  const EE_START   = $!ee_edge_start3;
  const N_EE       = $!n_ee_edges3;
  const PAGE_SIZE  = 80;
  let _matched3 = [];
  let _page3 = 0;
  let _lastShow = new Array(N_EE).fill(false);
  let _selectedEE = new Set();
  let _lastClickIdx = null;
  const KIND_LABEL = { gs: 'Ground → excited', ee: 'Excited → excited' };
  const REG_SHORT = {
    'X-ray':'XR','Vacuum UV':'VUV','UV':'UV','Visible':'Vis',
    'Near-IR':'NIR','Mid/Far-IR':'IR','Microwave':'MW'
  };

  function gd3() { return document.querySelector('.plotly-graph-div'); }
  function selKeyEE(e) { return e.ul + '\x00' + e.ll; }

  function updateSelCountEE() {
    const el = document.getElementById('ee-sel-count');
    if (el) el.textContent = _selectedEE.size
      ? `${_selectedEE.size} selected` : '';
  }

  function clearSelectionEE() {
    _selectedEE.clear();
    _lastClickIdx = null;
    renderTable3();
    updateSelCountEE();
  }

  function parseN(lbl) {
    const m = String(lbl).match(/^(\d+)/);
    return m ? parseInt(m[1]) : NaN;
  }

  function makeChecks(containerId, items, cls, colorFn, countFn, checkedDefault) {
    const el = document.getElementById(containerId);
    if (!el) return;
    items.forEach(item => {
      const lbl = document.createElement('label');
      const chk = document.createElement('input');
      chk.type = 'checkbox';
      chk.checked = typeof checkedDefault === 'function'
        ? checkedDefault(item) : !!checkedDefault;
      chk.dataset.value = item;
      chk.className = cls;
      if (colorFn) {
        const dot = document.createElement('span');
        dot.className = 'color-dot';
        dot.style.background = colorFn(item);
        lbl.appendChild(dot);
      }
      lbl.appendChild(chk);
      lbl.appendChild(document.createTextNode('\u00a0' + item));
      const badge = document.createElement('span');
      badge.className = 'count-badge';
      badge.textContent = countFn(item);
      lbl.appendChild(badge);
      el.appendChild(lbl);
    });
  }

  const regions = [...new Set(EDGES3.map(e => e.reg))].filter(Boolean).sort((a,b) => {
    const o = $!region_order_js;
    return o.indexOf(a) - o.indexOf(b);
  });
  const sources = [...new Set(EDGES3.map(e => e.src))].sort();
  const ltypes  = [...new Set(EDGES3.map(e => e.ltype))].sort();
  const kinds   = ['gs', 'ee'].filter(k => EDGES3.some(e => e.kind === k));

  kinds.forEach(k => {
    const el = document.getElementById('ee-kind-checks');
    if (!el) return;
    const lbl = document.createElement('label');
    const chk = document.createElement('input');
    chk.type = 'checkbox';
    chk.checked = true;
    chk.dataset.value = k;
    chk.className = 'ee-kc';
    lbl.appendChild(chk);
    lbl.appendChild(document.createTextNode('\u00a0' + (KIND_LABEL[k] || k)));
    const badge = document.createElement('span');
    badge.className = 'count-badge';
    badge.textContent = EDGES3.filter(e => e.kind === k).length;
    lbl.appendChild(badge);
    el.appendChild(lbl);
  });

  makeChecks('ee-region-checks', regions, 'ee-rc',
    r => REGION_COL[r] || '#999', r => EDGES3.filter(e => e.reg === r).length, false);
  makeChecks('ee-source-checks', sources, 'ee-sc',
    s => SRC_COL[s] || '#999', s => EDGES3.filter(e => e.src === s).length,
    s => s === 'NIST' || s === 'topology');
  makeChecks('ee-ltype-checks', ltypes, 'ee-lc', null,
    t => EDGES3.filter(e => e.ltype === t).length, true);

  function getChecked(cls) {
    return new Set([...document.querySelectorAll('.' + cls + ':checked')]
                   .map(c => c.dataset.value));
  }

  function parseOpt(id, def) {
    const v = document.getElementById(id).value.trim();
    return v === '' ? def : parseFloat(v);
  }

  function parseIntOpt(id, def) {
    const v = document.getElementById(id).value.trim();
    if (v === '') return def;
    const n = parseInt(v, 10);
    return isNaN(n) ? def : n;
  }

  function edgeN(e, which) {
    const stored = which === 'u' ? e.un : e.ln;
    const lbl = which === 'u' ? (e.ulp || e.ul) : (e.llp || e.ll);
    if (stored > 0) return stored;
    return parseN(lbl);
  }

  function nInRange(n, nMin, nMax, hasMin, hasMax) {
    if (isNaN(n)) return false;
    if (hasMin && n < nMin) return false;
    if (hasMax && n > nMax) return false;
    return true;
  }

  function renderTable3() {
    const tb = document.getElementById('ee-trans-tbody');
    const pg = document.getElementById('ee-page-info');
    const pager = document.getElementById('ee-table-pager');
    const pgLabel = document.getElementById('ee-page-label');
    if (!tb) return;
    tb.innerHTML = '';
    const total = _matched3.length;
    const nPages = Math.ceil(total / PAGE_SIZE) || 1;
    _page3 = Math.max(0, Math.min(_page3, nPages - 1));
    const start = _page3 * PAGE_SIZE;
    const rows = _matched3.slice(start, start + PAGE_SIZE);
    rows.forEach((e, rowIdx) => {
      const globalIdx = start + rowIdx;
      const key = selKeyEE(e);
      const isSel = _selectedEE.has(key);
      const col = REGION_COL[e.reg] || '#999';
      const tr = document.createElement('tr');
      tr.style.cursor = 'pointer';
      tr.style.background = isSel ? '#c8e0ff' : '';
      tr.title = 'Click / checkbox to select · Shift+click range · Dbl-click isolate';

      const chkTd = document.createElement('td');
      const chk = document.createElement('input');
      chk.type = 'checkbox';
      chk.checked = isSel;
      chk.style.cssText = 'cursor:pointer;accent-color:#4a7ec7;';
      chk.addEventListener('change', ev => {
        ev.stopPropagation();
        if (chk.checked) _selectedEE.add(key);
        else _selectedEE.delete(key);
        tr.style.background = chk.checked ? '#c8e0ff' : '';
        _lastClickIdx = globalIdx;
        updateSelCountEE();
      });
      chkTd.appendChild(chk);
      tr.appendChild(chkTd);

      [[e.ulp || e.ul, `color:${col};font-weight:bold`],
       [e.llp || e.ll, ''],
       [e.wl.toFixed(1), ''],
       [REG_SHORT[e.reg] || e.reg, `color:${col}`],
       [e.fosc.toExponential(2), ''],
      ].forEach(([text, style]) => {
        const td = document.createElement('td');
        td.textContent = text;
        if (style) td.style.cssText = style;
        tr.appendChild(td);
      });

      tr.addEventListener('click', ev => {
        if (ev.target === chk) return;
        if (ev.shiftKey && _lastClickIdx !== null) {
          const lo = Math.min(_lastClickIdx, globalIdx);
          const hi = Math.max(_lastClickIdx, globalIdx);
          for (let i = lo; i <= hi; i++) {
            if (_matched3[i]) _selectedEE.add(selKeyEE(_matched3[i]));
          }
          _lastClickIdx = globalIdx;
          renderTable3();
        } else {
          if (_selectedEE.has(key)) {
            _selectedEE.delete(key);
            chk.checked = false;
            tr.style.background = '';
          } else {
            _selectedEE.add(key);
            chk.checked = true;
            tr.style.background = '#c8e0ff';
          }
          _lastClickIdx = globalIdx;
        }
        updateSelCountEE();
      });

      tr.addEventListener('dblclick', ev => {
        ev.preventDefault();
        isolateEEByKeys([key]);
      });

      tb.appendChild(tr);
    });

    if (pg) {
      pg.textContent = total > 0
        ? `${total} matched` : '';
    }
    if (pager && pgLabel) {
      if (total > PAGE_SIZE) {
        pager.style.display = 'flex';
        pgLabel.textContent = `Page ${_page3 + 1}/${nPages}`;
      } else {
        pager.style.display = 'none';
      }
    }
    updateSelCountEE();
  }

  function tablePageEE(delta) {
    _page3 += delta;
    renderTable3();
  }

  function setEdgeVisibility(showArr) {
    const g = gd3();
    if (!g || !g.data) return;
    _lastShow = showArr.slice();
    const vis = g.data.map((tr, i) => {
      const ei = i - EE_START;
      if (ei >= 0 && ei < N_EE) return showArr[ei] || false;
      return tr.visible !== false;
    });
    Plotly.restyle(g, {visible: vis}, [...Array(g.data.length).keys()]);
  }

  function applyEEFilter() {
    const selR = getChecked('ee-rc');
    const selS = getChecked('ee-sc');
    const selL = getChecked('ee-lc');
    const selK = getChecked('ee-kc');
    const wlMin = parseOpt('ee-wl-min', 0);
    const wlMax = parseOpt('ee-wl-max', 1e9);
    const nMinRaw = document.getElementById('ee-n-min').value.trim();
    const nMaxRaw = document.getElementById('ee-n-max').value.trim();
    const hasNMin = nMinRaw !== '';
    const hasNMax = nMaxRaw !== '';
    const nMin = hasNMin ? parseInt(nMinRaw, 10) : 0;
    const nMax = hasNMax ? parseInt(nMaxRaw, 10) : 999;
    const fMin  = parseOpt('ee-fmin', 0);
    const ls    = (document.getElementById('ee-label-search').value || '').trim().toLowerCase();

    if (selR.size === 0 || selS.size === 0) {
      alert('Select at least one spectral region and one fosc source.');
      return;
    }
    if (selK.size === 0) {
      alert('Select at least one transition type (gs or ee).');
      return;
    }

    const show = EDGES3.map(e => {
      if (!selK.has(e.kind)) return false;
      if (!selR.has(e.reg)) return false;
      if (!selS.has(e.src)) return false;
      if (!selL.has(e.ltype)) return false;
      if (e.wl < wlMin || e.wl > wlMax) return false;
      if (e.fosc < fMin) return false;
      if (ls && !e.ul.toLowerCase().includes(ls) && !e.ll.toLowerCase().includes(ls)
          && !(e.ulp || '').toLowerCase().includes(ls)
          && !(e.llp || '').toLowerCase().includes(ls)) return false;
      if (hasNMin || hasNMax) {
        const uN = edgeN(e, 'u');
        const lN = edgeN(e, 'l');
        if (!nInRange(uN, nMin, nMax, hasNMin, hasNMax)) return false;
        if (!nInRange(lN, nMin, nMax, hasNMin, hasNMax)) return false;
      }
      return true;
    });

    _matched3 = EDGES3.filter((_, i) => show[i]);
    _page3 = 0;
    _selectedEE.clear();
    _lastClickIdx = null;
    setEdgeVisibility(show);
    const rc = document.getElementById('ee-result-count');
    if (rc) rc.textContent = `${_matched3.length} / ${EDGES3.length} transitions shown`;
    renderTable3();
  }

  function hideAllEE() {
    document.querySelectorAll('.ee-rc').forEach(c => c.checked = false);
    setEdgeVisibility(new Array(N_EE).fill(false));
    _matched3 = [];
    _page3 = 0;
    _selectedEE.clear();
    _lastClickIdx = null;
    const rc = document.getElementById('ee-result-count');
    if (rc) rc.textContent = 'No transitions shown';
    renderTable3();
  }

  function resetEEFilter() {
    document.querySelectorAll('.ee-rc').forEach(c => c.checked = false);
    document.querySelectorAll('.ee-sc').forEach(c => {
      c.checked = (c.dataset.value === 'NIST' || c.dataset.value === 'topology');
    });
    document.querySelectorAll('.ee-lc').forEach(c => c.checked = true);
    document.querySelectorAll('.ee-kc').forEach(c => c.checked = true);
    ['ee-wl-min','ee-wl-max','ee-n-min','ee-n-max','ee-label-search'].forEach(id => {
      const el = document.getElementById(id); if (el) el.value = '';
    });
    document.getElementById('ee-fmin').value = '0';
    hideAllEE();
  }

  function isolateEEByKeys(keys) {
    const keySet = new Set(keys);
    const show = EDGES3.map(e => keySet.has(selKeyEE(e)));
    setEdgeVisibility(show);
    _matched3 = EDGES3.filter(e => keySet.has(selKeyEE(e)));
    _page3 = 0;
    document.getElementById('ee-result-count').textContent =
      `Isolated ${_matched3.length} transition(s)`;
    renderTable3();
  }

  function isolateSelectedEE() {
    if (_selectedEE.size === 0) {
      alert('Select at least one transition first.');
      return;
    }
    isolateEEByKeys([..._selectedEE]);
  }

  window.applyEEFilter = applyEEFilter;
  window.resetEEFilter = resetEEFilter;
  window.hideAllEE = hideAllEE;
  window.isolateSelectedEE = isolateSelectedEE;
  window.clearSelectionEE = clearSelectionEE;
  window.tablePageEE = tablePageEE;

  function initEE3() {
    const g = gd3();
    if (!g || !g.data) { setTimeout(initEE3, 150); return; }
    hideAllEE();
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', initEE3);
  else initEE3();
})();
</script>
