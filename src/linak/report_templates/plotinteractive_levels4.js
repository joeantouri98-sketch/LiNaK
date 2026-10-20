
<script>
(function(){
  const EDGES4 = $!edges_js4;
  const RC4    = $!rc_js4;
  const N_ET4  = $!n_edge4_str;

  window.tog4 = function(){
    const fp=document.getElementById('fp4');
    fp.classList.toggle('collapsed');
    fp.querySelector('.tog').textContent=fp.classList.contains('collapsed')?'▼':'▲';
  };

  // Build region checkboxes (all unchecked by default)
  const regions4=[...new Set(EDGES4.map(e=>e.region))].sort((a,b)=>{
    const o=$!region_order_js;
    return o.indexOf(a)-o.indexOf(b);
  });
  const rgEl4=document.getElementById('rg4');
  regions4.forEach(r=>{
    const cnt=EDGES4.filter(e=>e.region===r).length;
    const lbl=document.createElement('label');
    const chk=document.createElement('input');
    chk.type='checkbox'; chk.checked=false;   // ← unchecked by default
    chk.dataset.v=r; chk.className='rc4';
    const dot=document.createElement('span');
    dot.className='dot4'; dot.style.background=RC4[r]||'#888';
    const b=document.createElement('span');
    b.className='bdg4'; b.textContent=cnt;
    lbl.appendChild(dot); lbl.appendChild(chk);
    lbl.appendChild(document.createTextNode('\u00a0'+r));
    lbl.appendChild(b); rgEl4.appendChild(lbl);
  });

  function gd4(){ return document.querySelector('.plotly-graph-div'); }

  // Pagination + multi-select state for the matched-transitions table
  const PAGE_SIZE4 = 80;
  let _matched4    = [];
  let _page4       = 0;
  let _selected4   = new Set();   // Set of edge index (e.i) for selected transitions
  let _isolated4   = [];          // trace indices currently force-shown by isolate
  let _lastShowEdge = [];         // last applyF4 visibility per edge trace

  window.tablePage4 = function(delta){
    _page4 = Math.max(0, Math.min(_page4+delta, Math.ceil(_matched4.length/PAGE_SIZE4)-1));
    fillTable4Page();
  };

  window.applyF4 = function(){
    const g=gd4(); if(!g||!g.data) return;
    _isolated4 = [];   // clear any prior isolation on new filter apply
    const selR=new Set([...document.querySelectorAll('.rc4:checked')].map(c=>c.dataset.v));
    const selS=new Set([...document.querySelectorAll('.sc4:checked')].map(c=>c.dataset.v));
    const nLo=parseInt(document.getElementById('nlo4').value)||$!gs_n4_str;
    const nHi=parseInt(document.getElementById('nhi4').value)||$!n_max4_str;
    const wLo=parseFloat(document.getElementById('wlo4').value)||0;
    const wHi=parseFloat(document.getElementById('whi4').value)||99999;
    const fMin=parseFloat(document.getElementById('fmin4').value)||0;
    const ls=document.getElementById('ls4').value.trim().toLowerCase();

    const showEdge=EDGES4.map(e=>{
      if(!selR.has(e.region))      return false;
      // Both endpoint sources must be selected — OR logic wrongly kept QDT→NIST
      // edges visible when only NIST was checked (lower_source=NIST passed alone).
      if(selS.size===0)            return false;
      if(!selS.has(e.us)||!selS.has(e.ls)) return false;
      if(e.wl<wLo||e.wl>wHi)     return false;
      if(e.fosc<fMin)              return false;
      if(ls&&!e.ul.toLowerCase().includes(ls)&&!e.ll.toLowerCase().includes(ls)
         &&!(e.ulp||'').toLowerCase().includes(ls)
         &&!(e.llp||'').toLowerCase().includes(ls)) return false;
      const uN=parseInt(e.ul), lN=parseInt(e.ll);
      let nMatch=false;
      if(!isNaN(uN)&&uN>=nLo&&uN<=nHi) nMatch=true;
      if(!isNaN(lN)&&lN>=nLo&&lN<=nHi) nMatch=true;
      if((!isNaN(uN)||!isNaN(lN))&&!nMatch) return false;
      return true;
    });

    _lastShowEdge = showEdge.slice();

    _matched4 = EDGES4.filter((_,i)=>showEdge[i]);
    _page4    = 0;
    _selected4.clear();
    document.getElementById('mc4').textContent=_matched4.length+' / '+EDGES4.length+' transitions';
    document.getElementById('cnt4').textContent=_matched4.length;

    const total=g.data.length;
    const vis=g.data.map((tr,i)=>{
      if(i===0) return true;                          // ionization grid
      if(i>=1&&i<=N_ET4) return showEdge[i-1]||false; // edge traces
      // Node traces grouped by g3d_src_NIST / g3d_src_QDT model
      const lg = tr.legendgroup || '';
      if (lg.startsWith('g3d_src_')) {
        const src = lg.slice('g3d_src_'.length);
        return selS.size > 0 && selS.has(src);
      }
      return true;                                    // text labels
    });
    Plotly.restyle(g,{visible:vis},[...Array(total).keys()]);
    fillTable4Page();
    updateSelCount4();
  };

  window.hideAll4 = function(){
    const g=gd4(); if(!g) return;
    document.querySelectorAll('.rc4').forEach(c=>c.checked=false);
    const selS=new Set([...document.querySelectorAll('.sc4:checked')].map(c=>c.dataset.v));
    const vis=g.data.map((tr,i)=>{
      if(i===0) return true;
      if(i>=1&&i<=N_ET4) return false;
      const lg = tr.legendgroup || '';
      if (lg.startsWith('g3d_src_')) {
        const src = lg.slice('g3d_src_'.length);
        return selS.size > 0 && selS.has(src);
      }
      return true;
    });
    Plotly.restyle(g,{visible:vis},[...Array(g.data.length).keys()]);
    document.getElementById('mc4').textContent='';
    document.getElementById('cnt4').textContent='–';
    _matched4=[]; _page4=0; _selected4.clear(); _isolated4=[];
    fillTable4Page();
    updateSelCount4();
  };

  window.rstF4 = function(){
    document.querySelectorAll('.rc4').forEach(c=>c.checked=false);
    document.querySelectorAll('.sc4').forEach(c=>c.checked=true);
    document.getElementById('nlo4').value='$!gs_n4_str';
    document.getElementById('nhi4').value='$!n_max4_str';
    document.getElementById('wlo4').value=0;
    document.getElementById('whi4').value=99999;
    document.getElementById('fmin4').value=0;
    document.getElementById('ls4').value='';
    hideAll4();
  };

  function updateSelCount4(){
    const el=document.getElementById('sel-count4');
    if(!el) return;
    el.textContent = _selected4.size>0
      ? `${_selected4.size} transition${_selected4.size>1?'s':''} selected`
      : '';
  }

  // Isolate selected transitions 
  // Each EDGES4 entry already carries its own unique Plotly trace index (e.i),
  // so isolation is a direct visibility restyle — no injected traces needed.
  window.isolateSelected4 = function(){
    const g=gd4(); if(!g||!g.data) return;
    if(_selected4.size===0){ alert('Select at least one transition first.'); return; }

    const selectedEdges = EDGES4.filter(e=>_selected4.has(e.i));
    _isolated4 = selectedEdges.map(e=>e.i);

    const total=g.data.length;
    const selS=new Set([...document.querySelectorAll('.sc4:checked')].map(c=>c.dataset.v));
    const vis=g.data.map((tr,i)=>{
      if(i===0) return true;                  // ionization grid always on
      if(i>=1&&i<=N_ET4) return _isolated4.includes(i);  // only selected edges
      const lg = tr.legendgroup || '';
      if (lg.startsWith('g3d_src_')) {
        const src = lg.slice('g3d_src_'.length);
        return selS.size > 0 && selS.has(src);
      }
      return true;
    });
    Plotly.restyle(g,{visible:vis},[...Array(total).keys()]);

    const desc = selectedEdges.length===1
      ? `${selectedEdges[0].ulp || selectedEdges[0].ul} → ${selectedEdges[0].llp || selectedEdges[0].ll} (${selectedEdges[0].wl.toFixed(1)} nm)`
      : `${selectedEdges.length} transitions`;
    document.getElementById('mc4').textContent = `Isolated: ${desc}`;
  };

  window.clearSelection4 = function(){
    _selected4.clear();
    if(_isolated4.length>0){
      _isolated4=[];
      applyF4();
    } else {
      fillTable4Page();
      updateSelCount4();
    }
  };

  // Paginated table with checkboxes 
  function fillTable4Page(){
    const tb=document.getElementById('lb4'); tb.innerHTML='';
    const pager  = document.getElementById('pager4');
    const pgInfo = document.getElementById('page-info4');
    const total  = _matched4.length;
    const nPages = Math.ceil(total/PAGE_SIZE4)||1;
    _page4 = Math.max(0, Math.min(_page4, nPages-1));
    const start  = _page4*PAGE_SIZE4;
    const rows   = _matched4.slice(start, start+PAGE_SIZE4);

    rows.forEach(e=>{
      const tr=document.createElement('tr');
      const c=RC4[e.region]||'#888';
      const isSel = _selected4.has(e.i);
      tr.style.background = isSel ? 'rgba(74,126,199,0.25)' : '';
      tr.style.cursor = 'pointer';
      tr.title = 'Click row or checkbox to select for isolation';

      const chkTd = document.createElement('td');
      const chk   = document.createElement('input');
      chk.type = 'checkbox'; chk.checked = isSel;
      chk.style.cssText = 'cursor:pointer;accent-color:#4a7ec7;';
      chk.addEventListener('change', ev=>{
        ev.stopPropagation();
        if(chk.checked) _selected4.add(e.i); else _selected4.delete(e.i);
        tr.style.background = chk.checked ? 'rgba(74,126,199,0.25)' : '';
        updateSelCount4();
      });
      chkTd.appendChild(chk);
      tr.appendChild(chkTd);

      const cells = [
        [e.ulp || e.ul, `color:${c};font-weight:bold`],
        [e.llp || e.ll, `color:${c}`],
        [e.wl.toFixed(1), `color:${c}`],
        [e.fosc>0?e.fosc.toFixed(4):'—', `color:${c}`],
     ];

      cells.forEach(([text,style])=>{
        const td=document.createElement('td');
        td.textContent=text;
        if(style) td.style.cssText=style;
        tr.appendChild(td);
      });

      tr.addEventListener('click', ev=>{
        if(ev.target===chk) return;
        chk.checked = !chk.checked;
        if(chk.checked) _selected4.add(e.i); else _selected4.delete(e.i);
        tr.style.background = chk.checked ? 'rgba(74,126,199,0.25)' : '';
        updateSelCount4();
      });

      // Double-click: isolate immediately just this one transition
      tr.addEventListener('dblclick', ev=>{
        ev.preventDefault();
        _selected4 = new Set([e.i]);
        isolateSelected4();
        fillTable4Page();
      });

      tb.appendChild(tr);
    });

    if(total>PAGE_SIZE4){
      pager.style.display='flex';
      pgInfo.textContent=`Page ${_page4+1}/${nPages} (${total} total)`;
    } else {
      pager.style.display='none';
    }
  }

  function init4(){
    const g=gd4();
    if(!g||!g.data){ setTimeout(init4,150); return; }
    _lastShowEdge = new Array(N_ET4).fill(false);

    // Manual legend double-click isolation for 3D plots 
    // Plotly's built-in itemdoubleclick='toggleothers' doesn't work reliably
    // in 3D. We track clicks ourselves: two clicks on the same legend item
    // within 400ms = isolate that group; double-click again = restore all.
    let _lastClickGroup = null;
    let _lastClickTime  = 0;
    let _isolated       = false;

    g.on('plotly_legendclick', function(ev) {
      const now     = Date.now();
      const group   = (g.data[ev.curveNumber].legendgroup) || String(ev.curveNumber);
      const isDouble = (group === _lastClickGroup && now - _lastClickTime < 400);
      _lastClickGroup = group;
      _lastClickTime  = now;

      if (!isDouble) return;  // single click — let Plotly handle toggle normally

      // Double-click: isolate or restore
      ev.event.preventDefault();
      ev.event.stopPropagation();

      if (_isolated && _isolated === group) {
        // Already isolated this group — restore last Apply filter
        _isolated = false;
        const selS=new Set([...document.querySelectorAll('.sc4:checked')].map(c=>c.dataset.v));
        const vis = g.data.map((tr, i) => {
          if (i === 0) return true;
          if (i >= 1 && i <= N_ET4) return !!_lastShowEdge[i - 1];
          const lg = tr.legendgroup || '';
          if (lg.startsWith('g3d_src_')) {
            const src = lg.slice('g3d_src_'.length);
            return selS.size > 0 && selS.has(src);
          }
          return true;
        });
        Plotly.restyle(g, {visible: vis}, [...Array(g.data.length).keys()]);
        document.getElementById('mc4').textContent =
          _matched4.length + ' / ' + EDGES4.length + ' transitions';
      } else {
        // Isolate: show only edges in this legend group that passed the last Apply
        _isolated = group;
        const vis = g.data.map((tr, i) => {
          if (i === 0) return true;
          if (i > N_ET4) return true;
          return tr.legendgroup === group && !!_lastShowEdge[i - 1];
        });
        Plotly.restyle(g, {visible: vis}, [...Array(g.data.length).keys()]);
      }
      return false;
    });
  }
  if(document.readyState==='loading') document.addEventListener('DOMContentLoaded',init4);
  else init4();
})();
</script>
