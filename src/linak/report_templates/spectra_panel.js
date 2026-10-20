
<script>
(function(){
  const ALL   = $!lines_js;
  const SRCC  = $!src_colors_js;
  const SRCI  = $!src_info_js;
  let wlLo=$!wl_lo, wlHi=$!wl_hi;

  // Build source checkboxes
  const se = document.getElementById('srcs');
  Object.entries(SRCI).forEach(([s,n])=>{
    const lbl=document.createElement('label');
    const chk=document.createElement('input');
    chk.type='checkbox'; chk.checked=true;
    chk.dataset.v=s; chk.className='sc';
    const dot=document.createElement('span');
    dot.className='dot'; dot.style.background=SRCC[s]||'#888';
    const b=document.createElement('span');
    b.className='badge'; b.textContent=n;
    lbl.appendChild(dot); lbl.appendChild(chk);
    lbl.appendChild(document.createTextNode('\u00a0'+s));
    lbl.appendChild(b); se.appendChild(lbl);
  });

  function gd(){ return document.querySelector('.plotly-graph-div'); }

  function go(){
    const g=gd(); if(!g||!g.data) return;
    wlLo=parseFloat(document.getElementById('lo').value)||200;
    wlHi=parseFloat(document.getElementById('hi').value)||900;
    const fmin=parseFloat(document.getElementById('fmin').value)||0;
    const ls=document.getElementById('lsrch').value.trim().toLowerCase();
    const sel=new Set([...document.querySelectorAll('.sc:checked')].map(c=>c.dataset.v));

    // Update x-axis range on both rows
    Plotly.relayout(g,{'xaxis.range':[wlLo,wlHi],'xaxis2.range':[wlLo,wlHi]});

    const matched=ALL.filter(l=>{
      if(l.wl<wlLo||l.wl>wlHi) return false;
      if(l.fosc<fmin) return false;
      if(!sel.has(l.src)) return false;
      if(ls && !l.label.toLowerCase().includes(ls)
            && !l.upper.toLowerCase().includes(ls)
            && !l.lower.toLowerCase().includes(ls)) return false;
      return true;
    });

    document.getElementById('mc').textContent=
      matched.length+' / '+ALL.length+' lines';
    document.getElementById('cnt').textContent=matched.length;
    fillTable(matched.slice(0,250));

    // Restyle bar traces: hide lines outside filter, show those inside
    // Trace layout: [dummy_abs, dummy_emis, abs_0, emis_0, abs_1, emis_1, ...]
    // ALL[i].trace_idx === i; trace pair for line i is at indices 2+2*i and 3+2*i
    const matchedIdx=new Set(matched.map(m=>m.trace_idx));
    const vis=[true,true];
    ALL.forEach(l=>{
      const show=matchedIdx.has(l.trace_idx);
      vis.push(show,show);
    });
    Plotly.restyle(g,{visible:vis},Array.from({length:vis.length},(_,i)=>i));
  }

  function fillTable(rows){
    const tb=document.getElementById('lb');
    tb.innerHTML='';
    rows.forEach(l=>{
      const tr=document.createElement('tr');
      const c=SRCC[l.src]||'#888';
      tr.innerHTML=
        `<td style="color:${c};font-weight:bold;max-width:95px;overflow:hidden;
          text-overflow:ellipsis;white-space:nowrap">${l.label}</td>`+
        `<td>${l.wl.toFixed(2)}</td>`+
        `<td>${l.fosc.toFixed(4)}</td>`+
        `<td style="color:${c}">${l.src}</td>`;
      tb.appendChild(tr);
    });
    if(rows.length===250){
      const tr=document.createElement('tr');
      tr.innerHTML='<td colspan="4" style="text-align:center;color:#bbb;font-style:italic">first 250 shown…</td>';
      tb.appendChild(tr);
    }
  }

  window.rgn=function(btn){
    document.querySelectorAll('.rb').forEach(b=>b.classList.remove('on'));
    btn.classList.add('on');
    document.getElementById('lo').value=btn.dataset.lo;
    document.getElementById('hi').value=btn.dataset.hi;
    go();
  };

  window.rst=function(){
    document.getElementById('lo').value=$!wl_lo;
    document.getElementById('hi').value=$!wl_hi;
    document.getElementById('fmin').value=0;
    document.getElementById('lsrch').value='';
    document.querySelectorAll('.sc').forEach(c=>c.checked=true);
    document.querySelectorAll('.rb').forEach(b=>b.classList.remove('on'));
    document.querySelector('.rb[data-lo="200"]').classList.add('on');
    document.getElementById('mc').textContent='';
    document.getElementById('cnt').textContent='–';
    fillTable([]);
    const g=gd();
    if(g) Plotly.relayout(g,{'xaxis.range':[$!wl_lo,$!wl_hi],
                               'xaxis2.range':[$!wl_lo,$!wl_hi]});
    // Restore all traces visible
    if(g&&g.data){
      const vis=g.data.map(()=>true);
      Plotly.restyle(g,{visible:vis},Array.from({length:vis.length},(_,i)=>i));
    }
  };

  window.go=go;

  document.addEventListener('keydown',function(e){
    if(e.target.tagName==='INPUT') return;
    const lo=parseFloat(document.getElementById('lo').value)||200;
    const hi=parseFloat(document.getElementById('hi').value)||900;
    const span=hi-lo, step=span*0.1;
    if(e.key==='ArrowLeft'){
      document.getElementById('lo').value=Math.max(10,lo-step).toFixed(0);
      document.getElementById('hi').value=Math.max(10+span,hi-step).toFixed(0);
      go();
    } else if(e.key==='ArrowRight'){
      document.getElementById('lo').value=(lo+step).toFixed(0);
      document.getElementById('hi').value=(hi+step).toFixed(0);
      go();
    } else if(e.key==='+'||e.key==='='){
      const mid=(lo+hi)/2,hs=span*0.4;
      document.getElementById('lo').value=(mid-hs).toFixed(0);
      document.getElementById('hi').value=(mid+hs).toFixed(0);
      go();
    } else if(e.key==='-'){
      const mid=(lo+hi)/2,hs=span*0.65;
      document.getElementById('lo').value=Math.max(10,mid-hs).toFixed(0);
      document.getElementById('hi').value=(mid+hs).toFixed(0);
      go();
    } else if(e.key==='r'||e.key==='R'){
      rst();
    }
  });

  // Init after Plotly renders
  function init(){
    const g=gd();
    if(!g||!g.data){ setTimeout(init,150); return; }
    go();
  }
  if(document.readyState==='loading') document.addEventListener('DOMContentLoaded',init);
  else init();
})();
</script>