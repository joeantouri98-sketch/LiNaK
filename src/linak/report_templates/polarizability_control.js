
<script>
(function(){
  const WL_GS = $!WL_GS;
  const A_GS  = $!A_GS;
  const WL_EX = $!WL_EX;
  const A_EX  = $!A_EX;
  const MASS_AMU = $!MASS_AMU;
  const A0 = $!A0, C_SI = $!C_SI, KB_SI = $!KB_SI,
        H_SI = $!H_SI, AMU_KG = $!AMU_KG;
  const U_PREF = 2*Math.PI*A0**3/C_SI;

  function gd(){ return document.querySelector('.plotly-graph-div'); }

  window.asToggle = function(){
    const el  = document.getElementById('as-ctrl');
    const tog = document.getElementById('as-tog');
    el.classList.toggle('collapsed');
    tog.textContent = el.classList.contains('collapsed') ? '▼' : '▲';
  };

  function alphaToUKFactor(I_kWcm2){
    const I_Wm2 = I_kWcm2*1e3/1e-4;
    return -U_PREF*I_Wm2/KB_SI*1e6;
  }

  window.asOnIntensity = function(logI){
    const I = Math.pow(10, parseFloat(logI));
    document.getElementById('as-ival').textContent = I.toFixed(2);
    const titleI = document.getElementById('title-I');
    if (titleI) titleI.textContent = I.toFixed(2);
    const g = gd(); if(!g||!g.layout) return;
    const yr = g.layout.yaxis.range;   // alpha axis range never changes
    const factor = alphaToUKFactor(I);
    Plotly.relayout(g, {'yaxis3.range': [yr[1]*factor, yr[0]*factor]});
  };

  function asInterp(xs, ys, x){
    if (x <= xs[0]) return ys[0];
    if (x >= xs[xs.length-1]) return ys[ys.length-1];
    let lo=0, hi=xs.length-1;
    while (hi-lo>1){ const mid=(lo+hi)>>1; if (xs[mid] <= x) lo=mid; else hi=mid; }
    const t = (x-xs[lo])/(xs[hi]-xs[lo]);
    return ys[lo]*(1-t) + ys[hi]*t;
  }

  window.asCalc = function(){
    const wl = parseFloat(document.getElementById('as-wl').value);
    const waist_um = parseFloat(document.getElementById('as-waist').value);
    const logI = parseFloat(document.getElementById('as-Islider').value);
    const I_kWcm2 = Math.pow(10, logI);
    const out = document.getElementById('as-out');
    if (!isFinite(wl)) { out.innerHTML = 'Enter a valid wavelength.'; return; }

    const a_gs = asInterp(WL_GS, A_GS, wl);
    const a_ex = A_EX ? asInterp(WL_EX, A_EX, wl) : NaN;
    const factor = alphaToUKFactor(I_kWcm2);
    const U_gs = a_gs * factor, U_ex = a_ex * factor;

    let html = `<b>λ = ${wl.toFixed(2)} nm</b>, I = ${I_kWcm2.toFixed(2)} kW/cm²<br>`
      + `α_gs = ${a_gs.toFixed(2)} au &nbsp; U_gs = <b>${U_gs.toFixed(3)} μK</b>`;
    if (A_EX) {
      html += `<br>α_ex = ${a_ex.toFixed(2)} au &nbsp; U_ex = <b>${U_ex.toFixed(3)} μK</b>`;
    }

    if (MASS_AMU) {
      const w0 = waist_um*1e-6, wl_m = wl*1e-9, m = MASS_AMU*AMU_KG;
      const zR = Math.PI*w0*w0/wl_m;
      function trapFreq(U_uK){
        const U0_J = Math.abs(U_uK)*1e-6*KB_SI;
        const wr = Math.sqrt(4*U0_J/(m*w0*w0));
        const wz = Math.sqrt(2*U0_J/(m*zR*zR));
        return [wr/(2*Math.PI), wz/(2*Math.PI)];
      }
      if (U_gs < 0) {
        const [fr,fz] = trapFreq(U_gs);
        html += `<br>Ground trap (w0=${waist_um}μm): ω_r/2π=${(fr/1e3).toFixed(2)} kHz, `
              + `ω_z/2π=${(fz/1e3).toFixed(3)} kHz`;
      }
      if (U_ex < 0) {
        const [fr,fz] = trapFreq(U_ex);
        html += `<br>Excited trap (w0=${waist_um}μm): ω_r/2π=${(fr/1e3).toFixed(2)} kHz, `
              + `ω_z/2π=${(fz/1e3).toFixed(3)} kHz`;
      }
    } else {
      html += `<br><span style="color:#667">mass unknown — no trap freq.</span>`;
    }
    out.innerHTML = html;
  };
})();
</script>