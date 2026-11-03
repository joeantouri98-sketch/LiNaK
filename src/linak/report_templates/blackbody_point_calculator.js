
<script>
(function(){
  const A0_GS = $!a0g;
  const A0_EX = $!a0e_js;
  const E_REF = $!E_ref, T_REF = $!T_ref, E_AU = $!E_au;
  const HZ_PER_HARTREE = $!hz_per_hartree;
  // Precomputed dynamic-corrected sweep (from the validated Planck-spectrum
  // integral) -- interpolated here rather than recomputed live, since the
  // full integral needs the actual transition network, not a closed form.
  const T_DYN  = $!T_dyn_js;
  const DYN_GS = $!dyn_gs_js;
  const DYN_EX = $!dyn_ex_js;

  const panel = document.getElementById('bb-ctrl');
  const toggle = document.getElementById('bb-ctrl-toggle');
  if (toggle && panel) {
    toggle.addEventListener('click', function(){
      panel.classList.toggle('collapsed');
      const expanded = !panel.classList.contains('collapsed');
      toggle.setAttribute('aria-expanded', expanded ? 'true' : 'false');
    });
  }

  function shiftStaticHz(alpha0, T){
    const E_rms_Vm = E_REF * Math.pow(T/T_REF, 2);
    const E_rms_au = E_rms_Vm / E_AU;
    const E2_au = E_rms_au * E_rms_au;
    const dE_au = -0.5 * alpha0 * E2_au;
    return dE_au * HZ_PER_HARTREE;
  }

  function interp(xs, ys, x){
    if (x <= xs[0]) return ys[0];
    if (x >= xs[xs.length-1]) return ys[ys.length-1];
    let lo=0, hi=xs.length-1;
    while (hi-lo>1){ const mid=(lo+hi)>>1; if (xs[mid] <= x) lo=mid; else hi=mid; }
    const t = (x-xs[lo])/(xs[hi]-xs[lo]);
    return ys[lo]*(1-t) + ys[hi]*t;
  }

  window.bbCalc = function(){
    const T = parseFloat(document.getElementById('bb-T').value);
    const out = document.getElementById('bb-out');
    if (!isFinite(T) || T <= 0) { out.innerHTML = 'Enter a valid temperature.'; return; }
    const dGs = shiftStaticHz(A0_GS, T);
    const tGs = interp(T_DYN, DYN_GS, T);
    let html = `<b>T = ${T.toFixed(1)} K</b><br>`
      + `dnu_stat(ground) = <b>${dGs.toFixed(4)} Hz</b><br>`
      + `dnu_total(ground) = <b>${tGs.toFixed(4)} Hz</b> `
      + `<span style="color:#8a9ac0;font-size:10px">(interp.)</span>`;
    if (A0_EX !== null && DYN_EX !== null) {
      const dEx = shiftStaticHz(A0_EX, T);
      const tEx = interp(T_DYN, DYN_EX, T);
      html += `<br>dnu_stat(excited) = <b>${dEx.toFixed(4)} Hz</b><br>`
            + `dnu_total(excited) = <b>${tEx.toFixed(4)} Hz</b><br>`
            + `dnu_stat(diff) = <b>${(dEx-dGs).toFixed(4)} Hz</b><br>`
            + `dnu_total(diff) = <b>${(tEx-tGs).toFixed(4)} Hz</b>`;
    }
    out.innerHTML = html;
  };
})();
</script>