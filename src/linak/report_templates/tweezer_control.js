
<script>
(function(){
  const EL = $!el_js;
  const WL = $!wl_data;
  const RE = $!re_data;
  const IM = $!im_data;
  const MASS = $!mass;
  const U_PREF = $!U_PREF;
  const R_PREF = $!R_PREF;
  const H = $!H_SI, KB = $!KB_SI, AMU = $!AMU_KG;

  function interp(xs, ys, x){
    if (x <= xs[0]) return ys[0];
    if (x >= xs[xs.length-1]) return ys[ys.length-1];
    let lo=0, hi=xs.length-1;
    while (hi-lo>1){ const mid=(lo+hi)>>1; if (xs[mid] <= x) lo=mid; else hi=mid; }
    const t = (x-xs[lo])/(xs[hi]-xs[lo]);
    const a = ys[lo], b = ys[hi];
    if (!isFinite(a) || !isFinite(b)) return NaN;
    return a*(1-t)+b*t;
  }
  function fmt(x, d){
    if (!isFinite(x)) return '—';
    const ax = Math.abs(x);
    if (ax !== 0 && (ax >= 1e4 || ax < 1e-3)) return x.toExponential(d);
    return x.toPrecision(d+1);
  }

  window.twToggle = function(){
    const box = document.getElementById('tw-ctrl');
    box.classList.toggle('collapsed');
    document.getElementById('tw-tog').textContent =
      box.classList.contains('collapsed') ? '▶' : '▼';
  };

  window.twCalc = function(){
    const wl = parseFloat(document.getElementById('tw-wl').value);
    const Ik = parseFloat(document.getElementById('tw-I').value);
    const w0 = parseFloat(document.getElementById('tw-w0').value);
    const out = document.getElementById('tw-out');
    if (!(wl > 0) || !(Ik >= 0) || !(w0 > 0)) {
      out.innerHTML = 'Enter valid λ, I, w₀.'; return;
    }
    const re = interp(WL, RE, wl);
    const im = interp(WL, IM, wl);
    const I = Ik * 1e3 / 1e-4;
    const U = -U_PREF * re * I;
    const UuK = U / KB * 1e6;
    const R = R_PREF * im * I;
    const lam = wl * 1e-9;
    const m = MASS * AMU;
    const Er = (H*H) / (2 * m * lam * lam);
    const Edot = 2 * Er * R;
    const heat = Edot / KB * 1e6;
    const tauH = (Math.abs(U) > 0 && Edot > 0) ? Math.abs(U)/Edot : NaN;
    const tauS = (R > 0) ? 1/R : NaN;
    const zR = Math.PI * (w0*1e-6)*(w0*1e-6) / lam;
    const Ur = Math.abs(U);
    const nur = (Ur > 0) ? Math.sqrt(4*Ur/(m*(w0*1e-6)*(w0*1e-6)))/(2*Math.PI) : NaN;
    const nuz = (Ur > 0) ? Math.sqrt(2*Ur/(m*zR*zR))/(2*Math.PI) : NaN;

    out.innerHTML =
      '<b>' + EL + ' @ ' + wl.toFixed(1) + ' nm</b><br>' +
      'I = ' + Ik + ' kW/cm² · w₀ = ' + w0 + ' µm<br>' +
      'Re[α] = <b>' + fmt(re,3) + '</b> au · Im[α] = <b>' + fmt(im,3) + '</b> au<br>' +
      'U/k_B = <b>' + fmt(UuK,3) + '</b> µK<br>' +
      'R_sc = <b>' + fmt(R,3) + '</b> s⁻¹ · ⟨Δt⟩ = <b>' + fmt(tauS,3) + '</b> s<br>' +
      'Heating = <b>' + fmt(heat,3) + '</b> µK/s<br>' +
      'τ_heat ≈ <b>' + fmt(tauH,3) + '</b> s<br>' +
      'ν_r = <b>' + fmt(nur,2) + '</b> Hz · ν_z = <b>' + fmt(nuz,2) + '</b> Hz' +
      '<div style="margin-top:6px;color:#6a7a9a;font-size:10px">' +
      'Live estimate: α(λ) interpolated from spectrum; scales ∝ I. ' +
      'Near resonances prefer the Python point calc.</div>';
  };
  twCalc();
})();
</script>