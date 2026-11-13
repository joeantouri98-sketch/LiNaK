const $ = id => document.getElementById(id);
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const esc = s => String(s == null ? '' : s).replace(/&/g, '&amp;').replace(/</g, '&lt;');
const SUP = {'-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹'};
const sci = (x, p) => {
  if (x == null || !isFinite(x)) return '–'; if (x === 0) return '0'; p = p || 4;
  const t = Number(x.toPrecision(p)), e = Math.floor(Math.log10(Math.abs(t)));
  return (e < 5 && e > -3) ? t.toString() : (t / 10 ** e).toFixed(p - 1) + '×10' + String(e).split('').map(c => SUP[c]).join('');
};
const fx = (x, d) => x == null || !isFinite(x) ? '–' : (+x).toFixed(d);
const LS = 'spdfghi';
const A0 = 0.529177210544;                       // Å
const UNITS = {a0: [1, 'a₀'], A: [A0, 'Å'], nm: [A0 / 10, 'nm']};
const ST = D.states, RENDER = D.render;
ST.forEach(s => {
  const k = s.umax / 32767;
  s.uu = Float64Array.from(s.u, v => v * k);
  s.rr = Float64Array.from(s.uu, (_, i) => Math.exp(s.x0 + i * s.dx));
  s.R = makeRadial(s.uu, s.x0, s.dx, s.lint);
  s.L = LS[s.lint] || '?';
});
const LPRESENT = [...new Set(ST.map(s => s.lint))].sort((a, b) => a - b);
const S = {tab: '3d', i: 0, mk: 'sum', norm: 'val', basis: 'complex', chan: 0, scale: 1, cmap: 0, unit: 'A', plane: 'xz', rq: 'P', rlog: true,
  pins: [], filt: new Set(LPRESENT), theme: 'auto', rail: true, sort: [0, 1]};
(function () {
  const h = new URLSearchParams(location.hash.slice(1));
  if (['3d', 'sl', 'rad', 'val', 'sc'].includes(h.get('tab'))) S.tab = h.get('tab');
  const k = ST.findIndex(s => s.lbl === h.get('s')); if (k >= 0) S.i = k;
})();
const uf = () => UNITS[S.unit][0], un = () => UNITS[S.unit][1];
const cur = () => ST[S.i];
const PAL = ['--nist', '--soc', '--sel', '--td', '--ov4'];
const LCOL = ['--nist', '--soc', '--sel', '--td', '--ov4', '--ov5', '--ov6'];

function query() {
  const s = cur(), m = S.mk;
  if (m === 'sum' || m === 'sph') return {kind: m, l: s.lint, m: 0, mjT: 0, jT: s.jt, basis: S.basis};
  const [k, v] = m.split(':');
  if (k === 'mj') return {kind: 'mj', l: s.lint, m: 0, mjT: +v, jT: s.jt, basis: S.basis};
  return {kind: 'ml', l: s.lint, m: +v, mjT: 0, jT: s.jt, basis: S.basis};
}
function chanOk(c) { const q = query(); return c === 0 || (q.kind === 'ml' && (c === 1 || S.basis === 'complex')); }
function subLabel() {
  const q = query(), s = cur();
  if (q.kind === 'sum') return s.jt > 0 ? 'level density, Σ over m<sub>j</sub>' : 'Σ over m<sub>ℓ</sub>';
  if (q.kind === 'sph') return 'radial probability r²R²';
  if (q.kind === 'mj') return '|m<sub>j</sub>| = ' + q.mjT + '/2, spin-summed';
  return 'm<sub>ℓ</sub> = ' + q.m + (S.basis === 'real' ? ' (real basis)' : ' (complex Y<sub>ℓ</sub><sup>m</sup>)');
}
const chanName = () => ['|ψ|²', 'Re ψ', 'Im ψ', 'arg ψ'][S.chan];
// colour scale reference: the maximum outside the core (core saturated) or the global maximum
const normR = s => (S.norm === 'val' && s.rc) ? s.rc : 0;
const maxLab = () => (S.norm === 'val' && cur().rc) ? 'max (r > r<sub>c</sub>)' : 'max';
const maxTxt = () => (S.norm === 'val' && cur().rc) ? 'max(r > rc)' : 'max';

// ═════════════════════════ 3D: WebGL2 volume ═════════════════════════
const CMAPN = ['hwave', 'viridis', 'plasma', 'hot', 'electric'];
const STOPS = {
  hwave: [[0, [.086, .043, .224]], [.12, [.271, .039, .412]], [.25, [.459, .106, .427]], [.38, [.643, .173, .376]],
    [.5, [.8, .255, .282]], [.62, [.922, .396, .153]], [.75, [.984, .6, .024]], [.88, [.965, .827, .247]], [1, [1, 1, 1]]],
  viridis: [[0, [.267, .005, .329]], [.2, [.255, .267, .529]], [.4, [.165, .471, .558]], [.6, [.135, .659, .518]], [.8, [.478, .821, .318]], [1, [.992, .906, .144]]],
  plasma: [[0, [.051, .031, .529]], [.2, [.416, 0, .659]], [.4, [.694, .165, .565]], [.6, [.882, .392, .384]], [.8, [.988, .652, .212]], [1, [.941, .976, .129]]],
  hot: [[0, [0, 0, 0]], [.4, [1, 0, 0]], [.75, [1, 1, 0]], [1, [1, 1, 1]]],
  electric: [[0, [0, .122, .247]], [.25, [0, .2, .4]], [.55, [.106, .435, .478]], [.8, [.322, .718, .533]], [1, [.851, .929, .573]]],
  rdbu: [[0, [.13, .4, .675]], [.25, [.404, .663, .812]], [.5, [.969, .969, .969]], [.75, [.957, .647, .51]], [1, [.698, .094, .169]]],
};
const plotlyScale = name => STOPS[name].map(([t, c]) => [t, 'rgb(' + c.map(v => Math.round(v * 255)).join(',') + ')']);
function stopColor(name, t) {
  const s = STOPS[name]; t = Math.min(Math.max(t, 0), 1);
  for (let i = 0; i < s.length - 1; i++) if (t <= s[i + 1][0]) { const u = (t - s[i][0]) / (s[i + 1][0] - s[i][0]); return s[i][1].map((v, j) => v + (s[i + 1][1][j] - v) * u); }
  return s[s.length - 1][1];
}
function hsv2rgbJS(h, s, v) { const f = n => { const k = (n + h * 6) % 6; return v - v * s * Math.max(Math.min(k, 4 - k, 1), 0); }; return [f(5), f(3), f(1)]; }
const HSV_SCALE = Array.from({length: 13}, (_, i) => [i / 12, 'rgb(' + hsv2rgbJS(i / 12, .85, 1).map(v => Math.round(v * 255)).join(',') + ')']);
const sliderVal = (id, d) => { const e = $(id); return e ? parseFloat(e.value) : d; };

function makeVolumeRenderer() {
  const STUB = {load() {}, clear() {}, resetView() {}, png() { return null; }};
  let hasData = false, half = 6, freshState = true, loadedKind = 'sum';
  const canvas = $('glcanvas'), ov = $('overlay'), ctx = ov.getContext('2d');
  const gl = canvas.getContext('webgl2', {antialias: false, alpha: false});
  const fail = html => { $('glfb').style.display = 'flex'; $('glfb').firstElementChild.innerHTML = html; return STUB; };
  if (!gl) return fail('<b>WebGL2 is not available</b><br><br>The 3D view needs WebGL2 (any current browser). The other tabs work.');
    const VERT_SRC = `#version 300 es
  layout(location=0) in vec3 aPos;
  uniform mat4 uView;
  uniform mat4 uProj;
  uniform float uHalfExtent;
  out vec3 vWorldPos;
  void main(){
    vWorldPos = aPos * uHalfExtent;
    gl_Position = uProj * uView * vec4(vWorldPos, 1.0);
  }
  `;

    const FRAG_SRC = `#version 300 es
  precision highp float;
  precision highp sampler3D;

  in vec3 vWorldPos;
  uniform vec3 uCamPos;
  uniform float uHalfExtent;
  uniform sampler3D uTex;
  uniform int uSteps;
  uniform float uThreshold;
  uniform float uGain;
  uniform float uOpacity;
  uniform int uMode;   // 0 density 1 re 2 im 3 phase
  uniform int uCmap;   // 0 hwave 1 viridis 2 plasma 3 hot 4 electric
  uniform int uScale;  // 0 linear |psi|^2, 1 amplitude |psi|, 2 logarithmic
  uniform int uCut;    // 1: remove the half facing the camera
  uniform vec2 uCutDir;
  out vec4 fragColor;

  const float PI = 3.14159265359;

  // |psi|^2 of a Rydberg state is hugely concentrated near the nucleus, so a
  // linear map shows only a tiny core. Compress it for display only.
  float dscale(float d){
    if(uScale==1) return sqrt(max(d,0.0));
    if(uScale==2) return clamp((log(max(d,1e-6))/2.302585 + 6.0)/6.0, 0.0, 1.0);
    return d;
  }

  bool intersectBox(vec3 ro, vec3 rd, float he, out float t0, out float t1){
    vec3 invD = 1.0/rd;
    vec3 tlo = (vec3(-he)-ro)*invD;
    vec3 thi = (vec3( he)-ro)*invD;
    vec3 tmn = min(tlo,thi);
    vec3 tmx = max(tlo,thi);
    t0 = max(max(tmn.x,tmn.y),tmn.z);
    t1 = min(min(tmx.x,tmx.y),tmx.z);
    return t1 > max(t0,0.0);
  }

  vec3 hwave(float t){
    t=clamp(t,0.0,1.0);
    vec3 c0=vec3(0.086,0.043,0.224);
    vec3 c1=vec3(0.271,0.039,0.412);
    vec3 c2=vec3(0.459,0.106,0.427);
    vec3 c3=vec3(0.643,0.173,0.376);
    vec3 c4=vec3(0.800,0.255,0.282);
    vec3 c5=vec3(0.922,0.396,0.153);
    vec3 c6=vec3(0.984,0.600,0.024);
    vec3 c7=vec3(0.965,0.827,0.247);
    vec3 c8=vec3(1.0,1.0,1.0);
    if(t<0.12) return mix(c0,c1,t/0.12);
    if(t<0.25) return mix(c1,c2,(t-0.12)/0.13);
    if(t<0.38) return mix(c2,c3,(t-0.25)/0.13);
    if(t<0.50) return mix(c3,c4,(t-0.38)/0.12);
    if(t<0.62) return mix(c4,c5,(t-0.50)/0.12);
    if(t<0.75) return mix(c5,c6,(t-0.62)/0.13);
    if(t<0.88) return mix(c6,c7,(t-0.75)/0.13);
    return mix(c7,c8,(t-0.88)/0.12);
  }
  vec3 viridisA(float t){
    t=clamp(t,0.0,1.0);
    vec3 c0=vec3(0.267,0.005,0.329);
    vec3 c1=vec3(0.255,0.267,0.529);
    vec3 c2=vec3(0.165,0.471,0.558);
    vec3 c3=vec3(0.135,0.659,0.518);
    vec3 c4=vec3(0.478,0.821,0.318);
    vec3 c5=vec3(0.992,0.906,0.144);
    if(t<0.2) return mix(c0,c1,t/0.2);
    if(t<0.4) return mix(c1,c2,(t-0.2)/0.2);
    if(t<0.6) return mix(c2,c3,(t-0.4)/0.2);
    if(t<0.8) return mix(c3,c4,(t-0.6)/0.2);
    return mix(c4,c5,(t-0.8)/0.2);
  }
  vec3 plasmaA(float t){
    t=clamp(t,0.0,1.0);
    vec3 c0=vec3(0.051,0.031,0.529);
    vec3 c1=vec3(0.416,0.000,0.659);
    vec3 c2=vec3(0.694,0.165,0.565);
    vec3 c3=vec3(0.882,0.392,0.384);
    vec3 c4=vec3(0.988,0.652,0.212);
    vec3 c5=vec3(0.941,0.976,0.129);
    if(t<0.2) return mix(c0,c1,t/0.2);
    if(t<0.4) return mix(c1,c2,(t-0.2)/0.2);
    if(t<0.6) return mix(c2,c3,(t-0.4)/0.2);
    if(t<0.8) return mix(c3,c4,(t-0.6)/0.2);
    return mix(c4,c5,(t-0.8)/0.2);
  }
  vec3 hotA(float t){
    t=clamp(t,0.0,1.0);
    vec3 c0=vec3(0.0,0.0,0.0);
    vec3 c1=vec3(1.0,0.0,0.0);
    vec3 c2=vec3(1.0,1.0,0.0);
    vec3 c3=vec3(1.0,1.0,1.0);
    if(t<0.4) return mix(c0,c1,t/0.4);
    if(t<0.75) return mix(c1,c2,(t-0.4)/0.35);
    return mix(c2,c3,(t-0.75)/0.25);
  }
  vec3 electricA(float t){
    t=clamp(t,0.0,1.0);
    vec3 c0=vec3(0.0,0.122,0.247);
    vec3 c1=vec3(0.0,0.2,0.4);
    vec3 c2=vec3(0.106,0.435,0.478);
    vec3 c3=vec3(0.322,0.718,0.533);
    vec3 c4=vec3(0.851,0.929,0.573);
    if(t<0.25) return mix(c0,c1,t/0.25);
    if(t<0.55) return mix(c1,c2,(t-0.25)/0.3);
    if(t<0.8) return mix(c2,c3,(t-0.55)/0.25);
    return mix(c3,c4,(t-0.8)/0.2);
  }
  vec3 rdbuA(float s){
    float t = clamp(s*0.5+0.5, 0.0, 1.0);
    vec3 c0=vec3(0.130,0.400,0.675);
    vec3 c1=vec3(0.404,0.663,0.812);
    vec3 c2=vec3(0.969,0.969,0.969);
    vec3 c3=vec3(0.957,0.647,0.510);
    vec3 c4=vec3(0.698,0.094,0.169);
    if(t<0.25) return mix(c0,c1,t/0.25);
    if(t<0.5) return mix(c1,c2,(t-0.25)/0.25);
    if(t<0.75) return mix(c2,c3,(t-0.5)/0.25);
    return mix(c3,c4,(t-0.75)/0.25);
  }
  vec3 colormapDensity(float t){
    if(uCmap==0) return hwave(t);
    if(uCmap==1) return viridisA(t);
    if(uCmap==2) return plasmaA(t);
    if(uCmap==3) return hotA(t);
    return electricA(t);
  }
  vec3 hsv2rgb(vec3 c){
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
  }

  void main(){
    vec3 rd = normalize(vWorldPos - uCamPos);
    vec3 ro = uCamPos;
    float t0, t1;
    if(!intersectBox(ro, rd, uHalfExtent, t0, t1)) discard;
    t0 = max(t0, 0.0);
    float segLen = max(t1 - t0, 0.0);
    int nSteps = uSteps;
    float dt = segLen / float(nSteps);
    vec3 pos = ro + rd * t0;
    vec3 stp = rd * dt;
    vec4 accum = vec4(0.0);
    float stepScale = 90.0 / float(nSteps);

    for(int i=0; i<400; i++){
      if(i >= nSteps) break;
      if(accum.a > 0.98) break;
      if(uCut == 1 && dot(pos.xy, uCutDir) > 0.0){ pos += stp; continue; }
      vec3 local = pos / uHalfExtent;
      // worker fills idx = (ix*N + iy)*N + iz  =>  texture s = z, t = y, r = x
      vec3 uvw = vec3(local.z, local.y, local.x) * 0.5 + 0.5;
      vec4 samp = texture(uTex, uvw);
      float density = dscale(samp.r);
      float rawDensity = samp.r;
      float re = samp.g;
      float im = samp.b;

      vec3 col; float a;
      if(uMode == 0){
        float f = clamp((density - uThreshold) / max(1.0 - uThreshold, 0.001), 0.0, 1.0);
        a = pow(f, uGain) * uOpacity * stepScale;
        col = colormapDensity(density);
      } else if(uMode == 1){
        float f = clamp((abs(re) - uThreshold) / max(1.0 - uThreshold, 0.001), 0.0, 1.0);
        a = pow(f, uGain) * uOpacity * stepScale;
        col = rdbuA(re);
      } else if(uMode == 2){
        float f = clamp((abs(im) - uThreshold) / max(1.0 - uThreshold, 0.001), 0.0, 1.0);
        a = pow(f, uGain) * uOpacity * stepScale;
        col = rdbuA(im);
      } else {
        float f = clamp((density - uThreshold) / max(1.0 - uThreshold, 0.001), 0.0, 1.0);
        a = pow(f, uGain) * uOpacity * stepScale;
        float hue = atan(im, re) / (2.0*PI) + 0.5;
        float val = clamp(sqrt(rawDensity) * 1.15, 0.0, 1.0);
        val = max(val, 0.35 * smoothstep(uThreshold, 1.0, density));
        col = hsv2rgb(vec3(hue, 0.85, val));
      }
      a = clamp(a, 0.0, 1.0);
      accum.rgb += (1.0 - accum.a) * col * a;
      accum.a += (1.0 - accum.a) * a;
      pos += stp;
    }
    if(accum.a < 0.004) discard;
    fragColor = vec4(accum.rgb, accum.a);
  }
  `;
  function compileShader(src, type) {
    const sh = gl.createShader(type); gl.shaderSource(sh, src); gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(sh));
    return sh;
  }
  let prog;
  try {
    prog = gl.createProgram();
    gl.attachShader(prog, compileShader(VERT_SRC, gl.VERTEX_SHADER)); gl.attachShader(prog, compileShader(FRAG_SRC, gl.FRAGMENT_SHADER));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
  } catch (err) { return fail('<b>Shader error</b><br><br>' + esc(String(err))); }
  const U = n => gl.getUniformLocation(prog, n);
  const L = {};
  ['uView', 'uProj', 'uHalfExtent', 'uCamPos', 'uSteps', 'uThreshold', 'uGain', 'uOpacity', 'uMode', 'uCmap', 'uTex', 'uScale', 'uCut', 'uCutDir'].forEach(n => { L[n] = U(n); });
  const vao = gl.createVertexArray(); gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1]), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array([4, 5, 6, 4, 6, 7, 0, 2, 1, 0, 3, 2, 0, 1, 5, 0, 5, 4, 3, 6, 2, 3, 7, 6, 0, 7, 3, 0, 4, 7, 1, 2, 6, 1, 6, 5]), gl.STATIC_DRAW);
  gl.bindVertexArray(null);
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_3D, tex);
  [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER].forEach(p => gl.texParameteri(gl.TEXTURE_3D, p, gl.LINEAR));
  [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T, gl.TEXTURE_WRAP_R].forEach(p => gl.texParameteri(gl.TEXTURE_3D, p, gl.CLAMP_TO_EDGE));
  gl.texImage3D(gl.TEXTURE_3D, 0, gl.RGBA16F, 1, 1, 1, 0, gl.RGBA, gl.FLOAT, new Float32Array(4));

  // orbit camera, world in a0, z up (quantisation axis)
  const cam = {theta: 0.8, phi: 1.15, radius: 16};
  let drag = false, lx = 0, ly = 0;
  canvas.addEventListener('pointerdown', e => { drag = true; lx = e.clientX; ly = e.clientY; canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener('pointerup', () => { drag = false; });
  canvas.addEventListener('pointercancel', () => { drag = false; });
  canvas.addEventListener('pointermove', e => {
    if (!drag) return;
    cam.theta -= (e.clientX - lx) * 0.007; cam.phi = Math.min(Math.max(cam.phi - (e.clientY - ly) * 0.007, 0.06), Math.PI - 0.06);
    lx = e.clientX; ly = e.clientY;
  });
  canvas.addEventListener('wheel', e => { e.preventDefault(); cam.radius = Math.min(Math.max(cam.radius * Math.exp(Math.sign(e.deltaY) * 0.09), half * 0.6), half * 12); }, {passive: false});
  const camPos = () => [cam.radius * Math.sin(cam.phi) * Math.cos(cam.theta), cam.radius * Math.sin(cam.phi) * Math.sin(cam.theta), cam.radius * Math.cos(cam.phi)];
  function persp(fovy, aspect, near, far) { const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far), o = new Float32Array(16); o[0] = f / aspect; o[5] = f; o[10] = (far + near) * nf; o[11] = -1; o[14] = 2 * far * near * nf; return o; }
  function lookAt(eye, up) {
    let z0 = eye[0], z1 = eye[1], z2 = eye[2], len = Math.hypot(z0, z1, z2) || 1; z0 /= len; z1 /= len; z2 /= len;
    let x0 = up[1] * z2 - up[2] * z1, x1 = up[2] * z0 - up[0] * z2, x2 = up[0] * z1 - up[1] * z0; len = Math.hypot(x0, x1, x2);
    if (len > 1e-6) { x0 /= len; x1 /= len; x2 /= len; } else { x0 = 1; x1 = 0; x2 = 0; }
    const y0 = z1 * x2 - z2 * x1, y1 = z2 * x0 - z0 * x2, y2 = z0 * x1 - z1 * x0, o = new Float32Array(16);
    o.set([x0, y0, z0, 0, x1, y1, z1, 0, x2, y2, z2, 0, -(x0 * eye[0] + x1 * eye[1] + x2 * eye[2]), -(y0 * eye[0] + y1 * eye[1] + y2 * eye[2]), -(z0 * eye[0] + z1 * eye[1] + z2 * eye[2]), 1]);
    return o;
  }
  function mul(a, b) { const o = new Float32Array(16); for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) { let s = 0; for (let k = 0; k < 4; k++) s += a[i + k * 4] * b[k + j * 4]; o[i + j * 4] = s; } return o; }
  function project(p, m, W, H) { const x = m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12], y = m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13], w = m[3] * p[0] + m[7] * p[1] + m[11] * p[2] + m[15]; return w <= 1e-6 ? null : [(x / w * .5 + .5) * W, (1 - (y / w * .5 + .5)) * H]; }
  const niceStep = x => { const p = Math.pow(10, Math.floor(Math.log10(x))), f = x / p; return (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * p; };
  function drawOverlay(view, proj) {
    const W = ov.width, H = ov.height; ctx.clearRect(0, 0, W, H);
    const ax = $('axes'); if (!hasData || (ax && !ax.checked)) return;
    const k = W / Math.max(canvas.clientWidth, 1), mvp = mul(proj, view), he = half, P = p => project(p, mvp, W, H);
    const line = (a, b, st, lw) => { const A = P(a), B = P(b); if (!A || !B) return; ctx.strokeStyle = st; ctx.lineWidth = lw * k; ctx.beginPath(); ctx.moveTo(A[0], A[1]); ctx.lineTo(B[0], B[1]); ctx.stroke(); };
    const c = [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]].map(v => v.map(a => a * he));
    [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]].forEach(e => line(c[e[0]], c[e[1]], 'rgba(150,165,210,0.20)', 1));
    const f = uf(), stepD = niceStep(he * f / 3), step = stepD / f;
    ctx.font = (11 * k) + 'px system-ui, sans-serif';
    [['x', [1, 0, 0], '232,112,122'], ['y', [0, 1, 0], '111,207,143'], ['z', [0, 0, 1], '111,168,255']].forEach(([name, d, rgb]) => {
      line(d.map(a => -a * he), d.map(a => a * he), 'rgba(' + rgb + ',0.55)', 1.4);
      for (let t = step, j = 1; t <= he * 1.0001; t += step, j++) {
        const q = P(d.map(a => a * t)); if (!q) continue;
        ctx.fillStyle = 'rgba(' + rgb + ',0.9)'; ctx.beginPath(); ctx.arc(q[0], q[1], 2.2 * k, 0, 6.2832); ctx.fill();
        ctx.fillText((+(j * stepD).toPrecision(3)) + ' ' + un(), q[0] + 5 * k, q[1] - 4 * k);
      }
      const e = P(d.map(a => a * he * 1.06));
      if (e) { ctx.fillStyle = 'rgba(' + rgb + ',1)'; ctx.font = (13 * k) + 'px system-ui, sans-serif'; ctx.fillText(name === 'z' ? 'z (quantisation axis)' : name, e[0] + 4 * k, e[1]); ctx.font = (11 * k) + 'px system-ui, sans-serif'; }
    });
  }
  const lgc = $('lg-bar').getContext('2d'); let lgKey = '';
  const fmt = x => x >= 0.01 ? x.toFixed(2) : x.toExponential(0);
  function updateLegend(thr) {
    const el = $('legend'); if (!hasData) { el.style.display = 'none'; return; }
    el.style.display = 'block';
    const mode = S.chan, key = [mode, S.scale, S.cmap, thr, S.norm, S.i].join('|'); if (key === lgKey) return; lgKey = key;
    for (let x = 0; x < 240; x++) {
      const u = x / 239;
      const c = mode === 0 ? stopColor(CMAPN[S.cmap], thr + (1 - thr) * u) : mode === 3 ? hsv2rgbJS(u, .85, 1) : stopColor('rdbu', u);
      lgc.fillStyle = 'rgb(' + c.map(v => Math.round(v * 255)).join(',') + ')'; lgc.fillRect(x, 0, 1, 12);
    }
    const T = $('lg-title'), lo = $('lg-lo'), mid = $('lg-mid'), hi = $('lg-hi');
    if (mode === 0) { const q = t => S.scale === 0 ? t : S.scale === 1 ? t * t : Math.pow(10, 6 * (t - 1)); T.textContent = '|ψ|² / ' + maxTxt() + ' (' + ['linear', 'amplitude |ψ|', 'log'][S.scale] + ' scale)'; lo.textContent = fmt(q(thr)); mid.textContent = ''; hi.textContent = '1'; }
    else if (mode === 3) { T.textContent = 'arg ψ (brightness ∝ |ψ|)'; lo.textContent = '−π'; mid.textContent = '0'; hi.textContent = '+π'; }
    else { T.textContent = (mode === 1 ? 'Re ψ' : 'Im ψ') + ' / |ψ|' + maxTxt() + ' (hidden below ' + thr.toFixed(2) + ')'; lo.textContent = '−1'; mid.textContent = '0'; hi.textContent = '+1'; }
  }
  // draw only when something changed (camera, data, controls or size): no idle GPU work
  let lastKey = '';
  function frame() {
    requestAnimationFrame(frame);
    if (S.tab !== '3d' || !canvas.clientWidth) return;
    const key = [hasData, half, cam.theta, cam.phi, cam.radius, canvas.clientWidth, canvas.clientHeight, window.devicePixelRatio, S.chan, S.cmap, S.scale, S.unit, S.norm, S.i, S.mk, S.basis, lgKey,
      ...['thr-sl', 'steps-sl', 'gain-sl', 'op-sl'].map(id => sliderVal(id, 0)), $('cut') && $('cut').checked, $('axes') && $('axes').checked].join('|');
    if (key === lastKey) return;
    lastKey = key;
    render();
  }
  function render() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2), w = Math.max(1, Math.floor(canvas.clientWidth * dpr)), h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; ov.width = w; ov.height = h; gl.viewport(0, 0, w, h); }
    gl.clearColor(0.02, 0.024, 0.05, 1); gl.clear(gl.COLOR_BUFFER_BIT);
    const thr = sliderVal('thr-sl', 0.04);
    updateLegend(thr);
    if (!hasData) { ctx.clearRect(0, 0, ov.width, ov.height); return; }
    const eye = camPos(), view = lookAt(eye, [0, 0, 1]), proj = persp(45 * Math.PI / 180, w / h, Math.max(half * 0.02, 1e-3), half * 30 + 10);
    gl.useProgram(prog); gl.bindVertexArray(vao);
    gl.uniformMatrix4fv(L.uView, false, view); gl.uniformMatrix4fv(L.uProj, false, proj);
    gl.uniform1f(L.uHalfExtent, half); gl.uniform3f(L.uCamPos, eye[0], eye[1], eye[2]);
    gl.uniform1i(L.uSteps, sliderVal('steps-sl', 128)); gl.uniform1f(L.uThreshold, thr);
    gl.uniform1f(L.uGain, sliderVal('gain-sl', 1.6)); gl.uniform1f(L.uOpacity, sliderVal('op-sl', 1.4));
    gl.uniform1i(L.uMode, S.chan); gl.uniform1i(L.uCmap, S.cmap); gl.uniform1i(L.uScale, S.scale);
    const cut = $('cut') && $('cut').checked; gl.uniform1i(L.uCut, cut ? 1 : 0); gl.uniform2f(L.uCutDir, Math.cos(cam.theta), Math.sin(cam.theta));
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_3D, tex); gl.uniform1i(L.uTex, 0);
    gl.disable(gl.DEPTH_TEST); gl.enable(gl.CULL_FACE); gl.cullFace(gl.FRONT); gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.drawElements(gl.TRIANGLES, 36, gl.UNSIGNED_SHORT, 0);
    drawOverlay(view, proj);
  }
  requestAnimationFrame(frame);
  return {
    load(d, fresh) {
      gl.bindTexture(gl.TEXTURE_3D, tex); gl.texImage3D(gl.TEXTURE_3D, 0, gl.RGBA16F, d.N, d.N, d.N, 0, gl.RGBA, gl.FLOAT, d.data);
      half = d.half; if (fresh || freshState) { cam.radius = half * 3.3; cam.theta = 0.8; cam.phi = 1.15; freshState = false; }
      hasData = true; loadedKind = d.kind; lgKey = ''; lastKey = '';
    },
    clear() { hasData = false; lastKey = ''; },
    resetView() { cam.theta = 0.8; cam.phi = 1.15; cam.radius = half * 3.3; },
    png() {
      render();      // without preserveDrawingBuffer the frame must be read in the same task it is drawn
      const c = document.createElement('canvas'); c.width = canvas.width; c.height = canvas.height;
      const g = c.getContext('2d'); g.drawImage(canvas, 0, 0); g.drawImage(ov, 0, 0);
      g.fillStyle = '#e6e8f0'; g.font = Math.round(16 * c.width / Math.max(canvas.clientWidth, 1)) + 'px system-ui, sans-serif';
      g.fillText($('vptitle').textContent, 16, 28 * c.width / Math.max(canvas.clientWidth, 1));
      return c.toDataURL('image/png');
    },
  };
}

// ═════════════════════════ 3D: Plotly isosurfaces ═════════════════════════
function makeIsoRenderer() {
  const STUB = {load() {}, clear() {}, resetView() {}, png() { return null; }};
  const div = $('isodiv');
  $('glcanvas').style.display = 'none'; $('overlay').style.display = 'none'; div.style.display = 'block';
  if (typeof Plotly === 'undefined') { $('glfb').style.display = 'flex'; $('glfb').firstElementChild.innerHTML = '<b>plotly.min.js could not be loaded</b><br><br>Keep it next to this page, or re-run with --render volume.'; return STUB; }
  const EYE0 = {x: 1.5, y: 1.5, z: 1.0}, UP = {x: 0, y: 0, z: 1};
  let cache = null, lastSurfs = -1, chain = Promise.resolve(), plotted = false;
  const enqueue = f => { chain = chain.then(f).catch(e => console.error(e)); return chain; };
  const ax = t => ({title: {text: t, font: {color: '#aab', size: 11}}, gridcolor: '#2a2a3a', zerolinecolor: '#333', backgroundcolor: '#0a0a16', showbackground: true, tickfont: {color: '#aab', size: 10}});
  function draw() {
    if (!cache || S.tab !== '3d') return;
    const c = cache, f = uf(), amp = sliderVal('lscale', 0) === 1;
    const iso = sliderVal('iso-sl', 0.04), hi = Math.max(sliderVal('isohi-sl', 0.95), iso + 0.02), surfs = Math.round(sliderVal('surf-sl', 4)), opv = sliderVal('op-sl', 0.55);
    const val = amp ? c.dens.map(Math.sqrt) : c.dens;
    const x = c.x.map(v => v * f), y = c.y.map(v => v * f), z = c.z.map(v => v * f), h = c.half * f;
    const trace = {type: 'isosurface', x, y, z, value: val, isomin: iso, isomax: hi, surface: {count: surfs}, colorscale: plotlyScale(CMAPN[S.cmap]),
      opacity: opv, showscale: true, caps: {x: {show: false}, y: {show: false}, z: {show: false}}, hoverinfo: 'skip',
      colorbar: {x: 0.99, xanchor: 'right', len: 0.6, thickness: 14, title: {text: (amp ? '|ψ| / ' : '|ψ|² / ') + maxTxt(), side: 'right', font: {color: '#c3c8d8'}}, tickfont: {color: '#aab', size: 10}}};
    const lay = {paper_bgcolor: '#05060c', font: {family: css('--sans'), color: '#ddd'}, margin: {l: 0, r: 0, t: 0, b: 0}, uirevision: 'keep',
      scene: {bgcolor: '#05060c', aspectmode: 'cube', xaxis: ax('x (' + un() + ')'), yaxis: ax('y (' + un() + ')'), zaxis: ax('z (' + un() + '), quantisation axis'), camera: {eye: EYE0, up: UP}}};
    ['xaxis', 'yaxis', 'zaxis'].forEach(k => { lay.scene[k].range = [-h, h]; });
    enqueue(() => {
      if (!plotted || surfs !== lastSurfs) { lastSurfs = surfs; plotted = true; return Plotly.react(div, [trace], lay, {responsive: true, displaylogo: false}); }
      return Plotly.react(div, [trace], lay, {responsive: true, displaylogo: false});
    });
  }
  ['lscale', 'iso-sl', 'isohi-sl', 'surf-sl', 'op-sl'].forEach(id => { const e = $(id); if (e) e.addEventListener('change', draw); });
  return {
    load(d) {
      const N = d.N, tot = N * N * N, h = d.half, co = new Float32Array(N);
      for (let i = 0; i < N; i++) co[i] = -h + 2 * h * i / (N - 1);
      const x = new Float32Array(tot), y = new Float32Array(tot), z = new Float32Array(tot), dens = new Float32Array(tot);
      let idx = 0;
      for (let ix = 0; ix < N; ix++) for (let iy = 0; iy < N; iy++) for (let iz = 0; iz < N; iz++, idx++) { x[idx] = co[ix]; y[idx] = co[iy]; z[idx] = co[iz]; dens[idx] = d.data[idx * 4]; }
      cache = {x, y, z, dens, half: h}; draw();
    },
    redraw: draw,
    clear() { cache = null; },
    resetView() { enqueue(() => Plotly.relayout(div, {'scene.camera': {eye: EYE0, up: UP}})); },
    png() { return null; },
  };
}

// ═════════════════════════ 3D data (worker) ═════════════════════════
const WORKER_SRC = $('orbcore').textContent + `
self.onmessage = function (e) {
  const p = e.data, R = makeRadial(p.u, p.x0, p.dx, p.q.l), h = p.half, N = p.N;
  const co = new Float64Array(N);
  for (let i = 0; i < N; i++) co[i] = -h + 2 * h * i / (N - 1);
  const tot = N * N * N, data = new Float32Array(tot * 4);
  let idx = 0, mx = 0, mxAll = 0;
  const rn2 = p.rn * p.rn;
  for (let ix = 0; ix < N; ix++) for (let iy = 0; iy < N; iy++) for (let iz = 0; iz < N; iz++, idx++) {
    const v = psiAt(p.q, R, co[ix], co[iy], co[iz]), b = idx * 4;
    data[b] = v[0]; data[b + 1] = v[1]; data[b + 2] = v[2]; data[b + 3] = 1;
    if (v[0] > mxAll) mxAll = v[0];
    if (v[0] > mx && co[ix] * co[ix] + co[iy] * co[iy] + co[iz] * co[iz] > rn2) mx = v[0];
  }
  if (!(mx > 0)) mx = mxAll;
  // half floats hold up to 65504: the saturated core is clipped well below that
  if (mx > 0) { const iD = 1 / mx, iA = 1 / Math.sqrt(mx); for (let i = 0; i < tot; i++) { const b = i * 4; data[b] = Math.min(data[b] * iD, 1e4); data[b + 1] = Math.max(Math.min(data[b + 1] * iA, 100), -100); data[b + 2] = Math.max(Math.min(data[b + 2] * iA, 100), -100); } }
  self.postMessage({data, N, half: h}, [data.buffer]);
};`;
const WORKER_URL = URL.createObjectURL(new Blob([WORKER_SRC], {type: 'application/javascript'}));
let REND = null, worker = null, loadedKey = '', lastSel = -1;
const gridN = s => Math.min(D.grid + 4 * s.lint, D.maxn);
const boxHalf = s => s.rbox * 1.05;
function ensure3d() {
  if (!REND) REND = RENDER === 'isosurface' ? makeIsoRenderer() : makeVolumeRenderer();
  const s = cur(), q = query(), key = S.i + '|' + normR(s) + '|' + JSON.stringify(q);
  if (key === loadedKey) { if (REND.redraw) REND.redraw(); return; }
  if (worker) worker.terminate();
  $('vpmsg').textContent = 'Computing…';
  worker = new Worker(WORKER_URL);
  const fresh = S.i !== lastSel;
  worker.onmessage = e => {
    worker.terminate(); worker = null; loadedKey = key; lastSel = S.i; $('vpmsg').textContent = '';
    REND.load({data: e.data.data, N: e.data.N, half: e.data.half, kind: q.kind}, fresh);
  };
  worker.onerror = err => { worker = null; $('vpmsg').textContent = 'Worker error: ' + err.message; };
  worker.postMessage({u: s.uu, x0: s.x0, dx: s.dx, q, half: boxHalf(s), N: gridN(s), rn: normR(s)});
}

// ═════════════════════════ Plotly helpers ═════════════════════════
function base() {
  const ink = css('--ink'), ink2 = css('--ink2'), grid = css('--grid'), bg = css('--bg');
  return {ink, ink2, grid, bg, lay: {paper_bgcolor: bg, plot_bgcolor: bg, font: {family: css('--sans'), color: ink}, hovermode: 'closest', margin: {l: 80, r: 30, t: 56, b: 56}, showlegend: true,
    legend: {orientation: 'h', x: 0, xanchor: 'left', y: 1.02, yanchor: 'bottom', font: {size: 12, color: ink2}},
    hoverlabel: {bgcolor: css('--panel'), bordercolor: css('--edge'), font: {color: ink, family: css('--sans'), size: 12}}}};
}
const cfg = {displaylogo: false, responsive: true, modeBarButtonsToRemove: ['lasso2d', 'select2d']};
const axs = (b, t, extra) => Object.assign({title: {text: t, font: {color: b.ink2, size: 12}}, gridcolor: b.grid, linecolor: b.ink2, tickfont: {color: b.ink2, size: 11}, ticks: 'outside', tickcolor: b.ink2, zerolinecolor: b.grid, exponentformat: 'power'}, extra || {});

// ═════════════════════════ Slice ═════════════════════════
function buildSl() {
  const b = base(), s = cur(), q = query(), f = uf(), h = boxHalf(s), N = 241, co = [];
  for (let i = 0; i < N; i++) co.push(-h + 2 * h * i / (N - 1));
  const V = [], Re = [], Im = [], rn2 = normR(s) ** 2;
  let mx = 0, mxAll = 0;
  for (let j = 0; j < N; j++) {
    const rv = [], rr = [], ri = [];
    for (let i = 0; i < N; i++) {
      const v = S.plane === 'xz' ? psiAt(q, s.R, co[i], 0, co[j]) : psiAt(q, s.R, co[i], co[j], 0);
      rv.push(v[0]); rr.push(v[1]); ri.push(v[2]);
      if (v[0] > mxAll) mxAll = v[0];
      if (v[0] > mx && co[i] * co[i] + co[j] * co[j] > rn2) mx = v[0];
    }
    V.push(rv); Re.push(rr); Im.push(ri);
  }
  mx = mx || mxAll || 1;
  const xs = co.map(v => v * f), tr = [];
  const hov = (lab) => `x = %{x:.4g} ${un()}<br>${S.plane === 'xz' ? 'z' : 'y'} = %{y:.4g} ${un()}<br>${lab} = %{z:.4g}<extra></extra>`;
  let title;
  if (S.chan === 0) {
    const Z = V.map(r => r.map(v => { const t = v / mx; return S.scale === 0 ? t : S.scale === 1 ? Math.sqrt(t) : Math.max(Math.log10(Math.max(t, 1e-12)), -6); }));
    title = ['|ψ|² / ' + maxLab(), '|ψ| / ' + maxLab(), 'log₁₀(|ψ|² / ' + maxLab() + ')'][S.scale];
    tr.push({type: 'heatmap', x: xs, y: xs, z: Z, colorscale: plotlyScale(CMAPN[S.cmap]), zmin: S.scale === 2 ? -6 : 0, zmax: S.scale === 2 ? 0 : 1, zsmooth: 'best',
      colorbar: {title: {text: title, side: 'right', font: {color: b.ink2}}, tickfont: {color: b.ink2}}, hovertemplate: hov(title), showlegend: false});
  } else if (S.chan === 1 || S.chan === 2) {
    const A = (S.chan === 1 ? Re : Im), am = Math.sqrt(mx), Z = A.map(r => r.map(v => v / am));
    title = (S.chan === 1 ? 'Re ψ' : 'Im ψ') + ' / |ψ|' + maxLab();
    tr.push({type: 'heatmap', x: xs, y: xs, z: Z, colorscale: plotlyScale('rdbu'), zmin: -1, zmax: 1, zsmooth: 'best',
      colorbar: {title: {text: title, side: 'right', font: {color: b.ink2}}, tickfont: {color: b.ink2}}, hovertemplate: hov(title), showlegend: false});
    tr.push({type: 'contour', x: xs, y: xs, z: Z, showscale: false, contours: {start: 0, end: 0, size: 1, coloring: 'lines'}, colorscale: [[0, b.ink], [1, b.ink]], line: {color: b.ink, width: 1}, hoverinfo: 'skip', showlegend: false, name: 'nodal lines'});
  } else {
    const Z = V.map((r, j) => r.map((v, i) => v / mx < 1e-3 ? null : Math.atan2(Im[j][i], Re[j][i])));
    title = 'arg ψ (rad)';
    tr.push({type: 'heatmap', x: xs, y: xs, z: Z, colorscale: HSV_SCALE, zmin: -Math.PI, zmax: Math.PI, zsmooth: false,
      colorbar: {title: {text: title, side: 'right', font: {color: b.ink2}}, tickvals: [-Math.PI, 0, Math.PI], ticktext: ['−π', '0', '+π'], tickfont: {color: b.ink2}}, hovertemplate: hov(title), showlegend: false});
  }
  const shapes = [];
  const circ = (r, color, dash) => shapes.push({type: 'circle', xref: 'x', yref: 'y', x0: -r * f, x1: r * f, y0: -r * f, y1: r * f, line: {color, width: 1.2, dash}});
  s.nodes.filter(r => r < h).forEach(r => circ(r, css('--ink2'), 'dot'));
  if (s.rc && s.rc < h) circ(s.rc, css('--soc'), 'dash');
  const vlab = S.plane === 'xz' ? 'z (' + un() + '), quantisation axis' : 'y (' + un() + ')';
  const lay = Object.assign({}, b.lay, {uirevision: 'sl' + S.i + S.plane + S.unit, showlegend: false, shapes, margin: {l: 80, r: 30, t: 30, b: 56},
    xaxis: axs(b, 'x (' + un() + ')', {range: [-h * f, h * f], constrain: 'domain'}), yaxis: axs(b, vlab, {range: [-h * f, h * f], scaleanchor: 'x', constrain: 'domain'})});
  Plotly.react('psl', tr, lay, cfg);
  $('slinfo').innerHTML = `${s.lblp} · ${subLabel()} · ${chanName()} · ${S.plane === 'xz' ? 'y = 0 plane' : 'z = 0 plane'}`;
}

// ═════════════════════════ Radial ═════════════════════════
function buildRad() {
  const b = base(), s = cur(), f = uf(), tr = [];
  const list = [S.i, ...S.pins.filter(k => k !== S.i)].slice(0, 5);
  let ymax = 0, rmin = Infinity, rmaxv = 0;
  list.forEach((k, n) => {
    const t = ST[k], col = css(PAL[n]);
    // on a log-r axis P is shown per unit ln r (r P), so equal areas are equal probabilities
    const lnP = S.rlog && S.rq === 'P';
    const x = Array.from(t.rr, r => r * f), y = Array.from(t.uu, (u, i) => S.rq === 'P' ? (lnP ? u * u * t.rr[i] : u * u / f) : u / Math.sqrt(f));
    y.forEach(v => { if (Math.abs(v) > ymax) ymax = Math.abs(v); });
    let wmax = 0; t.rr.forEach((r, i) => { wmax = Math.max(wmax, r * t.uu[i] ** 2); });
    const lo = t.rr.findIndex((r, i) => r * t.uu[i] ** 2 > 1e-4 * wmax); rmin = Math.min(rmin, t.rr[Math.max(lo, 0)] * f); rmaxv = Math.max(rmaxv, t.rbox * 1.2 * f);
    tr.push({x, y, mode: 'lines', name: t.lblp, line: {color: col, width: n ? 1.6 : 2.4}, hovertemplate: `<b>${t.lblp}</b><br>r = %{x:.4g} ${un()}<br>${S.rq === 'P' ? (lnP ? 'rP' : 'P') : 'u'} = %{y:.4g}<extra></extra>`});
    if (t.nodes.length) tr.push({x: t.nodes.map(r => r * f), y: t.nodes.map(() => 0), mode: 'markers', showlegend: false, marker: {symbol: 'line-ns-open', size: 12, color: col, line: {width: 2, color: col}}, hovertemplate: `${t.lblp} node<br>r = %{x:.4g} ${un()}<extra></extra>`});
  });
  // enclosed probability of the current state
  const cum = [0]; for (let i = 1; i < s.uu.length; i++) cum.push(cum[i - 1] + 0.5 * (s.uu[i] ** 2 * s.rr[i] + s.uu[i - 1] ** 2 * s.rr[i - 1]) * s.dx);
  const tot = cum[cum.length - 1] || 1;
  tr.push({x: Array.from(s.rr, r => r * f), y: cum.map(c => c / tot), yaxis: 'y2', mode: 'lines', name: 'enclosed probability, ' + s.lblp, line: {color: css('--muted'), width: 1.2, dash: 'dash'}, hovertemplate: 'r = %{x:.4g} ' + un() + '<br>∫₀ʳ P = %{y:.4f}<extra></extra>'});
  const y0 = S.rq === 'P' ? 0 : -ymax * 1.08, y1 = ymax * 1.08;
  if (s.rc) tr.push({x: [rmin, s.rc * f, s.rc * f, rmin], y: [y0, y0, y1, y1], fill: 'toself', mode: 'lines', line: {width: 0}, fillcolor: css('--soc'), opacity: 0.1, name: 'inside r<sub>c</sub>', hoverinfo: 'skip', showlegend: false});
  if (s.rc) tr.push({x: [s.rc * f, s.rc * f], y: [y0, y1], mode: 'lines+text', text: ['', 'r<sub>c</sub>'], textposition: 'top right', cliponaxis: false, textfont: {color: css('--soc'), size: 11}, line: {color: css('--soc'), width: 1, dash: 'dash'}, showlegend: false, hovertemplate: 'core radius r<sub>c</sub> = %{x:.4g} ' + un() + '<extra></extra>'});
  [[s.tin, 'r₋'], [s.tout, 'r₊']].forEach(([r, lab]) => {
    if (!r || r <= 0) return;
    tr.push({x: [r * f, r * f], y: [y0, y1], mode: 'lines+text', text: ['', lab], textposition: 'top right', cliponaxis: false, textfont: {color: css('--ink2'), size: 11}, line: {color: css('--ink2'), width: 1, dash: 'dot'}, showlegend: false, hovertemplate: 'turning point ' + lab + ' = %{x:.4g} ' + un() + '<extra></extra>'});
  });
  const ylab = S.rq === 'P' ? (S.rlog ? 'r P(r), probability per unit ln r' : `P(r) = u² (${un()}⁻¹)`) : `u(r) = rR (${un()}<sup>−1/2</sup>)`;
  const lay = Object.assign({}, b.lay, {uirevision: 'rad' + S.unit + S.rq + S.rlog,
    xaxis: axs(b, 'r (' + un() + ')', S.rlog ? {type: 'log', range: [Math.log10(rmin), Math.log10(rmaxv)]} : {range: [0, rmaxv]}),
    yaxis: axs(b, ylab, {range: [S.rq === 'P' ? 0 : -ymax * 1.08, ymax * 1.08]}),
    yaxis2: axs(b, 'Enclosed probability', {overlaying: 'y', side: 'right', range: [S.rq === 'P' ? 0 : -1.04, 1.04], tickvals: [0, 0.25, 0.5, 0.75, 1], showgrid: false, zeroline: false})});
  lay.margin = {l: 80, r: 70, t: 56, b: 56};
  Plotly.react('prad', tr, lay, cfg);
  $('radinfo').innerHTML = list.map((k, n) => `<span class="sw" style="background:${css(PAL[n])}"></span> ${ST[k].lblp}`).join(' &nbsp; ');
}

// ═════════════════════════ Values ═════════════════════════
function table(id, cols, rows) {
  const t = $(id), [k, dir] = S.sort, key = cols[k].s || (() => 0);
  rows = [...rows].sort((a, c) => { const x = key(a), y = key(c); return ((x == null) - (y == null)) || (x > y ? 1 : x < y ? -1 : 0) * dir; });
  t.tHead.innerHTML = '<tr>' + cols.map((c, j) => `<th data-k="${j}" class="${c.s ? '' : 'nosort'}">${c.h}${k === j ? '<span class="arr">' + (dir > 0 ? '▲' : '▼') + '</span>' : ''}</th>`).join('') + '</tr>';
  t.tBodies[0].innerHTML = rows.map(r => `<tr class="t${r.k === S.i ? ' sel' : ''}" data-i="${r.k}">` + cols.map(c => `<td class="${c.c === undefined ? 'm' : c.c}">${c.f(r)}</td>`).join('') + '</tr>').join('');
  t.tHead.querySelectorAll('th[data-k]').forEach(th => { const j = +th.dataset.k; if (!cols[j].s) return; th.onclick = () => { S.sort = [j, S.sort[0] === j ? -S.sort[1] : 1]; draw(); }; });
  t.tBodies[0].querySelectorAll('tr').forEach(tr => { tr.onclick = () => select(+tr.dataset.i); });
}
const hydR = s => (3 * s.nstar ** 2 - s.lint * (s.lint + 1)) / (2 * D.zcore);
function buildVal() {
  const f = uf(), L = n => (n == null ? '–' : sci(n * f, 4));
  const rows = ST.map((s, k) => ({s, k}));
  table('vtab', [
    {h: 'State', c: '', f: r => `<b>${r.s.lblp}</b>${r.s.gs ? '<span class="pm"> ground</span>' : ''}`, s: r => r.s.E},
    {h: 'E (eV)', f: r => fx(r.s.E, 5), s: r => r.s.E},
    {h: 'n*', f: r => fx(r.s.nstar, 4), s: r => r.s.nstar},
    {h: 'ν (radial)', f: r => fx(r.s.nu, 4), s: r => r.s.nu},
    {h: 'ν − n*', f: r => r.s.method === 'model' ? sci(r.s.nu - r.s.nstar, 2) : '–', s: r => r.s.nu - r.s.nstar},
    {h: 'Nodes', f: r => `${r.s.nodes.length}<span class="pm">/${r.s.n - r.s.lint - 1}</span>`, s: r => r.s.nodes.length},
    {h: `⟨r⟩ (${un()})`, f: r => L(r.s.r1), s: r => r.s.r1},
    {h: '⟨r⟩ / hydrogenic', f: r => fx(r.s.r1 / hydR(r.s), 3), s: r => r.s.r1 / hydR(r.s)},
    {h: `⟨r²⟩<sup>½</sup> (${un()})`, f: r => L(Math.sqrt(r.s.r2)), s: r => r.s.r2},
    {h: '⟨1/r⟩ (a₀⁻¹)', f: r => sci(r.s.rm1, 4), s: r => r.s.rm1},
    {h: '⟨1/r³⟩ (a₀⁻³)', f: r => sci(r.s.rm3, 4), s: r => r.s.rm3},
    {h: `r₊ (${un()})`, f: r => L(r.s.tout), s: r => r.s.tout},
    {h: `r<sub>c</sub> (${un()})`, f: r => r.s.rc ? L(r.s.rc) : '–', s: r => r.s.rc},
    {h: 'P(r < r<sub>c</sub>)', f: r => r.s.rc ? sci(r.s.pcore, 3) : '–', s: r => r.s.pcore},
    {h: `R<sub>LR</sub> pair (${un()})`, f: r => L(4 * Math.sqrt(r.s.r2)), s: r => r.s.r2},
    {h: 'Radial', c: '', f: r => `<span class="meth ${r.s.method}">${r.s.method}</span>`, s: r => r.s.method},
  ], rows);
}

// ═════════════════════════ Scaling ═════════════════════════
function buildSc() {
  const b = base(), f = uf(), tr = [];
  const ns = ST.map(s => s.nstar), n0 = Math.min(...ns) * 0.9, n1 = Math.max(...ns) * 1.1;
  LPRESENT.forEach(l => {
    const ss = ST.filter(s => s.lint === l), col = css(LCOL[l % LCOL.length]), name = LS[l] + ' series';
    tr.push({x: ss.map(s => s.nstar), y: ss.map(s => s.r1 * f), text: ss.map(s => s.lblp), mode: 'markers', name, legendgroup: name, marker: {color: col, size: 8},
      hovertemplate: `%{text}<br>n* = %{x:.4f}<br>⟨r⟩ = %{y:.4g} ${un()}<extra></extra>`});
    const xs = []; for (let i = 0; i <= 80; i++) { const n = n0 * Math.pow(n1 / n0, i / 80); if (n * n > l * (l + 1)) xs.push(n); }
    tr.push({x: xs, y: xs.map(n => (3 * n * n - l * (l + 1)) / (2 * D.zcore) * f), mode: 'lines', showlegend: false, legendgroup: name, line: {color: col, width: 1, dash: 'dot'}, hoverinfo: 'skip'});
    const pc = ss.filter(s => s.rc && s.pcore > 0);
    if (pc.length) {
      tr.push({x: pc.map(s => s.nstar), y: pc.map(s => s.pcore), text: pc.map(s => s.lblp), xaxis: 'x2', yaxis: 'y2', mode: 'markers', showlegend: false, legendgroup: name, marker: {color: col, size: 8},
        hovertemplate: '%{text}<br>n* = %{x:.4f}<br>P(r < r<sub>c</sub>) = %{y:.3g}<extra></extra>'});
      const a = pc.reduce((p, s) => s.nstar > p.nstar ? s : p), k = a.pcore * a.nstar ** 3;
      tr.push({x: [n0, n1], y: [k / n0 ** 3, k / n1 ** 3], xaxis: 'x2', yaxis: 'y2', mode: 'lines', showlegend: false, legendgroup: name, line: {color: col, width: 1, dash: 'dot'}, hoverinfo: 'skip'});
    }
  });
  const lay = Object.assign({}, b.lay, {uirevision: 'sc' + S.unit, grid: {rows: 1, columns: 2, pattern: 'independent', xgap: 0.12},
    xaxis: axs(b, 'n*', {type: 'log'}), yaxis: axs(b, `⟨r⟩ (${un()})`, {type: 'log'}),
    xaxis2: axs(b, 'n*', {type: 'log'}), yaxis2: axs(b, 'P(r < r<sub>c</sub>), probability inside the core', {type: 'log'})});
  Plotly.react('psc', tr, lay, cfg);
  $('scinfo').textContent = `${ST.length} states`;
}

// ═════════════════════════ Rail, chips, notes ═════════════════════════
function buildList() {
  $('lfilt').innerHTML = LPRESENT.map(l => `<button class="pill" data-l="${l}" aria-pressed="${S.filt.has(l)}">${LS[l]}</button>`).join('');
  $('lfilt').querySelectorAll('button').forEach(b => b.onclick = () => { const l = +b.dataset.l; if (S.filt.has(l)) { if (S.filt.size > 1) S.filt.delete(l); } else S.filt.add(l); buildList(); });
  $('slist').innerHTML = ST.map((s, k) => S.filt.has(s.lint) ? `<div class="srow${k === S.i ? ' on' : ''}" data-i="${k}" title="${esc(s.method)} radial function"><span>${s.lblp}${s.gs ? '<span class="gs">ground</span>' : ''}</span><span class="ns">n* ${fx(s.nstar, 3)}</span><span class="tg">${s.tau ? sci(s.tau, 3) + ' ns' : ''}</span></div>` : '').join('');
  $('slist').querySelectorAll('.srow').forEach(r => r.onclick = () => select(+r.dataset.i));
}
function buildSubs() {
  const s = cur(), l = s.lint;
  const pill = (txt, mk) => `<button class="pill" data-mk="${mk}" aria-pressed="${S.mk === mk}">${txt}</button>`;
  let h = '<div class="grp">Level</div><div class="pillrow">' + pill(s.jt > 0 ? 'Σ m<sub>j</sub>' : 'Σ m<sub>ℓ</sub>', 'sum') + pill('r²R²', 'sph') + '</div>';
  if (s.jt > 0) { h += '<div class="grp">j-coupled, spin-summed</div><div class="pillrow">'; for (let k = 1; k <= s.jt; k += 2) h += pill('|m<sub>j</sub>| = ' + k + '/2', 'mj:' + k); h += '</div>'; }
  if (l > 0) {
    h += '<div class="grp">Spin-free (LS) m<sub>ℓ</sub></div><div class="pillrow">'; for (let m = -l; m <= l; m++) h += pill((m < 0 ? '−' + (-m) : m), 'ml:' + m); h += '</div>';
    h += `<div class="seg" id="basis" role="group" aria-label="Basis" style="margin-top:6px"><button data-v="complex" aria-pressed="${S.basis === 'complex'}">Complex Y<sub>ℓ</sub><sup>m</sup></button><button data-v="real" aria-pressed="${S.basis === 'real'}">Real</button></div>`;
  }
  $('subs').innerHTML = h;
  $('subs').querySelectorAll('[data-mk]').forEach(b => b.onclick = () => { S.mk = b.dataset.mk; if (!chanOk(S.chan)) S.chan = 0; draw(); });
  $('subs').querySelectorAll('#basis button').forEach(b => b.onclick = () => { S.basis = b.dataset.v; if (!chanOk(S.chan)) S.chan = 0; draw(); });
  const q = query();
  $('subnote').innerHTML = q.kind === 'sum' ? (s.jt > 0 ? 'Averaged over m<sub>j</sub> and spin: spherical.' : 'Summed over m<sub>ℓ</sub>: spherical.')
    : q.kind === 'sph' ? 'Radial probability density; not |ψ|².'
    : q.kind === 'mj' ? 'Spin-orbit eigenstate |ℓ j m<sub>j</sub>⟩; two spinor components, so only the density is defined; ±m<sub>j</sub> give the same density.'
    : (S.basis === 'complex' ? 'L<sub>z</sub> eigenstate; the density is symmetric about z.' : 'Real combination of ±m.') + (s.jt > 0 ? ' Ignores spin–orbit coupling.' : '');
}
function syncRail() {
  const segs = [['chan', 'chan'], ['unit', 'unit'], ['plane', 'plane'], ['rq', 'rq'], ['norm', 'norm']];
  segs.forEach(([id, k]) => document.querySelectorAll('#' + id + ' button').forEach(b => b.setAttribute('aria-pressed', String(String(S[k]) === b.dataset.v))));
  document.querySelectorAll('#chan button').forEach(b => { b.disabled = !chanOk(+b.dataset.v); });
  $('dscale').value = S.scale; $('cmap').value = S.cmap; $('rlog').checked = S.rlog;
  $('pins').innerHTML = S.pins.map(k => `<span class="pin">${ST[k].lblp}<button data-k="${k}" aria-label="Remove">✕</button></span>`).join('');
  $('pins').querySelectorAll('button').forEach(b => b.onclick = () => { S.pins = S.pins.filter(k => k !== +b.dataset.k); draw(); });
  $('pinbtn').disabled = S.pins.includes(S.i) || S.pins.length >= 4;
}
function stats() {
  const s = cur(), f = uf();
  const k = (key, v, d) => `<div class="stat"><div class="k">${key}</div><div class="v">${v}</div>${d ? `<div class="d">${d}</div>` : ''}</div>`;
  $('stats').innerHTML = k('State', s.lblp, `E = ${fx(s.E, 5)} eV${s.tau ? ' · τ = ' + sci(s.tau, 4) + ' ns' : ''}`)
    + k('n* (level)', fx(s.nstar, 5), s.method === 'model' ? 'ν (radial) = ' + fx(s.nu, 5) : s.method)
    + k('⟨r⟩', sci(s.r1 * f, 4) + ' ' + un(), '⟨r²⟩<sup>½</sup> = ' + sci(Math.sqrt(s.r2) * f, 4) + ' ' + un())
    + k('Radial nodes', String(s.nodes.length), 'n − ℓ − 1 = ' + (s.n - s.lint - 1))
    + (s.rc ? k('Inside the core', sci(s.pcore, 3), 'P(r < r<sub>c</sub>), r<sub>c</sub> = ' + sci(s.rc * f, 3) + ' ' + un()) : '')
    + k('Radial function', `<span class="meth ${s.method}">${s.method}</span>`, '');
}
function methodNote(s) {
  if (s.method === 'model') return `Radial function: Marinescu model potential (ℓ-dependent core screening, core polarization, spin–orbit), solved for n − ℓ − 1 = ${s.n - s.lint - 1} nodes; ν = ${fx(s.nu, 5)} against the level's n* = ${fx(s.nstar, 5)}.`;
  if (s.method === 'hydrogenic') return 'Radial function: hydrogenic, exact.';
  return `Radial function: Coulomb (Whittaker) function at n* = ${fx(s.nstar, 4)}, exact outside r<sub>c</sub> = ${sci(s.rc * uf(), 3)} ${un()}; inside it u ∝ r<sup>ℓ+1</sup> (the core is not modelled).`;
}
function notes() {
  const s = cur(), f = uf(), h = boxHalf(s), N = gridN(s), vox = 2 * h / (N - 1);
  const coreLine = s.rc ? `Core radius r<sub>c</sub>: ${s.method === 'model' ? 'the sphere holding 99 % of the core electrons in this ℓ channel' : 'inner classical turning point (at least 1/Z a₀)'}.` : '';
  const finest = s.nodes.length ? Math.min(...s.nodes.map((r, i) => i ? r - s.nodes[i - 1] : r)) : Infinity;
  $('n3d').innerHTML = [
    `${s.lblp}, ${subLabel()}${RENDER === 'isosurface' ? ', isosurfaces of ' + (sliderVal('lscale', 0) === 1 ? '|ψ|' : '|ψ|²') : ', ' + chanName()}.`,
    (normR(s) ? 'Colour scale set by the maximum outside r<sub>c</sub>; the core peak is saturated.' : 'Colour scale set by the state\'s own maximum.') + ` The box holds 99.9 % of the probability (half-width ${sci(h * f, 3)} ${un()}).`,
    `Grid ${N}³, voxel ${sci(vox * f, 3)} ${un()}.` + (finest < vox ? ` Radial nodes closer than one voxel (the innermost at ${sci(s.nodes[0] * f, 3)} ${un()}) are not resolved here; the Slice and Radial tabs resolve them.` : ''),
    RENDER === 'volume' ? 'Cutaway removes the half facing the viewer.' : '',
    methodNote(s)].filter(Boolean).join('<br>');
  $('nsl').innerHTML = [
    `${S.plane === 'xz' ? 'x–z plane (y = 0), containing the quantisation axis z.' : 'x–y plane (z = 0), perpendicular to the quantisation axis.'}`,
    'Dotted circles: radial nodes (nodal spheres in every sub-state).',
    s.rc ? 'Dashed circle: core radius r<sub>c</sub>.' : '',
    S.chan === 1 || S.chan === 2 ? 'Black lines: nodal lines of the plotted component.' : '',
    normR(s) ? 'Colour scale set by the maximum outside r<sub>c</sub>; the core peak is saturated.' : '',
    S.chan === 3 ? 'Phase hidden where |ψ|² < 10⁻³ of the scale maximum.' : '',
    methodNote(s)].filter(Boolean).join('<br>');
  $('nrad').innerHTML = [
    S.rq === 'P' && S.rlog ? 'r P(r) on the log-r axis: probability per unit ln r, so equal areas hold equal probability (P = u², u = rR).' : (S.rq === 'P' ? 'P(r) = u(r)², u = rR; ∫P dr = 1.' : 'u(r) = rR(r), normalised to ∫u² dr = 1.'),
    'Ticks on the axis: radial nodes. r₋, r₊: classical turning points at n*.',
    s.rc ? 'Shaded: inside the core radius r<sub>c</sub>. ' + coreLine : '',
    'Dashed (right axis): probability enclosed within r, for the selected state.',
    methodNote(s)].filter(Boolean).join('<br>');
  $('nval').innerHTML = [
    'Expectation values from the radial function; lengths in ' + un() + ', ⟨1/r⟩ and ⟨1/r³⟩ in atomic units.',
    '⟨r⟩ / hydrogenic: against (3n*² − ℓ(ℓ+1))/2Z at the level\'s n*; departures come from the core.',
    '⟨1/r³⟩ sets the fine-structure scale; it diverges for s states.',
    'ν − n*: model eigenvalue against the level energy.',
    'R<sub>LR</sub> pair: Le Roy radius 2(⟨r²⟩<sub>A</sub><sup>½</sup> + ⟨r²⟩<sub>B</sub><sup>½</sup>) for two atoms in this state; the C₆ (van der Waals) description holds beyond it.',
    coreLine].filter(Boolean).join('<br>');
  $('nsc').innerHTML = [
    'Left: ⟨r⟩ against n*; dotted, the hydrogenic (3n*² − ℓ(ℓ+1))/2Z.',
    'Right: probability inside the core; dotted, the n*⁻³ law through the highest state of each series.',
    'Penetrating series (s, p) follow n*⁻³, the same scaling as the binding-energy spacing, which keeps their quantum defect nearly constant.',
    coreLine].filter(Boolean).join('<br>');
}

// ═════════════════════════ Draw ═════════════════════════
function select(k) {
  if (k === S.i) return;
  S.i = k;
  if (!['sum', 'sph'].includes(S.mk)) S.mk = 'sum';
  if (!chanOk(S.chan)) S.chan = 0;
  buildList(); draw();
  const row = $('slist').querySelector('.srow.on'); if (row) row.scrollIntoView({block: 'nearest'});
}
function draw() {
  const t = S.tab;
  ['3d', 'sl', 'rad', 'val', 'sc'].forEach(k => { $('pane-' + k).hidden = k !== t; });
  document.querySelectorAll('.tabs button').forEach(b => b.setAttribute('aria-selected', String(b.dataset.t === t)));
  buildSubs(); syncRail(); stats(); notes();
  $('slist').querySelectorAll('.srow').forEach(r => r.classList.toggle('on', +r.dataset.i === S.i));
  $('vptitle').innerHTML = `${D.species} ${cur().lblp} · ${subLabel()}`;
  const h = new URLSearchParams(); h.set('tab', t); h.set('s', cur().lbl); history.replaceState(null, '', '#' + h.toString());
  if (t === '3d') ensure3d();
  if (t === 'sl') buildSl();
  if (t === 'rad') buildRad();
  if (t === 'val') buildVal();
  if (t === 'sc') buildSc();
}

// ═════════════════════════ Wiring ═════════════════════════
$('h1').textContent = D.species + ' — orbitals';
const nm = {model: 0, hydrogenic: 0, coulomb: 0}; ST.forEach(s => nm[s.method]++);
$('sub').textContent = `${ST.length} states · radial: ` + Object.entries(nm).filter(([, v]) => v).map(([k, v]) => `${k} ${v}`).join(', ');
buildList();
document.querySelectorAll('.tabs button').forEach(b => b.onclick = () => { S.tab = b.dataset.t; draw(); });
document.querySelectorAll('#chan button').forEach(b => b.onclick = () => { if (chanOk(+b.dataset.v)) { S.chan = +b.dataset.v; draw(); } });
[['unit', 'unit'], ['plane', 'plane'], ['rq', 'rq'], ['norm', 'norm']].forEach(([id, k]) => document.querySelectorAll('#' + id + ' button').forEach(b => b.onclick = () => { S[k] = b.dataset.v; draw(); }));
$('dscale').onchange = e => { S.scale = +e.target.value; draw(); };
$('cmap').onchange = e => { S.cmap = +e.target.value; draw(); };
$('rlog').onchange = e => { S.rlog = e.target.checked; draw(); };
$('pinbtn').onclick = () => { if (!S.pins.includes(S.i) && S.pins.length < 4) S.pins.push(S.i); draw(); };
$('resetview').onclick = () => { if (REND) REND.resetView(); };
document.querySelectorAll('.rail input[type=range]').forEach(r => {
  const lab = $(r.dataset.lab); const upd = () => { if (lab) lab.textContent = (+r.value).toFixed(+(r.dataset.dec || 2)); };
  r.addEventListener('input', upd); upd();
});
if ($('lscale')) $('lscale').addEventListener('change', () => notes());
$('railbtn').onclick = () => { S.rail = !S.rail; $('app').classList.toggle('closed', !S.rail); $('railbtn').textContent = S.rail ? 'Hide controls' : 'Show controls'; $('railbtn').setAttribute('aria-expanded', String(S.rail)); setTimeout(() => { document.querySelectorAll('.pane:not([hidden]) .p').forEach(p => Plotly.Plots.resize(p)); if (REND && REND.redraw) REND.redraw(); }, 30); };
$('theme').onclick = () => { const o = ['auto', 'light', 'dark']; S.theme = o[(o.indexOf(S.theme) + 1) % 3]; const h = document.documentElement; S.theme === 'auto' ? h.removeAttribute('data-theme') : h.setAttribute('data-theme', S.theme); $('theme').textContent = 'Theme: ' + S.theme; draw(); };
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => draw());
const SK = 'qali-orbital-rail'; let shut = {}; try { shut = JSON.parse(localStorage.getItem(SK) || '{}'); } catch (e) { /* storage unavailable */ }
const secs = [...document.querySelectorAll('.rail section')].filter(x => x.querySelector(':scope>h2'));
const keepRail = () => { try { localStorage.setItem(SK, JSON.stringify(shut)); } catch (e) { /* storage unavailable */ } };
secs.forEach(sec => {
  const h = sec.querySelector(':scope>h2'), key = h.textContent.trim(); h.tabIndex = 0; h.setAttribute('role', 'button');
  const set = v => { sec.classList.toggle('shut', v); h.setAttribute('aria-expanded', String(!v)); shut[key] = v; };
  set(!!shut[key]); const tog = () => { set(!sec.classList.contains('shut')); keepRail(); };
  h.onclick = tog; h.onkeydown = e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); tog(); } };
});
$('colall').onclick = () => { secs.forEach(x => { x.classList.add('shut'); x.querySelector('h2').setAttribute('aria-expanded', 'false'); shut[x.querySelector('h2').textContent.trim()] = true; }); keepRail(); };
$('expall').onclick = () => { secs.forEach(x => { x.classList.remove('shut'); x.querySelector('h2').setAttribute('aria-expanded', 'true'); shut[x.querySelector('h2').textContent.trim()] = false; }); keepRail(); };

// export
const closeCM = () => { const m = $('cm'); if (m) m.remove(); };
document.addEventListener('click', e => { if (!e.target.closest('.cm')) closeCM(); });
async function lightly(fn) { const h = document.documentElement, prev = h.getAttribute('data-theme'); h.setAttribute('data-theme', 'light'); try { draw(); await fn(); } finally { if (prev == null) h.removeAttribute('data-theme'); else h.setAttribute('data-theme', prev); draw(); } }
function dl(name, href) { const a = document.createElement('a'); a.href = href; a.download = name; document.body.appendChild(a); a.click(); a.remove(); }
const blob = (t, type) => URL.createObjectURL(new Blob([t], {type}));
const q = x => '"' + String(x == null ? '' : x).replace(/"/g, '""') + '"';
const strip = s => String(s).replace(/<[^>]+>/g, '');
function csvTable(id) { const t = $(id), hr = t.tHead.rows[0]; return [[...hr.cells].map(c => q(strip(c.innerHTML).replace(/[▲▼]/g, '').trim())).join(','), ...[...t.tBodies[0].rows].map(r => [...r.cells].map(c => q(strip(c.innerHTML).replace(/\s+/g, ' ').trim())).join(','))].join('\n'); }
function radialCsv() { const s = cur(); return ['r_a0,u_a0^-1/2,P_a0^-1', ...Array.from(s.rr, (r, i) => [r.toPrecision(8), s.uu[i].toPrecision(7), (s.uu[i] ** 2).toPrecision(7)].join(','))].join('\n'); }
function valuesJson() { return JSON.stringify({species: D.species, units: 'atomic (a0)', states: ST.map(s => ({state: s.lbl, J: s.J, energy_eV: s.E, n_star: s.nstar, nu_radial: s.nu, method: s.method, nodes_a0: s.nodes, r_mean: s.r1, r2_mean: s.r2, inv_r_mean: s.rm1, inv_r3_mean: s.rm3, r_core: s.rc, p_core: s.pcore, r_outer_turning: s.tout}))}, null, 1); }
const PLOT_OF = {sl: ['psl', 'Slice'], rad: ['prad', 'Radial'], sc: ['psc', 'Scaling']};
$('exportbtn').onclick = e => {
  e.stopPropagation(); closeCM();
  const b = e.currentTarget.getBoundingClientRect(), m = document.createElement('div'); m.className = 'cm'; m.id = 'cm';
  const p = PLOT_OF[S.tab];
  let h = '';
  if (p) h += `<button data-a="svg">${p[1]} as SVG</button><button data-a="png">${p[1]} as PNG</button>`;
  if (S.tab === '3d') h += '<button data-a="png3d">3D view as PNG</button>';
  if (S.tab === 'val') h += '<button data-a="csv">Table as CSV</button>';
  h += '<button data-a="rcsv">Radial function as CSV</button><button data-a="json">Values as JSON</button><button data-a="pdf">Page as PDF (print dialog)</button>';
  m.innerHTML = h; document.body.appendChild(m);
  m.style.top = (b.bottom + 4) + 'px'; m.style.left = Math.max(8, Math.min(b.left, innerWidth - m.offsetWidth - 8)) + 'px';
  m.onclick = ev => {
    const a = ev.target.dataset.a; if (!a) return; closeCM();
    const fn = `${D.element}_${cur().lbl}_${S.tab}`.replace(/[^A-Za-z0-9_.-]/g, '_');
    if (a === 'svg') lightly(() => Plotly.downloadImage(p[0], {format: 'svg', width: 1400, height: 1000, filename: fn}));
    if (a === 'png') lightly(() => Plotly.downloadImage(p[0], {format: 'png', width: 1400, height: 1000, scale: 3, filename: fn}));
    if (a === 'png3d') { if (RENDER === 'isosurface') Plotly.downloadImage('isodiv', {format: 'png', width: 1400, height: 1000, scale: 2, filename: fn}); else { const u = REND && REND.png(); if (u) dl(fn + '.png', u); } }
    if (a === 'csv') dl(`${D.element}_orbital_values.csv`, blob(csvTable('vtab'), 'text/csv'));
    if (a === 'rcsv') dl(`${D.element}_${cur().lbl}_radial.csv`.replace(/[^A-Za-z0-9_.-]/g, '_'), blob(radialCsv(), 'text/csv'));
    if (a === 'json') dl(`${D.element}_orbital_values.json`, blob(valuesJson(), 'application/json'));
    if (a === 'pdf') window.print();
  };
};
draw();
