"""
The ``tweezer`` pipeline: optical-tweezer scattering & heating.
-----------------------------------------------------------------------------
Photon scattering rate, recoil heating and an estimate of how long an atom
stays trapped in a far-off-resonant single-beam Gaussian tweezer. The physics
is in ``linak.physics.tweezer_core``; this module loads its inputs, prints a
summary, writes the JSON and draws the report. (Near-resonant laser-cooling
curves are the ``scattering_rate`` pipeline; the conservative AC Stark depth
and trap frequencies are also in ``polarizability``.)

Inputs (in the data folder):
    <element>_polarizability.json   cached ground-state transitions
                                    (else rebuilt from rydberg + transitions)
    <element>_lifetimes.json        Γ = A_total per upper state, atomic mass

Outputs:
    data/<element>_tweezer.json
    plots/<element>/<element>_tweezer.html

Usage:
    linak --pipeline tweezer Na
    linak --pipeline tweezer Na --wl 1064 --intensity 50 --waist 0.8
    linak --pipeline tweezer --all
"""

__version__ = '1.0'

from linak.core import user_tables as _user_tables
import argparse
import json
import math
import os
from dataclasses import asdict

from linak.core.constants import ATOMIC_MASS_AMU
from linak.core.errors import MissingDataError
from linak.core.data_inputs import resolve_data_file
from linak.core.fileio import write_json
from linak.core.workspace import Workspace
from linak.physics.polarizability_core import (
    build_transitions_for_state,
    get_available_elements,
    load_cached_transitions,
    load_element,
)
from linak.physics.tweezer_core import R_SC_PREFACTOR, U_PREFACTOR, compute
from linak.report_templates import (
    fullscreen_css,
    inject_body,
    inject_head,
    js_constant,
    load as load_template,
    render as render_template,
    write_offline_report,
)

DEFAULT_WL_NM = {
    'Li': 1064.0, 'Na': 1064.0, 'K': 1064.0,
    'Rb': 1064.0, 'Cs': 1064.0, 'Fr': 1064.0,
}
_user_tables.apply("tweezer.default_wavelength", DEFAULT_WL_NM)
DEFAULT_I_KWCM2 = 50.0
DEFAULT_WAIST_UM = 1.0


# ============================================================================
# INPUTS
# ============================================================================

def load_ground_transitions(element, data_dir, polarizability_file=None):
    """(ground-state label, transition network), or None without the data.

    Prefers the network cached by the polarizability pipeline; otherwise
    rebuilds it from the rydberg + transitions JSON.
    """
    cached = load_cached_transitions(element, data_dir, path=polarizability_file)
    if polarizability_file and not (cached and cached.get('transitions_gs')):
        raise MissingDataError(f"{os.path.basename(polarizability_file)} has no ground-state transitions")
    if cached and cached.get('transitions_gs'):
        gs_label = cached.get('gs_label') or '?'
        print(f"  Ground {gs_label}: {len(cached['transitions_gs'])} cached transitions "
              f"from {os.path.basename(polarizability_file or element + '_polarizability.json')}")
        return gs_label, cached['transitions_gs']
    data = load_element(element, data_dir)
    if data is None:
        return None
    gs_label = data['gs_label']
    transitions = build_transitions_for_state(gs_label, data, data['transitions'])
    print(f"  Ground {gs_label}: built {len(transitions)} transitions "
          f"(no polarizability cache)")
    return gs_label, transitions


def load_upper_state_rates(element, data_dir, lifetimes_file=None):
    """({state label: A_total in s⁻¹}, atomic mass in amu or None) from lifetimes."""
    path = lifetimes_file or os.path.join(data_dir, f"{element}_lifetimes.json")
    if not os.path.exists(path):
        print(f"  ! {path} not found — scattering needs upper-state Γ")
        return {}, None

    with open(path, encoding='utf-8') as f:
        data = json.load(f)

    rates = {}
    mass = None
    for row in data.get('lifetimes', []):
        state, A = row.get('state'), row.get('A_total_s')
        if not state or A is None or A <= 0:
            continue
        rates[state] = float(A)
        if mass is None and row.get('mass_amu'):
            mass = float(row['mass_amu'])
    return rates, mass


# RUN ELEMENT

def run_element(element, wl_nm=None, I_kWcm2=DEFAULT_I_KWCM2,
                waist_um=DEFAULT_WAIST_UM, wl_min=400.0, wl_max=1600.0,
                n_grid=1500, data_dir=None, polarizability_file=None, lifetimes_file=None):
    """(ground-state label, TweezerResult) for *element*, or None without data."""
    if data_dir is None:
        data_dir = str(Workspace.current().data_json_dir)
    print(f"\n{'='*65}")
    print(f"  {element}  —  Optical tweezer scattering & heating")
    print(f"{'='*65}")

    loaded = load_ground_transitions(element, data_dir, polarizability_file)
    if loaded is None:
        return None
    gs_label, transitions = loaded
    rates, mass_lt = load_upper_state_rates(element, data_dir, lifetimes_file)
    mass = mass_lt or ATOMIC_MASS_AMU.get(element, 1.0)
    if wl_nm is None:
        wl_nm = DEFAULT_WL_NM.get(element, 1064.0)

    result = compute(
        transitions, rates, mass, wl_nm=wl_nm, I_kWcm2=I_kWcm2, waist_um=waist_um,
        wl_min=wl_min, wl_max=wl_max, n_grid=n_grid,
    )
    print(f"  Gamma attached for {result.n_with_gamma}/{result.n_transitions} channels "
          f"(skipped {result.n_core_skipped} core)")
    _print_point(result.point)
    return gs_label, result


def _print_point(point):
    def _fmt(x, fmt='.4g'):
        return f"{x:{fmt}}" if isinstance(x, float) and math.isfinite(x) else str(x)

    print(f"  Point @ lambda={point.wl_nm:.1f} nm, I={point.I_kWcm2:g} kW/cm^2, "
          f"w0={point.waist_um:g} um")
    print(f"    Re[alpha] = {_fmt(point.alpha_re_au, '.2f')} au   "
          f"Im[alpha] = {_fmt(point.alpha_im_au, '.3e')} au")
    u_khz = abs(point.U_kHz_h) if math.isfinite(point.U_kHz_h) else point.U_kHz_h
    print(f"    U/k_B = {_fmt(point.U_uK, '.2f')} uK   "
          f"(|U|/h = {_fmt(u_khz, '.2f')} kHz)")
    print(f"    R_sc  = {_fmt(point.R_sc_Hz, '.4g')} /s   "
          f"(<dt> = {_fmt(point.tau_scatter_s, '.3g')} s)")
    print(f"    Heating = {_fmt(point.heating_uK_s, '.4g')} uK/s   "
          f"tau_heat ~ {_fmt(point.tau_heat_s, '.3g')} s")
    if math.isfinite(point.nu_r_Hz):
        print(f"    nu_r = {_fmt(point.nu_r_Hz, '.1f')} Hz   "
              f"nu_z = {_fmt(point.nu_z_Hz, '.1f')} Hz")
    if point.top_channels:
        print("    Top scattering channels:")
        for ch in point.top_channels[:3]:
            print(f"      {ch.label:10s} lambda={ch.wl_nm:.1f} nm  "
                  f"R={ch.R_sc_Hz:.3g} /s  [{ch.source}]")



# PLOT

def plot_tweezer(element, gs_label, result, out_dir):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    el = element
    sp = result.spectra
    pt = result.point
    I_ref = result.I_ref_kWcm2
    waist = result.waist_um

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(
            f"Photon scattering rate  ·  I = {I_ref:g} kW/cm²",
            "Recoil heating rate  ·  Ė/k_B",
            "Trap-heating lifetime  τ_heat ≈ |U|/Ė",
        ),
    )

    fig.add_trace(go.Scatter(
        x=sp.wl_nm, y=sp.R_sc_Hz, mode='lines',
        name='R_sc', line=dict(color='#4ea1ff', width=1.6),
        hovertemplate='λ=%{x:.2f} nm<br>R_sc=%{y:.4g} s⁻¹<extra></extra>',
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=sp.wl_nm, y=sp.heating_uK_s, mode='lines',
        name='heating', line=dict(color='#ff8c42', width=1.6),
        hovertemplate='λ=%{x:.2f} nm<br>Ė/k_B=%{y:.4g} µK/s<extra></extra>',
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=sp.wl_nm, y=sp.tau_heat_s, mode='lines',
        name='τ_heat', line=dict(color='#7dce82', width=1.6),
        hovertemplate='λ=%{x:.2f} nm<br>τ_heat=%{y:.4g} s<extra></extra>',
    ), row=3, col=1)

    for res in result.resonances:
        wl = res.wl_nm
        if not (result.wl_min <= wl <= result.wl_max):
            continue
        for row in (1, 2, 3):
            fig.add_vline(
                x=wl, line=dict(color='rgba(200,80,80,0.35)', width=1, dash='dot'),
                row=row, col=1)
        fig.add_annotation(
            x=wl, y=1.02, yref='y domain', xref='x',
            text=res.label, showarrow=False, textangle=-90,
            font=dict(size=9, color='#c66'), row=1, col=1)

    if math.isfinite(pt.R_sc_Hz):
        fig.add_trace(go.Scatter(
            x=[pt.wl_nm], y=[pt.R_sc_Hz], mode='markers',
            name='point', marker=dict(size=10, color='#fff',
                                      line=dict(color='#4ea1ff', width=2)),
            hovertemplate=f"point λ={pt.wl_nm:.1f} nm<br>"
                          f"R_sc={pt.R_sc_Hz:.4g} s⁻¹<extra></extra>",
            showlegend=False,
        ), row=1, col=1)

    fig.update_yaxes(type='log', title_text='R_sc (s⁻¹)', row=1, col=1)
    fig.update_yaxes(type='log', title_text='µK / s', row=2, col=1)
    fig.update_yaxes(type='log', title_text='τ_heat (s)', row=3, col=1)
    fig.update_xaxes(title_text='Wavelength (nm)', row=3, col=1)

    fig.update_layout(
        template='plotly_dark',
        autosize=True,
        margin=dict(l=70, r=40, t=80, b=50),
        title=dict(
            text=(f"<b>{el} — Optical tweezer scattering & heating</b><br>"
                  f"<sup>ground {gs_label}  ·  "
                  f"m = {result.mass_amu:.3f} amu (IUPAC avg)  ·  "
                  f"{result.n_with_gamma} channels with Γ  ·  "
                  f"w₀ = {waist:g} µm</sup>"),
            x=0.02, xanchor='left',
        ),
        legend=dict(orientation='h', y=1.02, x=1, xanchor='right'),
        hovermode='x unified',
    )

    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{el}_tweezer.html")
    raw = fig.to_html(include_plotlyjs=False, full_html=True)
    raw = inject_head(raw, fullscreen_css())
    raw = _inject_controls(raw, element, result)
    write_offline_report(out_file, raw)
    print(f"  ✓ {out_file}")
    return out_file


def _inject_controls(raw_html, element, result):
    """Floating point calculator. Live α(λ) from spectrum interpolation;
    R_sc, U, heating scale linearly with I in the far-detuned regime."""
    pt = result.point
    sp = result.spectra
    el = element
    mass = result.mass_amu
    waist = result.waist_um

    step = max(1, len(sp.wl_nm) // 800)
    wl_js = sp.wl_nm[::step]
    re_js = sp.alpha_re_au[::step]
    im_js = sp.alpha_im_au[::step]

    css = load_template('tweezer_control.css')

    html = render_template(
        'tweezer_control.html',
        wl_default=pt.wl_nm,
        I_default=pt.I_kWcm2,
        waist=waist,
    )

    js = render_template(
        'tweezer_control.js',
        el_js=json.dumps(el),
        wl_data=json.dumps(wl_js),
        re_data=json.dumps(re_js),
        im_data=json.dumps(im_js),
        mass=mass,
        U_PREF=U_PREFACTOR,
        R_PREF=R_SC_PREFACTOR,
        H_SI=js_constant('H_SI'),
        KB_SI=js_constant('KB_SI'),
        AMU_KG=js_constant('AMU_KG'),
    )

    raw_html = inject_head(raw_html, css)
    raw_html = inject_body(raw_html, html + js)
    return raw_html

# SAVE JSON

def _clean(obj):
    """Replace non-JSON floats with None."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return obj


def save_json(element, gs_label, result, data_dir, inputs=None):
    out = dict(
        element=element,
        gs_label=gs_label,
        mass_amu=result.mass_amu,
        n_transitions=result.n_transitions,
        n_with_gamma=result.n_with_gamma,
        I_ref_kWcm2=result.I_ref_kWcm2,
        waist_um=result.waist_um,
        wl_range_nm=[result.wl_min, result.wl_max],
        point=_clean(asdict(result.point)),
        note=(
            'R_sc from Im[alpha(omega)] with Gamma = A_total from lifetimes.json, scaled as Gamma(omega/omega_k)^2 per channel (Grimm et al. 2000, Eq. 10). '
            'Heating uses 2 E_r(lambda_trap) per scatter. '
            'tau_heat = |U|/E_dot is an order-of-magnitude trap lifetime. '
            'Conservative depth/freqs also in polarizability.py AC Stark panel.'
        ),
    )
    if inputs:                       # only when a file other than the default was chosen
        out['inputs'] = inputs
    path = os.path.join(data_dir, f"{element}_tweezer.json")
    write_json(path, out)
    print(f"  ✓ {path}")

# MAIN

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="linak --pipeline tweezer",
        description='Optical-tweezer photon scattering & recoil heating')
    parser.add_argument('element', nargs='?', default=None)
    parser.add_argument('--wl', type=float, default=None,
                        help='Trap wavelength (nm); default 1064')
    parser.add_argument('--intensity', type=float, default=DEFAULT_I_KWCM2,
                        help='Peak intensity (kW/cm²); default 50')
    parser.add_argument('--waist', type=float, default=DEFAULT_WAIST_UM,
                        help='1/e² waist (µm); default 1')
    parser.add_argument('--wl-min', type=float, default=400.0)
    parser.add_argument('--wl-max', type=float, default=1600.0)
    parser.add_argument('--n-grid', type=int, default=1500)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--data-dir', default=str(Workspace.current().data_json_dir))
    parser.add_argument('--polarizability-file', default=None, metavar='FILE',
                        help='Polarizability data to read instead of <species>_polarizability.json '
                             '(a path, or a name under the data folder).')
    parser.add_argument('--lifetimes-file', default=None, metavar='FILE',
                        help='Lifetimes data to read instead of <species>_lifetimes.json '
                             '(a path, or a name under the data folder).')
    parser.add_argument('--no-plot', action='store_true',
                        help='Write the data (JSON) only; skip the interactive plot.')
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    data_dir = args.data_dir
    plots_dir = Workspace.current().plots_dir
    os.makedirs(plots_dir, exist_ok=True)
    print(f"linak --pipeline tweezer (v{__version__})")

    from linak.physics.polarizability_core import require_alpha_species

    if args.all:
        elements = []
        for el in get_available_elements(data_dir):
            pol = os.path.join(data_dir, f"{el}_polarizability.json")
            life = os.path.join(data_dir, f"{el}_lifetimes.json")
            if os.path.exists(pol) and os.path.exists(life):
                elements.append(el)
        if not elements:
            raise MissingDataError(
                f"No elements with polarizability+lifetimes in {data_dir}.")
        print("Running:", ", ".join(elements))
    else:
        if args.element is None:
            parser.error(
                "missing ELEMENT (or pass --all); available: "
                f"{', '.join(get_available_elements(data_dir)) or '(none)'}"
            )
        elements = [require_alpha_species(args.element, "tweezer.py")["species_id"]]

    if args.all and (args.polarizability_file or args.lifetimes_file):
        parser.error("--polarizability-file / --lifetimes-file apply to one species; drop --all")
    failed = []
    for el in elements:
        el_out = os.path.join(plots_dir, el)
        os.makedirs(el_out, exist_ok=True)
        pol_file = resolve_data_file(args.polarizability_file, el, data_dir, "polarizability")
        life_file = resolve_data_file(args.lifetimes_file, el, data_dir, "lifetimes")
        inputs = {k: os.path.basename(v) for k, v in
                  (("polarizability", pol_file), ("lifetimes", life_file)) if v}
        try:
            result = run_element(
                el,
                wl_nm=args.wl,
                I_kWcm2=args.intensity,
                waist_um=args.waist,
                wl_min=args.wl_min,
                wl_max=args.wl_max,
                n_grid=args.n_grid,
                data_dir=data_dir,
                polarizability_file=pol_file, lifetimes_file=life_file,
            )
        except MissingDataError as exc:
            if not args.all:
                raise
            print(f"  ⚠  {el}: {exc}")
            result = None
        if result is None:
            failed.append(el)
            continue
        gs_label, tweezer = result
        save_json(el, gs_label, tweezer, data_dir, inputs)
        if not args.no_plot:
            plot_tweezer(el, gs_label, tweezer, el_out)
    if failed:
        print(f"  Failed: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
