"""
Interactive Energy Level Visualization with Plotly

Features:
- Hover to see detailed info
- Zoom and pan
- Toggle levels / Rydberg / transitions / ORCA on/off
- Grotrian diagram with allowed E1 transition arrows
- ORCA TD-DFT excitations overlaid for comparison
- ORCA CASSCF/SOC D-lines overlaid automatically when data_json/<element>_soc.json exists
- Export to HTML

Usage: linak --pipeline plotinteractive <SPECIES_ID> [options]
      SPECIES_ID    neutral symbol (Na) or registered ion (Mg_c1, Na_a1)
      --soc-only  plot only data_json/<species>_soc.json (skip TD-DFT)
      (--soc is accepted as an alias for --soc-only)
      --extra / -e  overlay another ORCA-format JSON (repeatable); source
                    name is the filename stem after species_ (e.g. Na_BLY3P.json -> BLY3P).
                    Default: extras must share the same species_id prefix.
      --plots N     kept for old command lines; only the Grotrian page is written (the spectrum
                    is the Spectrum stage's page, <species>_spectrum.html).

Layout: ``inputs`` reads the data folder, ``view`` applies the energy axis and
the display choices, then ``grotrian_page`` writes the report. The spectrum is the Spectrum
stage's own page (``pipelines/spectra.py``).
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict

from linak.core.species import resolve_species
from linak.core.workspace import Workspace

from .grotrian_page import plot_grotrian
from .inputs import load_inputs
from .view import View, prepare_view


def _parse_plots_max(spec):
    """Return highest plot index to generate (1..2); argparse ``type=``."""
    spec = (spec or 'all').strip().lower()
    if spec in ('all', '*', '1-2', '1,2'):
        return 2
    wanted = set()
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            try:
                wanted.update(range(int(a), int(b) + 1))
            except ValueError:
                raise argparse.ArgumentTypeError(f"invalid range {part!r}") from None
        else:
            try:
                wanted.add(int(part))
            except ValueError:
                raise argparse.ArgumentTypeError(
                    f"invalid value {part!r} (use 1..2, e.g. --plots 1)") from None
    if not wanted or any(n < 1 or n > 2 for n in wanted):
        raise argparse.ArgumentTypeError("select plot numbers in 1..2 (e.g. --plots 1)")
    return max(wanted)


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser for this pipeline."""
    _parser = argparse.ArgumentParser(
        prog="linak --pipeline plotinteractive",
        description='Interactive energy-level / Grotrian plots for a species '
                    '(neutral or registered ion)',
    )
    _parser.add_argument('element',
                        help='Species id: symbol (Na) or ion stem (Mg_c1, Na_a1)')
    _parser.add_argument('--soc-only', '--soc', action='store_true',
                        help='Plot only data/<species>_soc.json (skip TD-DFT)')
    for _kind, _what in (('rydberg', 'Rydberg'), ('transitions', 'Transitions'), ('lifetimes', 'Lifetimes')):
        _parser.add_argument(f'--{_kind}-file', default=None, metavar='FILE',
                             help=f'{_what} data to read instead of <species>_{_kind}.json '
                                  '(a path, or a name under the data folder).')
    _parser.add_argument('--soc-file', default=None, metavar='FILE',
                        help='SOC data to overlay (or plot with --soc-only). Bare name or stem resolves '
                             'under data_json/; default: <species>_soc.json. Must share the species id.')
    _parser.add_argument('--no-soc-overlay', action='store_true',
                        help='Do not overlay the SOC levels on the TD-DFT diagram.')
    _parser.add_argument('--extra', '-e', action='append', default=[],
                        metavar='FILE',
                        help='ORCA-format JSON to overlay (repeatable). '
                             'Bare name or stem resolves under data_json/. '
                             'Must share the same species_id prefix.')
    _parser.add_argument('--plots', default='all', metavar='N', type=_parse_plots_max,
                        help=argparse.SUPPRESS)   # old option: plot 2 (the spectrum) is now the Spectrum stage

    # Energy reference for the ORCA/SOC level axis.
    _parser.add_argument('--energy-shift', choices=('1', '2', '3'), default='1',
                         help='Level energy reference: 1=excitation energies '
                              '(ground = 0 eV, default), 2=absolute energies '
                              '(shift by --absolute-ground), 3=custom shift '
                              '(add --custom-shift to every level).')
    _parser.add_argument('--absolute-ground', type=float, default=0.0,
                         help='Ground-state energy added to all levels when '
                              '--energy-shift is 2 (default: 0).')
    _parser.add_argument('--custom-shift', type=float, default=0.0,
                         help='Energy added to all levels when --energy-shift '
                              'is 3 (default: 0).')

    # ORCA/SOC continuum filter (only applies when levels above IE exist).
    _parser.add_argument('--filter-ie', choices=('y', 'n'), default='y',
                         help='When levels above the ionization threshold are '
                              'detected, filter them out (y, default) or keep them (n).')

    # NIST/Rydberg transition display.
    _parser.add_argument('--transition-display', choices=('1', '2', '3', '4', '5'),
                         default='1',
                         help='NIST/Rydberg transitions displayed: 1=all (default), '
                              '2=visible (380-750 nm), 3=UV + visible, '
                              '4=custom range (--transition-wl-min/max), 5=none.')
    _parser.add_argument('--transition-wl-min', type=float, default=None,
                         help='Lower wavelength bound (nm) when --transition-display is 4.')
    _parser.add_argument('--transition-wl-max', type=float, default=None,
                         help='Upper wavelength bound (nm) when --transition-display is 4.')

    # ORCA/extra transition display.
    _parser.add_argument('--orca-display', choices=('1', '2', '3', '4', '5'), default='1',
                         help='ORCA/--extra transitions displayed: 1=all (default), '
                              '2=visible (380-750 nm), 3=UV + visible, '
                              '4=custom range (--orca-wl-min/max), 5=none.')
    _parser.add_argument('--orca-wl-min', type=float, default=None,
                         help='Lower wavelength bound (nm) when --orca-display is 4.')
    _parser.add_argument('--orca-wl-max', type=float, default=None,
                         help='Upper wavelength bound (nm) when --orca-display is 4.')

    # Optional sub-launches (further pipeline stages run from here).
    _parser.add_argument('--also-orbital3d', action='store_true',
                         help='Also run linak --pipeline orbital3d for this species after plotting.')
    _parser.add_argument('--orbital3d-n-max', type=int, default=None,
                         help='--n-max passed to the orbital3d pipeline '
                              '(default: the Rydberg data n_max, else 10).')
    _parser.add_argument('--orbital3d-grid', type=int, default=None,
                         help='--grid passed to the orbital3d pipeline '
                              '(grid points per axis; default: orbital3d\'s own).')
    _parser.add_argument('--also-spectra', action='store_true',
                         help='Also run linak --pipeline spectra for this species after plotting.')
    _parser.add_argument('--spectra-wl-min', type=float, default=None,
                         help='--wl-min passed to the spectra pipeline (initial view, vacuum nm).')
    _parser.add_argument('--spectra-wl-max', type=float, default=None,
                         help='--wl-max passed to the spectra pipeline (initial view, vacuum nm).')
    return _parser


def _run_extra_viewers(cli, element, rydberg_data, data_dir, plots_dir) -> None:
    """Run the viewers requested with ``--also-orbital3d`` / ``--also-spectra``.

    They are ordinary pipelines, so they run in-process (a frozen bundle has
    no Python interpreter or ``.py`` files to launch). A failing viewer only
    warns: the Grotrian plots are already written.
    """
    from linak.core.errors import PipelineError

    viewers = []
    if cli.also_orbital3d:
        from linak.pipelines import orbital3d

        n_max = cli.orbital3d_n_max
        if n_max is None:
            n_max = rydberg_data.get('n_max', 10) if rydberg_data else 10
        argv = [element, '--n-max', str(n_max), '--data-dir', data_dir]
        if cli.orbital3d_grid is not None:
            argv += ['--grid', str(cli.orbital3d_grid)]
        viewers.append((orbital3d, argv, f"{element}_orbital3d.html",
                        "6.", "Orbital |ψ|² viewer (click node → render)"))
    if cli.also_spectra:
        from linak.pipelines import spectra

        argv = [element, '--data-dir', data_dir]
        if cli.spectra_wl_min is not None:
            argv += ['--wl-min', str(cli.spectra_wl_min)]
        if cli.spectra_wl_max is not None:
            argv += ['--wl-max', str(cli.spectra_wl_max)]
        viewers.append((spectra, argv, f"{element}_spectrum.html",
                        "7.", "Spectrum page"))

    for module, argv, out_name, number, what in viewers:
        name = module.__name__.rsplit('.', 1)[-1]
        print("\n" + "=" * 70)
        print(f"\n  Running: linak --pipeline {name} {' '.join(argv)}")
        print("  " + "─" * 60)
        try:
            status = module.main(argv)
        except SystemExit as exc:
            status = exc.code
        except PipelineError as exc:
            print(f"\n  ⚠  {name}: {exc}")
            continue
        out_file = os.path.join(plots_dir, element, out_name)
        if status in (0, None) and os.path.exists(out_file):
            print(f"\n   Saved: {os.path.abspath(out_file)}")
            print(f"   {number} {out_name:<30s} - {what}")
        else:
            print(f"\n  ⚠  {name} exited with code {status}")
        print("=" * 70 + "\n")


def main(argv: list[str] | None = None) -> int:
    """Run the interactive plot pipeline; return an exit status."""
    parser = build_parser()
    cli = parser.parse_args(argv)
    try:
        species = resolve_species(cli.element.strip())
    except ValueError as exc:
        parser.error(str(exc))
    element = species["species_id"]  # stem used for all data_json / plot paths
    if species["kind"] == "ion":
        print(f" Species: {element} ({species['label']}, ion)")
    # Workspace roots for this run, resolved now
    data_dir = str(Workspace.current().data_json_dir)
    plots_dir = Workspace.current().plots_dir

    inputs = load_inputs(species, cli, data_dir)
    view = prepare_view(inputs, cli)
    output_dir = os.path.join(plots_dir, element)
    os.makedirs(output_dir, exist_ok=True)

    plot_grotrian(inputs, view, output_dir)
    _print_summary(element, view, output_dir)

    _run_extra_viewers(cli, element, inputs.rydberg, data_dir, plots_dir)
    return 0


def _print_summary(element, view: View, output_dir) -> None:
    print("\n" + "=" * 70)
    print(" INTERACTIVE PLOTS CREATED")
    print("=" * 70)
    print(f"\n All files saved in: {os.path.abspath(output_dir)}/")
    print("\n Interactive HTML files:")
    print(f"   1. {element}_grotrian.html          - Grotrian diagram + transitions")
    print("   (the spectrum is the Spectrum stage's page: linak --pipeline spectra "
          f"{element})")
    print("\n INTERACTIVE FEATURES:")
    print("   - Hover over levels/arrows -> energy, wavelength, f, A, lifetimes")
    print("   - Grotrian: filter by region, source, f, n; ground / excited lower level;")
    print("     right-click a transition for 'View more' (incl. 68% / 95% extrapolation bands)")
    print("   - Click and drag -> Zoom | Double-click -> Reset")

    if view.use_transitions:
        print(f"\n TRANSITIONS: {len(view.visible_transitions)} available")
        rc = defaultdict(int)
        for t in view.visible_transitions:
            rc[t['region']] += 1
        for region, count in rc.items():
            print(f"   {region:15s}: {count}")
    print("\n" + "=" * 70)
