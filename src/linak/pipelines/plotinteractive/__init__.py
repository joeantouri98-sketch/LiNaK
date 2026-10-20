"""
Interactive Energy Level Visualization with Plotly

Features:
- Hover to see detailed info
- Zoom and pan
- Toggle levels / Rydberg / transitions / ORCA on/off
- Grotrian diagram with allowed E1 transition arrows
- ORCA TD-DFT excitations overlaid for comparison
- ORCA CASSCF/SOC D-lines overlaid automatically when data_json/<element>_soc.json exists
- Plot 3: excited→excited Grotrian by orbital l (NIST / Numerov / Coulomb)
- Interactive spectrum
- 3D view option
- Export to HTML

Usage: linak --pipeline plotinteractive <SPECIES_ID> [options]
      SPECIES_ID    neutral symbol (Na) or registered ion (Mg_c1, Na_a1)
      --soc-only  plot only data_json/<species>_soc.json (skip TD-DFT)
      (--soc is accepted as an alias for --soc-only)
      --extra / -e  overlay another ORCA-format JSON (repeatable); source
                    name is the filename stem after species_ (e.g. Na_BLY3P.json -> BLY3P).
                    Default: extras must share the same species_id prefix.
      --plots N     write plots 1..N only (1=grotrian, 2=+spectrum_levels,
                    3=+ee_grotrian, 4=+3d). Example: --plots 1

Layout: ``inputs`` reads the data folder, ``view`` applies the energy axis and
the display choices, then one module per plot writes its report (``grotrian``,
``spectrum``, ``ee_grotrian``, ``grotrian3d``).
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict

from linak.core.species import resolve_species
from linak.core.workspace import Workspace

from .ee_grotrian import plot_ee_grotrian
from .grotrian import plot_grotrian
from .grotrian3d import plot_3d
from .inputs import load_inputs
from .spectrum import plot_spectrum
from .view import View, prepare_view


def _parse_plots_max(spec):
    """Return highest plot index to generate (1..4); argparse ``type=``."""
    spec = (spec or 'all').strip().lower()
    if spec in ('all', '*', '1-4', '1,2,3,4'):
        return 4
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
                    f"invalid value {part!r} (use 1..4, e.g. --plots 1)") from None
    if not wanted or any(n < 1 or n > 4 for n in wanted):
        raise argparse.ArgumentTypeError("select plot numbers in 1..4 (e.g. --plots 1)")
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
    _parser.add_argument('--extra', '-e', action='append', default=[],
                        metavar='FILE',
                        help='ORCA-format JSON to overlay (repeatable). '
                             'Bare name or stem resolves under data_json/. '
                             'Must share the same species_id prefix.')
    _parser.add_argument('--plots', default='all', metavar='N', type=_parse_plots_max,
                        help='Highest plot to write: 1=grotrian only, 2=+spectrum, '
                             '3=+ee_grotrian, 4=+3d (default: all). '
                             'Also accepts 1,2 or 1-3 (max number wins).')

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

    # Plot 3 (excited-to-excited Grotrian) display.
    _parser.add_argument('--plot3-choice', choices=('1', '2', '3', '4'), default='1',
                         help='Plot 3 transitions: 1=NIST only (default), '
                              '2=NIST + Numerov (n ≤ 8), 3=NIST + Numerov (n ≤ 12), '
                              '4=all theoretical (capped at 300 strongest).')

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
                         help='--wl-min passed to the spectra pipeline (default: 200).')
    _parser.add_argument('--spectra-wl-max', type=float, default=None,
                         help='--wl-max passed to the spectra pipeline (default: 900).')
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
        viewers.append((spectra, argv, f"{element}_spectra.html",
                        "7.", "Absorption + emission spectrum"))

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
    if cli.plots < 4:
        print(f" --plots: writing plots 1..{cli.plots} only "
              f"(skipping {', '.join(str(n) for n in range(cli.plots + 1, 5))})")
    # Workspace roots for this run, resolved now
    data_dir = str(Workspace.current().data_json_dir)
    plots_dir = Workspace.current().plots_dir

    inputs = load_inputs(species, cli, data_dir)
    view = prepare_view(inputs, cli)
    output_dir = os.path.join(plots_dir, element)
    os.makedirs(output_dir, exist_ok=True)

    plot_grotrian(inputs, view, output_dir)
    if cli.plots >= 2:
        plot_spectrum(inputs, view, output_dir)
    if cli.plots >= 3:
        plot_ee_grotrian(inputs, view, cli.plot3_choice, output_dir)
    if cli.plots >= 4:
        plot_3d(inputs, output_dir)
    _print_summary(element, view, cli.plots, output_dir)

    _run_extra_viewers(cli, element, inputs.rydberg, data_dir, plots_dir)
    return 0


def _print_summary(element, view: View, plots_max, output_dir) -> None:
    if plots_max <= 3:
        print("\n" + "=" * 70)
        print(f" INTERACTIVE PLOTS CREATED (--plots {plots_max})" if plots_max == 1
              else f"INTERACTIVE PLOTS CREATED (--plots {plots_max})")
        print("=" * 70)
        print(f"\n Saved in: {os.path.abspath(output_dir)}/")
        if plots_max == 1:
            print(f"   1. {element}_grotrian.html          - Grotrian diagram + transitions")
        print({1: "   (plots 2-4 skipped)", 2: "   (plots 3-4 skipped)",
               3: "   (plot 4 skipped)"}[plots_max])
        print("=" * 70)
        return

    print("\n" + "="*70)
    print(" INTERACTIVE PLOTS CREATED")
    print("="*70)
    print(f"\n📂 All files saved in: {output_dir}/")
    print("\n Interactive HTML files:")
    print(f"   1. {element}_grotrian.html          - Grotrian diagram + transitions")
    print(f"   2. {element}_spectrum_levels.html   - Spectrum + energy levels")
    print(f"   3. {element}_ee_grotrian.html        - Grotrian by l (gs + ee transitions)")
    print(f"   4. {element}_3d_visualization.html  - 3D view (ROTATE ME!)")
    print("\n INTERACTIVE FEATURES:")
    print("   - Hover over levels/arrows → See energy, wavelength, label")
    print("   - Click legend → Toggle series on/off")
    print("   - Plot 1 (Grotrian) → Toggle transitions by spectral region")
    print("   - Plot 3 → l-column Grotrian, region-coloured arrows, filter + isolate")
    print("   - Click and drag → Zoom | Double-click → Reset")
    print("   - 3D VIEW: Click + drag to ROTATE!")

    if view.use_transitions:
        print(f"\n TRANSITIONS: {len(view.visible_transitions)} displayed")
        rc = defaultdict(int)
        for t in view.visible_transitions:
            rc[t['region']] += 1
        for region, count in rc.items():
            print(f"   {region:15s}: {count}")

    print(f"\n💾 File location: {os.path.abspath(output_dir)}")
    print("\n" + "="*70)
