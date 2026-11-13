"""What the Grotrian diagram shows: the energy axis, the IE filter and the display choices."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from qali.core.spectroscopy import spectral_region

from .common import ionization_energy_eV, orca_line_strength
from .inputs import PlotInputs


@dataclass(frozen=True)
class View:
    """The energy axis and the levels / transitions selected for display."""

    energy_shift: float           # added to every excitation energy
    shift_label: str              # 'Excitation', 'Absolute' or 'Shifted'
    ground_state_energy: float
    IE_for_filter: float | None   # ionization energy on the excitation axis
    IE_for_orca_filter: float | None   # None for ions (no neutral continuum)
    label_lookup: dict            # ORCA level energy -> label
    fosc_by_energy: dict          # ORCA level energy -> total f
    has_fosc: bool
    sorted_energies: list         # ORCA level energies (after the IE filter)
    shifted_energies: list        # the same on the displayed axis
    n_levels: int
    use_rydberg: bool
    sorted_rydberg: list          # Rydberg/NIST level energies on the displayed axis
    rydberg_energy_counts: dict
    rydberg_label_lookup_conv: dict   # displayed energy -> state label
    rydberg_level_meta: dict      # displayed energy -> {label, theoretical, unc, source}
    use_transitions: bool
    visible_transitions: list
    visible_orca_tddft: list
    visible_orca_soc: list
    visible_extra_overlays: list  # copies of the --extra overlays, display-filtered


def _apply_orca_wl_filter(ex_list, ochoice, wl_min=None, wl_max=None):
    """Filter ORCA excitations by wavelength display choice."""
    if ochoice == "5" or not ex_list:
        return []
    if ochoice == "1":
        return list(ex_list)
    if ochoice == "2":
        return [e for e in ex_list
                if e['energy_eV'] > 0
                and 380 <= 1239.84 / e['energy_eV'] <= 750]
    if ochoice == "3":
        return [e for e in ex_list
                if e['energy_eV'] > 0
                and 1239.84 / e['energy_eV'] <= 750]
    if ochoice == "4":
        try:
            wl_min = float(wl_min)
            wl_max = float(wl_max)
            return [e for e in ex_list
                    if e['energy_eV'] > 0
                    and wl_min <= 1239.84 / e['energy_eV'] <= wl_max]
        except (TypeError, ValueError):
            return list(ex_list)
    return list(ex_list)


def prepare_view(inputs: PlotInputs, cli) -> View:
    """Apply --energy-shift, --filter-ie and the display choices; print the summaries."""
    _cli = cli
    excitations = inputs.excitations
    soc_excitations = inputs.soc_excitations
    extra_overlays = inputs.extra_overlays
    rydberg_data = inputs.rydberg
    rydberg_excitations = inputs.rydberg_excitations
    transitions_data = inputs.transitions
    use_transitions = inputs.uses_transitions
    tdata = inputs.transitions_meta
    is_ion_species = inputs.is_ion
    level_excitations = inputs.level_excitations
    has_soc_overlay = inputs.has_soc_overlay
    _disp = inputs.disp
    label_lookup = {}
    energy_shift = 0.0
    shift_label = "Excitation"
    use_rydberg = False
    rydberg_label_lookup_conv   = {}   # plot_energy -> state label
    rydberg_level_meta          = {}   # plot_energy -> {label, theoretical, unc, source}
    sorted_rydberg              = []

    for ex in level_excitations:
        e = ex.get('energy_eV')
        lbl = ex.get('label')
        if e is not None and lbl and e not in label_lookup:
            label_lookup[e] = lbl

    # Energies from the main (ORCA) data

    energy_counts = defaultdict(int)
    fosc_by_energy = defaultdict(float)   # total dipole strength per physical level

    for ex in level_excitations:
        e = ex["energy_eV"]
        energy_counts[e] = 1
        fosc_by_energy[e] = orca_line_strength(ex)

    # Check if we have real oscillator strengths or just zeros
    has_fosc = any(f > 0 for f in fosc_by_energy.values())
    if has_fosc:
        print(" Oscillator strengths available , spectrum will use fosc as intensity")
    else:
        print("  No oscillator strengths found , spectrum will use degeneracy as proxy")
        print("   run qali --pipeline orca_to_json to get real fosc values")

    sorted_energies = sorted(energy_counts.keys())
    n_levels = len(sorted_energies)

    print(f" Found {n_levels} unique energy levels (ORCA data)")

    # Energy shift option

    choice = _cli.energy_shift

    if choice == "2":
        abs_ground = _cli.absolute_ground
        energy_shift = abs_ground
        shift_label = "Absolute"
        use_rydberg = True
        print(f" Will shift all energies by {abs_ground:.3f} eV")
        if rydberg_excitations:
            print(" Rydberg series will be included (already in absolute energies)")
    elif choice == "3":
        custom_shift = _cli.custom_shift
        energy_shift = custom_shift
        shift_label = "Shifted"
        print(f" Will shift all energies by {custom_shift:.3f} eV")
    else:
        print(" Using excitation energies (no shift)")

    # Apply shift to main data
    shifted_energies = [e + energy_shift for e in sorted_energies]
    ground_state_energy = 0 + energy_shift

    # Filter levels above the ionization threshold

    IE_for_filter = rydberg_data.get('ionization_energy_eV') if rydberg_data else None

    # In absolute mode the user already entered the ground state energy.
    # IE = -abs_ground (ionization limit sits at 0 eV). No second prompt needed.
    if IE_for_filter is None and shift_label == "Absolute" and not is_ion_species:
        IE_for_filter = -abs_ground

    # Ions: never clip ORCA/SOC/--extra with a neutral IE. An ion's own
    # *_rydberg.json IE (e.g. Mg+ -> Mg2+) may still guide Rydberg/transitions.
    IE_for_orca_filter = None if is_ion_species else IE_for_filter
    if is_ion_species:
        print(" Ion species: skipping IE filter for ORCA/SOC levels "
              "(neutral continuum does not apply)")

    if IE_for_orca_filter:
        # In excitation mode: IE is the threshold (excitation energies > IE = above ionization)
        # In absolute mode: threshold is at 0 (levels above 0 = above ionization)
        threshold = 0.0 if shift_label == "Absolute" else IE_for_orca_filter
        above_IE_pairs = [(orig, sh) for orig, sh in zip(sorted_energies, shifted_energies)
                          if sh > threshold]
        if above_IE_pairs:
            print(f"\n⚠  {len(above_IE_pairs)} ORCA level(s) above ionization threshold "
                  f"(IE = {IE_for_orca_filter:.3f} eV):")
            for orig, sh in above_IE_pairs:
                print(f"   {sh:.3f} eV  fosc={fosc_by_energy[orig]:.4f}")
            print("   These are unphysical (virtual/continuum states)")
            filter_choice = str(_cli.filter_ie).strip().lower()
            if filter_choice == "y":
                pairs = [(orig, sh) for orig, sh in zip(sorted_energies, shifted_energies)
                         if sh <= threshold]
                sorted_energies  = [p[0] for p in pairs]
                shifted_energies = [p[1] for p in pairs]
                n_levels = len(sorted_energies)
                print(f"   Filtered : {n_levels} levels remaining")
            else:
                print("  Keeping all levels")

    # Extract Rydberg energies , convert to match the current energy axis
    # In excitation mode:  plot_e = IE + ryd_e_raw  (ryd energies are negative from IE)
    # In absolute mode:    plot_e = ryd_e_raw + energy_shift
    rydberg_energy_counts       = defaultdict(int)

    if rydberg_excitations:
        use_rydberg = True   # show Rydberg levels regardless of energy mode
        IE_ryd = ionization_energy_eV(rydberg_data)
        for ex in rydberg_excitations:
            ryd_e_raw = ex.get('energy_eV', 0)
            lbl       = ex.get('label', '')
            if shift_label == "Absolute":
                ryd_e_plot = ryd_e_raw
            else:
                ryd_e_plot = IE_ryd + ryd_e_raw
            rydberg_energy_counts[ryd_e_plot] += 1
            if lbl and ryd_e_plot not in rydberg_label_lookup_conv:
                rydberg_label_lookup_conv[ryd_e_plot] = lbl
            # Prefer measured over theoretical / QDT if energies collide
            prev = rydberg_level_meta.get(ryd_e_plot)
            cand = dict(
                label=lbl or (prev or {}).get('label', 'Rydberg'),
                label_pretty=(ex.get('label_pretty')
                              or _disp(lbl)
                              or (prev or {}).get('label_pretty')),
                theoretical=bool(ex.get('theoretical')),
                uncertainty_eV=ex.get('uncertainty_eV'),
                source=ex.get('source', 'QDT model'),
                # Excitation from ground (mode-independent) for TM auto-show cutoff
                exc_eV=float(IE_ryd) + float(ryd_e_raw),
            )
            if prev is None:
                rydberg_level_meta[ryd_e_plot] = cand
            elif prev.get('theoretical') and not cand['theoretical']:
                rydberg_level_meta[ryd_e_plot] = cand
            elif prev.get('source') != 'NIST' and cand['source'] == 'NIST':
                rydberg_level_meta[ryd_e_plot] = cand
        sorted_rydberg = sorted(rydberg_energy_counts.keys())
        n_theo = sum(1 for m in rydberg_level_meta.values() if m.get('theoretical'))
        print(f" Found {len(sorted_rydberg)} unique Rydberg levels"
              f" ({n_theo} theoretical [ ])")

    # Transition filter: the wavelength range to display

    visible_transitions = transitions_data
    if use_transitions and transitions_data:
        print("\n" + "="*70)
        print("TRANSITION DISPLAY OPTIONS")
        print("="*70)
        print(f"  Total transitions available: {len(transitions_data)}")
        rc = tdata.get('region_counts', {})
        for region, count in rc.items():
            print(f"    {region:15s}: {count}")

        tchoice = _cli.transition_display

        if tchoice == "2":
            visible_transitions = [t for t in transitions_data if 380 <= t['wavelength_nm'] <= 750]
        elif tchoice == "3":
            visible_transitions = [t for t in transitions_data if t['wavelength_nm'] <= 750]
        elif tchoice == "4":
            wl_min = _cli.transition_wl_min
            wl_max = _cli.transition_wl_max
            if wl_min is not None and wl_max is not None:
                visible_transitions = [t for t in transitions_data
                                       if wl_min <= t['wavelength_nm'] <= wl_max]
        elif tchoice == "5":
            visible_transitions = []
            use_transitions = False

        if use_transitions:
            print(f" Will display {len(visible_transitions)} NIST/Rydberg transitions")


    visible_orca_tddft = excitations
    visible_orca_soc = soc_excitations if (has_soc_overlay or not excitations) else []
    visible_extra_overlays = [
        {'name': e['name'], 'slug': e['slug'], 'excitations': list(e['excitations'])}
        for e in extra_overlays
    ]
    _all_orca = list(visible_orca_tddft) + list(visible_orca_soc)
    for _exv in visible_extra_overlays:
        _all_orca.extend(_exv['excitations'])

    if _all_orca:
        # Build region breakdown (skip spin-forbidden and above-IE)
        orca_regions = {}
        for ex in _all_orca:
            if ex.get('spin_forbidden'):
                continue
            if IE_for_orca_filter and ex['energy_eV'] >= IE_for_orca_filter:
                continue
            if ex['energy_eV'] <= 0:
                continue
            wl = 1239.84 / ex['energy_eV']
            r = spectral_region(wl)
            orca_regions[r] = orca_regions.get(r, 0) + 1

        if orca_regions:
            print("\n" + "="*70)
            print("ORCA TRANSITION DISPLAY OPTIONS")
            print("="*70)
            print("  Allowed ORCA transitions (below IE):")
            for region, count in orca_regions.items():
                print(f"    {region:15s}: {count}")
            if has_soc_overlay:
                print("  (includes TD-DFT + CASSCF/SOC overlay)")
            if visible_extra_overlays:
                print("  (also applies to --extra overlays: "
                      + ', '.join(e['name'] for e in visible_extra_overlays) + ')')

            ochoice = _cli.orca_display
            wl_min = wl_max = None
            if ochoice == "4":
                wl_min = _cli.orca_wl_min
                wl_max = _cli.orca_wl_max

            visible_orca_tddft = _apply_orca_wl_filter(excitations, ochoice, wl_min, wl_max)
            if has_soc_overlay or (soc_excitations and not excitations):
                visible_orca_soc = _apply_orca_wl_filter(soc_excitations, ochoice, wl_min, wl_max)
            else:
                visible_orca_soc = []
            for _exv in visible_extra_overlays:
                _exv['excitations'] = _apply_orca_wl_filter(
                    _exv['excitations'], ochoice, wl_min, wl_max)

            _orca_pool = list(visible_orca_tddft) + list(visible_orca_soc)
            for _exv in visible_extra_overlays:
                _orca_pool.extend(_exv['excitations'])
            n_unique = len(set(e['energy_eV'] for e in _orca_pool
                              if not e.get('spin_forbidden')
                              and (not IE_for_orca_filter
                                   or e['energy_eV'] < IE_for_orca_filter)))
            print(f" Will display {n_unique} unique ORCA/extra transitions")

    print("="*70)

    return View(
        energy_shift=energy_shift,
        shift_label=shift_label,
        ground_state_energy=ground_state_energy,
        IE_for_filter=IE_for_filter,
        IE_for_orca_filter=IE_for_orca_filter,
        label_lookup=label_lookup,
        fosc_by_energy=fosc_by_energy,
        has_fosc=has_fosc,
        sorted_energies=sorted_energies,
        shifted_energies=shifted_energies,
        n_levels=n_levels,
        use_rydberg=use_rydberg,
        sorted_rydberg=sorted_rydberg,
        rydberg_energy_counts=rydberg_energy_counts,
        rydberg_label_lookup_conv=rydberg_label_lookup_conv,
        rydberg_level_meta=rydberg_level_meta,
        use_transitions=use_transitions,
        visible_transitions=visible_transitions,
        visible_orca_tddft=visible_orca_tddft,
        visible_orca_soc=visible_orca_soc,
        visible_extra_overlays=visible_extra_overlays,
    )
