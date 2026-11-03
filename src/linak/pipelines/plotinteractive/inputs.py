"""Everything one interactive-plot run reads from the data folder."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from linak.core.errors import MissingDataError
from linak.core.workspace import Workspace
from linak.integrations.orca_parser import deduplicate_excitations
from linak.physics.rydberg import label_to_pretty, unicode_digits_to_plain

from .common import _display_label


@dataclass(frozen=True)
class PlotInputs:
    """The ORCA, rydberg, transitions and lifetimes data of one species."""

    element: str                  # species id: the stem of every data and plot path
    species: dict                 # resolve_species(...)
    is_ion: bool                  # also True when an ORCA file says kind == 'ion'
    excitations: list             # TD-DFT levels (degenerate roots merged)
    soc_excitations: list         # CASSCF/SOC levels
    extra_overlays: list          # --extra: {name, slug, path, excitations}
    rydberg: dict | None          # <El>_rydberg.json
    rydberg_excitations: list
    transitions_meta: dict | None  # <El>_transitions.json
    transitions: list             # its 'transitions'
    has_lifetimes: bool           # <El>_lifetimes.json exists
    lifetimes_transitions: list   # its 'transitions' table
    lifetimes_by_state: dict      # state label -> lifetimes row
    fosc_by_transition: dict      # (upper, lower) and (lower, upper) -> f
    A_by_transition: dict         # (upper, lower) and (lower, upper) -> A (1/s)
    pretty_by_label: dict         # ASCII label -> display label

    @property
    def species_label(self) -> str:
        return self.species["label"]

    @property
    def uses_transitions(self) -> bool:
        return len(self.transitions) > 0

    @property
    def has_soc_overlay(self) -> bool:
        return bool(self.excitations and self.soc_excitations)

    @property
    def level_excitations(self) -> list:
        """Primary level lines: TD-DFT when available, otherwise SOC-only data."""
        return self.excitations if self.excitations else self.soc_excitations

    @property
    def n_start(self):
        return self.rydberg.get('n_start', 2) if self.rydberg else 2

    @property
    def gs_meta(self) -> dict:
        return (self.rydberg.get('ground_state') or {}) if self.rydberg else {}

    @property
    def gs_label(self) -> str:
        """The ground-state label from the rydberg ground_state block (not a hardcoded 's')."""
        gs = self.gs_meta
        return (gs.get('label') or f"{gs.get('n', self.n_start)}{gs.get('l', 's')}"
                if gs else f"{self.n_start}s")

    def disp(self, label):
        """Readable spectroscopic label (see ``common._display_label``)."""
        return _display_label(label, self.pretty_by_label)


def _extra_source_name(path, element):
    """Derive filter/legend name from filename: Na_BLY3P.json -> BLY3P."""
    stem = os.path.splitext(os.path.basename(path))[0]
    prefix = f"{element}_"
    if stem.lower().startswith(prefix.lower()):
        stem = stem[len(prefix):]
    return stem or os.path.splitext(os.path.basename(path))[0]


def _extra_matches_species(path, species_id):
    """True if basename is species_id.json or species_id_*.json."""
    stem = os.path.splitext(os.path.basename(path))[0]
    sid = species_id.lower()
    s = stem.lower()
    return s == sid or s.startswith(sid + "_")


def _resolve_extra_json(spec, element, data_dir=None):
    """
    Resolve --extra argument to a filesystem path.
    Accepts absolute/relative paths, bare filenames under data_json/,
    or stem-only labels (BLY3P -> data_json/Na_BLY3P.json).
    Rejects files that do not share the same species_id prefix.
    """
    if not spec:
        return None
    if data_dir is None:
        data_dir = str(Workspace.current().data_json_dir)
    spec = spec.strip()
    if os.path.isfile(spec):
        path = os.path.normpath(spec)
        if not _extra_matches_species(path, element):
            print(f"  Warning: --extra '{spec}' is not for species {element}; "
                  f"skipping (cross-species overlay is opt-in later)")
            return None
        return path
    candidates = [os.path.join(data_dir, spec)]
    if not spec.lower().endswith('.json'):
        candidates.append(os.path.join(data_dir, f"{spec}.json"))
        candidates.append(os.path.join(data_dir, f"{element}_{spec}.json"))
        candidates.append(f"{element}_{spec}.json")
        candidates.append(f"{spec}.json")
    else:
        base = os.path.basename(spec)
        if not base.lower().startswith(f"{element}_".lower()):
            candidates.append(os.path.join(data_dir, f"{element}_{base}"))
    for c in candidates:
        if os.path.isfile(c):
            path = os.path.normpath(c)
            if not _extra_matches_species(path, element):
                print(f"  Warning: --extra '{spec}' resolved to {path} but "
                      f"is not for species {element}; skipping")
                return None
            return path
    return None


def _load_extra_excitations(path, element):
    """Load ORCA-format excitations JSON; return dict or None."""
    name = _extra_source_name(path, element)
    try:
        with open(path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
    except Exception as exc:
        print(f"  Warning: could not read --extra {path}: {exc}")
        return None
    raw = payload.get('excitations')
    if not isinstance(raw, list) or not raw:
        print(f"  Warning: --extra {path} has no excitations list; skipping")
        return None
    exc_list = deduplicate_excitations(raw)
    print(f" Extra [{name}]: {len(raw)} roots -> {len(exc_list)} levels "
          f"from {path}")
    return {
        'name': name,
        'slug': name.lower().replace(' ', '_'),
        'path': path.replace('\\', '/'),
        'excitations': exc_list,
    }


def load_inputs(species, cli, data_dir) -> PlotInputs:
    """Read the ORCA (TD-DFT, SOC, --extra), rydberg, transitions and lifetimes JSON."""
    element = species["species_id"]
    is_ion_species = species["kind"] == "ion"
    pretty_by_label = {}   # ASCII key → display label, from the rydberg/transitions JSON
    excitations = []
    soc_excitations = []
    extra_overlays = []
    rydberg_data = None
    rydberg_excitations = []
    tdata = None
    transitions_data = []
    lifetimes_transitions = []
    lifetimes_by_state  = {}   # state_label -> lifetime entry dict
    fosc_by_transition  = {}   # (upper_label, lower_label) -> fosc
    A_by_transition     = {}   # (upper_label, lower_label) -> A_s

    json_file = f"{data_dir}/{element}.json"
    soc_file = f"{data_dir}/{element}_soc.json"
    if not os.path.exists(soc_file):
        # runorca appends recipe/norb suffixes for non-default SOC runs
        # (e.g. Cl_soc_casscf.json, Rb_soc_norb8.json), use the newest match.
        import glob as _glob
        _soc_variants = sorted(_glob.glob(f"{data_dir}/{element}_soc*.json"),
                               key=os.path.getmtime, reverse=True)
        if _soc_variants:
            soc_file = _soc_variants[0].replace("\\", "/")
            print(f"  (Using SOC data: {soc_file})")
    use_soc_only = bool(cli.soc_only)
    if use_soc_only:
        if not os.path.exists(soc_file):
            raise MissingDataError(
                f"{soc_file} not found; run: linak --pipeline runorca {element} --mode soc")
        with open(soc_file, 'r', encoding='utf-8') as f:
            soc_data = json.load(f)
        if soc_data.get("kind") == "ion":
            is_ion_species = True
        soc_excitations = soc_data.get("excitations", [])
        if not soc_excitations:
            raise MissingDataError(f"no SOC excitation data in {soc_file}")
        print(f" Loaded {len(soc_excitations)} SOC-resolved levels for {element} (--soc-only)")
    else:
        if os.path.exists(json_file):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if data.get("kind") == "ion":
                is_ion_species = True
            excitations_raw = data.get("excitations", [])
            if excitations_raw:
                excitations = deduplicate_excitations(excitations_raw)
                print(f" Loaded {len(excitations_raw)} TD-DFT roots → "
                      f"{len(excitations)} levels for {element}")
        if os.path.exists(soc_file):
            with open(soc_file, 'r', encoding='utf-8') as f:
                soc_data = json.load(f)
            if soc_data.get("kind") == "ion":
                is_ion_species = True
            soc_excitations = soc_data.get("excitations", [])
            if soc_excitations:
                print(f" Found {len(soc_excitations)} SOC levels in {soc_file} will overlay on plot")
        if not excitations and not soc_excitations:
            raise MissingDataError(
                f"no ORCA data found for {element}: expected {json_file} and/or "
                f"{soc_file}; run: linak --pipeline runorca {element}")

    _seen_extra_names = set()
    for _espec in cli.extra or []:
        _epath = _resolve_extra_json(_espec, element)
        if _epath is None:
            print(f"  Warning: --extra '{_espec}' not found; skipping")
            continue
        _eloaded = _load_extra_excitations(_epath, element)
        if _eloaded is None:
            continue
        _ename = _eloaded['name']
        if _ename in _seen_extra_names:
            print(f"  Warning: duplicate extra source name '{_ename}'; skipping {_epath}")
            continue
        _seen_extra_names.add(_ename)
        extra_overlays.append(_eloaded)
    if extra_overlays:
        print(f" {len(extra_overlays)} extra overlay source(s): "
              + ', '.join(e['name'] for e in extra_overlays))

    # Try to load Rydberg data
    rydberg_file = f"{data_dir}/{element}_rydberg.json"

    if os.path.exists(rydberg_file):
        try:
            with open(rydberg_file, 'r', encoding='utf-8') as f:
                rydberg_data = json.load(f)
            rydberg_excitations = rydberg_data.get("excitations", [])
            print(f" Found {len(rydberg_excitations)} Rydberg states in {element}_rydberg.json")
            for _ex in rydberg_excitations:
                _lab = _ex.get('label')
                if _lab:
                    pretty_by_label[_lab] = unicode_digits_to_plain(
                        _ex.get('label_pretty') or label_to_pretty(_lab))
            _gs_block = rydberg_data.get('ground_state') or {}
            if _gs_block.get('label'):
                pretty_by_label[_gs_block['label']] = unicode_digits_to_plain(
                    _gs_block.get('label_pretty')
                    or label_to_pretty(_gs_block['label']))
        except Exception as e:
            print(f"  Could not load Rydberg data: {e}")

    # Try to load transitions
    transitions_file = f"{data_dir}/{element}_transitions.json"

    if os.path.exists(transitions_file):
        try:
            with open(transitions_file, 'r', encoding='utf-8') as f:
                tdata = json.load(f)
            transitions_data = tdata.get("transitions", [])
            print(f" Found {len(transitions_data)} transitions in {element}_transitions.json")
            # Metal transitions carry term pretty labels — feed _disp lookup
            for _t in transitions_data:
                for _lab, _pretty in (
                    (_t.get('upper_label'), _t.get('upper_label_pretty')),
                    (_t.get('lower_label'), _t.get('lower_label_pretty')),
                ):
                    if _lab and _pretty and _lab not in pretty_by_label:
                        pretty_by_label[_lab] = unicode_digits_to_plain(_pretty)
        except Exception as e:
            print(f"  Could not load transitions: {e}")
    else:
        print(f"  (No transitions file found , run linak --pipeline transitions {element} to generate)")

    # Try to load lifetimes data
    lifetimes_file = f"{data_dir}/{element}_lifetimes.json"

    if os.path.exists(lifetimes_file):
        try:
            with open(lifetimes_file, 'r', encoding='utf-8') as f:
                lt_raw = json.load(f)
            lifetimes_transitions = lt_raw.get('transitions', [])
            for lt in lt_raw.get('lifetimes', []):
                lifetimes_by_state[lt['state']] = lt
            print(f" Loaded lifetimes for {len(lifetimes_by_state)} states")
            # Build fosc lookup by (upper_label, lower_label) from lifetimes transitions.
            # Used to scale arrow width in the Grotrian diagram.
            for _lt in lt_raw.get('transitions', []):
                _ul = _lt.get('upper_label', '')
                _ll = _lt.get('lower_label', '')
                _f  = _lt.get('fosc', 0.0)
                _A  = _lt.get('A_s', 0.0)
                if _ul and _ll and _f > 0:
                    fosc_by_transition[(_ul, _ll)] = _f
                    fosc_by_transition[(_ll, _ul)] = _f
                    A_by_transition[(_ul, _ll)]    = _A
                    A_by_transition[(_ll, _ul)]    = _A
            print(f" fosc lookup: {len(fosc_by_transition)//2} unique transitions with fosc")
        except Exception as e:
            print(f"  Could not load lifetimes: {e}")

    return PlotInputs(
        element=element,
        species=species,
        is_ion=is_ion_species,
        excitations=excitations,
        soc_excitations=soc_excitations,
        extra_overlays=extra_overlays,
        rydberg=rydberg_data,
        rydberg_excitations=rydberg_excitations,
        transitions_meta=tdata,
        transitions=transitions_data,
        has_lifetimes=os.path.exists(lifetimes_file),
        lifetimes_transitions=lifetimes_transitions,
        lifetimes_by_state=lifetimes_by_state,
        fosc_by_transition=fosc_by_transition,
        A_by_transition=A_by_transition,
        pretty_by_label=pretty_by_label,
    )
