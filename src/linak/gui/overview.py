"""The Overview tab: a card dashboard of one species' computed results.

Reads the JSON files the pipelines wrote into the output folder; it never
recomputes anything.
"""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from linak.core.workspace import Workspace

from .stages import PIPELINE, stage_has_output


class SpeciesOverviewWidget(QWidget):
    """Compact species dashboard for the selected species."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.root = Path(".")
        self.species = ""

        self._root_widget = QWidget(self)
        self._cards_layout = QGridLayout(self._root_widget)
        self._cards_layout.setColumnStretch(0, 1)
        self._cards_layout.setColumnStretch(1, 1)

        outer = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._root_widget)
        outer.addWidget(scroll)

        btn_row = QHBoxLayout()
        refresh_btn = QPushButton("Refresh overview")
        refresh_btn.clicked.connect(self._refresh_from_button)
        copy_btn = QPushButton("Copy summary")
        copy_btn.clicked.connect(self._copy_summary)
        btn_row.addWidget(refresh_btn)
        btn_row.addWidget(copy_btn)
        btn_row.addStretch(1)
        outer.addLayout(btn_row)

        self._summary_parts: list[str] = []
        self._empty_state()

    def _refresh_from_button(self):
        self.refresh(self.root, self.species)

    def _copy_summary(self):
        if self._summary_parts:
            QApplication.clipboard().setText("\n\n".join(self._summary_parts))

    def _empty_state(self):
        self._summary_parts = []
        self._clear_cards()
        placeholder = QGroupBox("Species overview")
        form = QFormLayout(placeholder)
        form.addRow("Status", QLabel("Select a species to view the computed summary."))
        self._add_card(placeholder, 0, 0)

    def _clear_cards(self):
        while self._cards_layout.count():
            item = self._cards_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    @staticmethod
    def _number(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _format_value(value):
        if value is None:
            return "—"
        if isinstance(value, bool):
            return "Yes" if value else "No"
        if isinstance(value, (int, float)):
            x = float(value)
            if x == 0:
                return "0"
            if abs(x) >= 1e5 or abs(x) < 1e-4:
                return f"{x:.6e}"
            return f"{x:.6g}"
        return str(value)

    @staticmethod
    def _find_value(data, keys):
        if isinstance(data, dict):
            for key in keys:
                if key in data and not isinstance(data[key], (dict, list)):
                    return data[key]
            for value in data.values():
                found = SpeciesOverviewWidget._find_value(value, keys)
                if found is not None:
                    return found
        elif isinstance(data, list):
            for value in data:
                found = SpeciesOverviewWidget._find_value(value, keys)
                if found is not None:
                    return found
        return None

    @staticmethod
    def _find_list(data, keys):
        if isinstance(data, dict):
            for key in keys:
                if isinstance(data.get(key), list):
                    return data[key]
            for value in data.values():
                found = SpeciesOverviewWidget._find_list(value, keys)
                if found is not None:
                    return found
        elif isinstance(data, list):
            for value in data:
                found = SpeciesOverviewWidget._find_list(value, keys)
                if found is not None:
                    return found
        return None

    @staticmethod
    def _load_json(path):
        """(document, None), or (None, reason) when it is unreadable or not an object."""
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError) as exc:
            return None, str(exc)
        if not isinstance(data, dict):
            return None, f"expected a JSON object, found {type(data).__name__}"
        return data, None

    def _add_card(self, card, row, col):
        self._cards_layout.addWidget(card, row, col)

    def _card(self, title, rows):
        self._summary_parts.append(
            "\n".join([title, *(f"  {label}: {value}" for label, value in rows)]))
        box = QGroupBox(title)
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignRight)
        for label, value in rows:
            form.addRow(label, QLabel(str(value)))
        return box

    def _make_species_card(self, core_data, rydberg_data):
        rows = []
        for label, keys in (
            ("Species", ("label", "species_label", "species_id")),
            ("Element", ("element", "symbol")),
            ("Charge", ("charge",)),
            ("Multiplicity", ("multiplicity",)),
            ("Z", ("Z", "atomic_number")),
            ("Ground energy", ("ground_energy_eV",)),
            ("Ionization energy", ("ionization_energy_eV",)),
        ):
            value = next(
                (source[key] for source in (core_data, rydberg_data) for key in keys
                 if not isinstance(source.get(key), (dict, list, type(None)))),
                None,
            )
            if value is not None:
                rows.append((label, self._format_value(value)))

        if not rows:
            rows.append(("Data", "No core metadata found"))

        return self._card("Species information", rows)

    def _make_pipeline_card(self, root, species):
        rows = []
        ws = Workspace(root=root)
        for stage in PIPELINE:
            if stage["id"] == "compare":
                continue
            present = stage_has_output(stage, ws, species)
            rows.append((stage["label"], "Complete" if present else "Missing"))
        return self._card("Pipeline status", rows)

    def _make_rydberg_card(self, data):
        if not data:
            return self._card("Rydberg series", [("Status", "Not available")])

        rows = []
        for label, keys in (
            ("Method", ("method",)),
            ("Ionization energy", ("ionization_energy_eV",)),
            ("States", ("num_states",)),
            ("Theoretical states", ("num_theoretical",)),
            ("n min", ("n_start",)),
            ("n max", ("n_max",)),
        ):
            value = self._find_value(data, keys)
            if value is not None:
                rows.append((label, self._format_value(value)))

        defects = data.get("quantum_defects")
        if isinstance(defects, dict) and defects:
            defect_text = ", ".join(
                f"{k}={self._format_value(v)}" for k, v in defects.items()
            )
            rows.append(("Quantum defects", defect_text))

        return self._card("Rydberg series", rows)

    def _make_transitions_card(self, data):
        if not data:
            return self._card("Transitions", [("Status", "Not available")])

        trans_list = self._find_list(data, ("transitions", "lines"))
        if not trans_list:
            return self._card("Transitions", [("Status", "No transitions found")])

        wl_values = []
        fs = []
        for row in trans_list:
            if not isinstance(row, dict):
                continue
            wl = self._number(row.get("wavelength_nm", row.get("wl_nm", row.get("wl"))))
            if wl is not None and wl > 0:
                wl_values.append(wl)
            fosc = self._number(row.get("fosc", row.get("oscillator_strength")))
            if fosc is not None:
                fs.append((fosc, row))

        rows = [("Transitions", str(len(trans_list)))]
        if wl_values:
            rows.extend([
                ("λ min", f"{min(wl_values):.3f} nm"),
                ("λ max", f"{max(wl_values):.3f} nm"),
            ])
        if fs:
            strongest, row = max(fs, key=lambda x: x[0])
            label = row.get("label") or row.get("upper_label") or "transition"
            rows.append(("Strongest f", f"{strongest:.3g}  ({label})"))

        return self._card("Transitions", rows)

    def _make_lifetimes_card(self, data):
        if not data:
            return self._card("Radiative lifetimes", [("Status", "Not available")])

        lifetime_list = self._find_list(data, ("lifetimes", "states"))
        if not lifetime_list:
            return self._card("Radiative lifetimes", [("Status", "No lifetimes found")])

        taus = []
        channel_count = 0
        source_counts = {}
        for row in lifetime_list:
            if not isinstance(row, dict):
                continue
            tau = self._number(row.get("lifetime_s", row.get("tau_s", row.get("tau"))))
            if tau is not None:
                taus.append(float(tau))
            channels = row.get("channels", [])
            if isinstance(channels, list):
                channel_count += len(channels)
                for ch in channels:
                    if not isinstance(ch, dict):
                        continue
                    source = ch.get("fosc_source") or ch.get("source") or "unknown"
                    source_counts[source] = source_counts.get(source, 0) + 1

        rows = [("States", str(len(lifetime_list))), ("Channels", str(channel_count))]
        if taus:
            rows.extend([
                ("τ min", self._format_value(min(taus))),
                ("τ max", self._format_value(max(taus))),
            ])
        if source_counts:
            src_text = ", ".join(f"{k}: {v}" for k, v in sorted(source_counts.items()))
            rows.append(("Sources", src_text))

        return self._card("Radiative lifetimes", rows)

    def _make_polarizability_card(self, data):
        if not data:
            return self._card("Polarizability", [("Status", "Not available")])

        rows = []
        for key in ("alpha_0", "alpha_static", "C6_au", "C6_expt_au", "magic_wavelength_nm"):
            value = self._find_value(data, (key,))
            if value is not None:
                rows.append((key, self._format_value(value)))
        if not rows:
            rows.append(("Status", "File present but no summary values found"))
        return self._card("Polarizability", rows)


    def _make_blackbody_card(self, data):
        if not data:
            return self._card(
                "Blackbody radiation",
                [("Status", "Not available")],
            )

        rows = []

        # These are the physically useful scalar results emitted by blackbody.py. The search is recursive because some outputs are nested under result/shift/rates/continuum sections.
        scalar_fields = (
            ("Temperature", ("T_K", "temperature_K", "T")),
            ("State", ("state", "state_label", "excited_state", "excited")),
            ("Static shift", (
                "static_shift_hz_at_T",
                "shift_static_hz_at_T",
                "static_hz_at_T",
            )),
            ("Dynamic shift", (
                "dyn_shift_hz_at_T",
                "dynamic_shift_hz_at_T",
                "dyn_diff_hz_at_T",
            )),
            ("Total shift", (
                "shift_hz_at_T",
                "total_shift_hz_at_T",
                "total_hz_at_T",
            )),
            ("BBR depopulation rate", (
                "Gamma_BBR",
                "gamma_bbr",
                "Gamma_bbr",
                "rate_BBR",
                "bbr_rate_s",
            )),
            ("Photoionization rate", (
                "Gamma_PI",
                "gamma_pi",
                "Gamma_pi",
                "photoionization_rate",
                "pi_rate_s",
            )),
            ("Total quenching rate", (
                "Gamma_quench",
                "gamma_quench",
                "total_quench_rate",
            )),
            ("Effective lifetime", (
                "tau_eff",
                "effective_lifetime_s",
                "lifetime_eff_s",
            )),
            ("BBR lifetime", (
                "tau_BBR",
                "tau_bbr",
                "bbr_lifetime_s",
            )),
        )

        for label, keys in scalar_fields:
            value = self._find_value(data, keys)
            if value is not None:
                rows.append((label, self._format_value(value)))

        # Report whether optional physical blocks were included.
        if isinstance(data, dict):
            for block_name, label in (
                ("rates", "Bound-bound rates table"),
                ("continuum", "Continuum/photoionization block"),
                ("rydberg_shifts", "Rydberg shifts table"),
            ):
                if block_name in data:
                    block = data[block_name]
                    if isinstance(block, list):
                        value = f"Present ({len(block)} rows)"
                    elif isinstance(block, dict):
                        value = f"Present ({len(block)} fields)"
                    else:
                        value = "Present"
                    rows.append((label, value))

        if not rows:
            rows.append(
                (
                    "Status",
                    "File present but no recognized blackbody results found",
                )
            )

        return self._card("Blackbody radiation", rows)

    def _make_tweezer_card(self, data):
        if not data:
            return self._card("Optical tweezer", [("Status", "Not available")])

        rows = []
        for key in (
            "wl_nm", "I_kWcm2", "waist_um", "trap_depth_uK",
            "scattering_rate_s-1", "recoil_heating_K_s", "trap_frequency_Hz"
        ):
            value = self._find_value(data, (key,))
            if value is not None:
                rows.append((key, self._format_value(value)))

        if not rows:
            rows.append(("Status", "File present but no useful tweezer values found"))

        return self._card("Optical tweezer", rows)


    def _make_hyperfine_card(self, data):
        if not data:
            return self._card(
                "Hyperfine / Zeeman",
                [("Status", "Not available")],
            )

        self._summary_parts.append("Hyperfine / Zeeman (see the Overview tab)")
        box = QGroupBox("Hyperfine / Zeeman")
        outer = QVBoxLayout(box)

        summary = QFormLayout()
        summary.setLabelAlignment(Qt.AlignRight)

        for label, keys in (
            ("Isotope", ("isotope_A", "isotope")),
            ("Nuclear spin I", ("I", "nuclear_spin")),
            ("Magnetic moment μI", ("mu_I", "mu")),
            ("Quadrupole moment Q", ("Q_barn", "Q")),
            ("Nuclear g-factor", ("gI", "g_I")),
        ):
            value = self._find_value(data, keys)
            if value is not None:
                summary.addRow(label, QLabel(self._format_value(value)))

        special = data.get("special", {})
        clock = special.get("microwave_clock", {}) if isinstance(special, dict) else {}

        if clock:
            summary.addRow(
                "Clock transition",
                QLabel(
                    f"{clock.get('state', 'ground')}  "
                    f"F={clock.get('F_upper', '—')} ↔ "
                    f"F={clock.get('F_lower', '—')}"
                ),
            )

            for label, key, unit in (
                ("Clock frequency", "frequency_MHz", "MHz"),
                ("Clock frequency", "frequency_GHz", "GHz"),
                ("Clock frequency", "frequency_Hz", "Hz"),
                ("Clock wavelength", "wavelength_cm", "cm"),
                ("Clock wavelength", "wavelength_m", "m"),
            ):
                value = clock.get(key)
                if value is not None:
                    summary.addRow(
                        label,
                        QLabel(f"{self._format_value(value)} {unit}"),
                    )

            if clock.get("note"):
                summary.addRow("Clock note", QLabel(str(clock["note"])))

        states = data.get("states", [])
        if states:
            summary.addRow("Computed states", QLabel(str(len(states))))

        outer.addLayout(summary)

        if states:
            outer.addWidget(QLabel("Hyperfine constants and F-levels"))

            constants = QTableWidget()
            constants.setColumnCount(7)
            constants.setHorizontalHeaderLabels([
                "State",
                "J",
                "A (MHz)",
                "B (MHz)",
                "C (kHz)",
                "F",
                "E_hf / gF",
            ])
            constants.setRowCount(0)
            constants.setWordWrap(False)
            constants.setTextElideMode(Qt.ElideNone)
            constants.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            constants.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            constants.setSizeAdjustPolicy(
                QTableWidget.AdjustToContents
            )

            row_index = 0
            for state in states:
                f_levels = state.get("F_levels", [])
                if not f_levels:
                    f_levels = [{}]

                for f_level in f_levels:
                    constants.insertRow(row_index)

                    values = [
                        state.get("label", ""),
                        self._format_value(state.get("J")),
                        self._format_value(state.get("A_MHz")),
                        self._format_value(state.get("B_MHz")),
                        self._format_value(state.get("C_kHz")),
                        self._format_value(f_level.get("F")),
                        (
                            f"E={self._format_value(f_level.get('E_MHz'))} MHz; "
                            f"gF={self._format_value(f_level.get('gF'))}"
                        ),
                    ]

                    for column, value in enumerate(values):
                        constants.setItem(
                            row_index,
                            column,
                            QTableWidgetItem(str(value)),
                        )

                    row_index += 1

            constants.resizeColumnsToContents()
            constants.resizeRowsToContents()

            
            header_height = constants.horizontalHeader().height()
            frame_height = constants.frameWidth() * 2
            row_height = constants.verticalHeader().defaultSectionSize()
            total_height = (
                header_height
                + frame_height
                + constants.rowCount() * row_height
                + 8
            )
            constants.setMinimumHeight(total_height)
            constants.setMaximumHeight(total_height)

            outer.addWidget(constants)

        if not states and not clock:
            outer.addWidget(
                QLabel(
                    "Hyperfine file is present but contains no "
                    "states or clock transition."
                )
            )

        return box

    def _make_feshbach_card(self, data):
        if not data:
            return self._card("Feshbach summary", [("Status", "Not available")])

        rows = []
        for key in (
            "isotope_A", "isotope", "a_bg_default_a0", "C6_au",
            "C6_source", "beta6_a0", "a_bar_a0", "kT_uK",
            "mass_amu", "mass_source", "k_a0inv"
        ):
            value = self._find_value(data, (key,))
            if value is not None:
                rows.append((key, self._format_value(value)))

        resonances = data.get("resonances")
        if isinstance(resonances, list) and resonances:
            rows.append(("Resonances", str(len(resonances))))
            b0s = [float(r.get("B0_G")) for r in resonances if self._number(r.get("B0_G")) is not None]
            if b0s:
                rows.append(("B0 range", f"{min(b0s):.1f}–{max(b0s):.1f} G"))

        if not rows:
            rows.append(("Status", "File present but no summary values found"))

        return self._card("Feshbach summary", rows)


    def _make_feshbach_channels_card(self, data):
        if not data:
            return self._card(
                "Feshbach channels",
                [("Status", "Not available")],
            )

        resonances = data.get("resonances")
        if not isinstance(resonances, list) or not resonances:
            return self._card(
                "Feshbach channels",
                [("Status", "No resonances found")],
            )

        title = f"Feshbach channels / resonances ({len(resonances)})"
        self._summary_parts.append(f"{title} (see the Overview tab)")
        box = QGroupBox(title)
        layout = QVBoxLayout(box)

        table = QTableWidget()
        table.setColumnCount(7)
        table.setHorizontalHeaderLabels([
            "Label",
            "Channel",
            "Partial wave",
            "B0 (G)",
            "Δ (G)",
            "a_bg (a0)",
            "Reference",
        ])
        table.setRowCount(len(resonances))
        table.setWordWrap(False)
        table.setTextElideMode(Qt.ElideNone)
        table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        table.setSizeAdjustPolicy(QTableWidget.AdjustToContents)

        for row_index, resonance in enumerate(resonances):
            values = [
                resonance.get("label", ""),
                resonance.get("channel", ""),
                resonance.get("partial_wave", ""),
                self._format_value(resonance.get("B0_G")),
                self._format_value(resonance.get("Delta_G")),
                self._format_value(resonance.get("a_bg_a0")),
                resonance.get("ref", ""),
            ]

            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setToolTip(str(value))
                table.setItem(row_index, column, item)

        table.resizeColumnsToContents()
        table.resizeRowsToContents()

        header_height = table.horizontalHeader().height()
        frame_height = table.frameWidth() * 2
        row_height = table.verticalHeader().defaultSectionSize()
        total_height = (
            header_height
            + frame_height
            + len(resonances) * row_height
            + 12
        )
        table.setMinimumHeight(total_height)
        table.setMaximumHeight(total_height)

        layout.addWidget(table)
        return box

    def refresh(self, root, species):
        self.root = Path(root)
        self.species = str(species or "").strip()

        if not self.species:
            self._empty_state()
            return

        files = {}
        errors = []
        data_json_dir = Workspace(root=self.root).data_json_dir
        for label, suffix in (
            ("core", ".json"),
            ("rydberg", "_rydberg.json"),
            ("transitions", "_transitions.json"),
            ("lifetimes", "_lifetimes.json"),
            ("polarizability", "_polarizability.json"),
            ("blackbody", "_blackbody.json"),
            ("tweezer", "_tweezer.json"),
            ("hyperfine", "_hyperfine.json"),
            ("feshbach", "_feshbach.json"),
            ("feshbach_result", "_feshbach_out.json"),
        ):
            path = data_json_dir / f"{self.species}{suffix}"
            if path.exists():
                data, error = self._load_json(path)
                if error:
                    errors.append((path.name, error))
                else:
                    files[label] = data

        self._clear_cards()
        self._summary_parts = []

        core = files.get("core", {})
        rydberg = files.get("rydberg", {})
        trans = files.get("transitions", {})
        lt = files.get("lifetimes", {})
        polar = files.get("polarizability", {})
        bb = files.get("blackbody", {})
        tw = files.get("tweezer", {})
        hf = files.get("hyperfine", {})
        fesh = files.get("feshbach_result", files.get("feshbach", {}))

        cards = [
            *([self._card("Unreadable files", errors)] if errors else []),
            self._make_species_card(core, rydberg),
            self._make_pipeline_card(self.root, self.species),
            self._make_rydberg_card(rydberg),
            self._make_transitions_card(trans),
            self._make_lifetimes_card(lt),
            self._make_polarizability_card(polar),
            self._make_blackbody_card(bb),
            self._make_tweezer_card(tw),
            self._make_hyperfine_card(hf),
            self._make_feshbach_card(fesh),
            self._make_feshbach_channels_card(fesh),
        ]

        for index, card in enumerate(cards):
            row = index // 2
            col = index % 2
            self._add_card(card, row, col)
