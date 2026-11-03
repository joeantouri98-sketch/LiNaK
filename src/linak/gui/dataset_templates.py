"""Starter templates used by the GUI dataset editor."""

JSON_DATASET_TEMPLATES = {
    "new": "{}",
    # Input schema read by the feshbach pipeline (<El>_feshbach.json).
    "feshbach": '{\n  "element": "",\n  "isotope_A": null,\n  "a_bg_default_a0": null,\n  "resonances": [\n    {\n      "label": "",\n      "channel": "",\n      "partial_wave": "s",\n      "B0_G": null,\n      "Delta_G": null,\n      "a_bg_a0": null,\n      "ref": ""\n    }\n  ]\n}',
    "transitions": '{\n  "element": "",\n  "source": "manual",\n  "transitions": [\n    {\n      "lower_state": "",\n      "upper_state": "",\n      "wavelength_nm": null,\n      "oscillator_strength": null\n    }\n  ]\n}',
    "lifetimes": '{\n  "element": "",\n  "states": [\n    {\n      "state": "",\n      "lifetime_ns": null,\n      "A_total_s_inv": null\n    }\n  ]\n}',
    "polarizability": '{\n  "element": "",\n  "state": "",\n  "wavelength_nm": [],\n  "alpha_au": []\n}',
    "blackbody": '{\n  "element": "",\n  "temperature_K": null,\n  "states": [\n    {\n      "state": "",\n      "shift_Hz": null,\n      "rate_s_inv": null\n    }\n  ]\n}',
    # Input schema read by the hyperfine pipeline (<El>_hf_constants.json):
    # one entry per state label.
    "hyperfine": '{\n  "_source": "",\n  "_isotope": "",\n  "3s1/2": {\n    "A_MHz": null,\n    "B_MHz": 0.0,\n    "C_kHz": 0.0\n  }\n}',
    "rydberg": '{\n  "element": "",\n  "levels": [\n    {\n      "state": "",\n      "n": null,\n      "l": null,\n      "energy_cm1": null\n    }\n  ]\n}',
}

CSV_DATASET_TEMPLATES = {
    "new": [["column_1", "column_2"], ["", ""]],
    "lines": [["element", "isotope", "lower_state", "upper_state", "wavelength_nm", "oscillator_strength"], ["", "", "", "", "", ""]],
    "levels": [["element", "isotope", "state", "energy_cm1", "uncertainty_cm1"], ["", "", "", "", ""]],
    "nuclear_data": [["element", "isotope", "nuclear_spin_I", "magnetic_moment", "quadrupole_moment"], ["", "", "", "", ""]],
    "transitions": [["element", "isotope", "lower_state", "upper_state", "wavelength_nm", "oscillator_strength"], ["", "", "", "", "", ""]],
    "lifetimes": [["element", "state", "lifetime_ns", "A_total_s_inv"], ["", "", "", ""]],
}
