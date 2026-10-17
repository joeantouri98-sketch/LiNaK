"""Starter templates used by the GUI dataset editor."""

JSON_DATASET_TEMPLATES = {
    "new": "{}",
    "feshbach": '{\n  "element": "",\n  "isotope": null,\n  "a_bg": null,\n  "resonances": [\n    {\n      "B0": null,\n      "Delta": null,\n      "width": null,\n      "channel": ""\n    }\n  ]\n}',
    "transitions": '{\n  "element": "",\n  "source": "manual",\n  "transitions": [\n    {\n      "lower_state": "",\n      "upper_state": "",\n      "wavelength_nm": null,\n      "oscillator_strength": null\n    }\n  ]\n}',
    "lifetimes": '{\n  "element": "",\n  "states": [\n    {\n      "state": "",\n      "lifetime_ns": null,\n      "A_total_s_inv": null\n    }\n  ]\n}',
    "polarizability": '{\n  "element": "",\n  "state": "",\n  "wavelength_nm": [],\n  "alpha_au": []\n}',
    "blackbody": '{\n  "element": "",\n  "temperature_K": null,\n  "states": [\n    {\n      "state": "",\n      "shift_Hz": null,\n      "rate_s_inv": null\n    }\n  ]\n}',
    "hyperfine": '{\n  "element": "",\n  "isotope": null,\n  "states": [\n    {\n      "label": "",\n      "I": null,\n      "J": null,\n      "F": null,\n      "A_MHz": null,\n      "B_MHz": null\n    }\n  ]\n}',
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
