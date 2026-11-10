"""Help and pipeline reference dialog for the LiNaK desktop app."""

from __future__ import annotations

from html import escape
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QPushButton,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
)

from .project_io import open_pipeline_overview_html


class HelpDialog(QDialog):
    """Show generated CLI reference, stage explanations, and usage notes."""

    def __init__(self, root: Path, pipeline, stage_physics, parent=None):
        super().__init__(parent)
        self.root = root
        self.pipeline = pipeline
        self.stage_physics = stage_physics
        self.setWindowTitle("LiNaK Control Panel - Help")
        self.resize(800, 640)

        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        layout.addWidget(tabs)

        flags_view = QTextBrowser()
        flags_view.setHtml(self._flags_html())
        tabs.addTab(flags_view, "CLI flags reference")

        stages_view = QTextBrowser()
        stages_view.setHtml(self._stages_html())
        tabs.addTab(stages_view, "Pipeline stages")

        about_view = QTextBrowser()
        about_view.setHtml(self._about_html())
        tabs.addTab(about_view, "About / tips")

        button_row = QHBoxLayout()
        overview_button = QPushButton("Open full Pipeline Overview (HTML)...")
        overview_button.clicked.connect(
            lambda: open_pipeline_overview_html(self.root, self)
        )
        button_row.addWidget(overview_button)
        button_row.addStretch(1)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        button_row.addWidget(close_button)
        layout.addLayout(button_row)

    def _flags_html(self):
        parts = [
            "<h2>CLI flags, by pipeline stage</h2>",
            "<p>Every field in a stage's options form maps to one command-line "
            "flag of that pipeline. Leaving a field blank simply omits the flag, "
            "so the pipeline's own default applies.</p>",
        ]
        for stage in self.pipeline:
            parts.append(
                f"<h3>{escape(stage['label'])} &nbsp; "
                f"<code>linak --pipeline {escape(stage['pipeline'])}</code></h3>"
            )
            parts.append(f"<p>{escape(stage['desc'])}</p>")
            flags = stage.get("flags", [])
            if not flags:
                parts.append("<p><i>No configurable options for this stage.</i></p>")
                continue
            parts.append(
                "<table cellspacing='0' cellpadding='5' border='1' width='100%' "
                "style='border-collapse:collapse;'>"
                "<tr><th align='left'>Flag</th><th align='left'>Field</th>"
                "<th align='left'>Default</th><th align='left'>Meaning</th></tr>"
            )
            for spec in flags:
                kind = spec["kind"]
                default = spec.get("default", "")
                if callable(default):
                    default = default(self.root)
                if kind == "flag":
                    default_display = "off"
                elif kind == "invflag":
                    default_display = "on"
                elif kind == "choice":
                    default_display = escape(str(default))
                    for choice in spec["choices"]:
                        if isinstance(choice, tuple) and choice[0] == default:
                            default_display = escape(choice[1])
                            break
                else:
                    default_display = escape(str(default)) if default else "<i>(blank)</i>"
                help_text = spec.get("help") or spec["label"]
                parts.append(
                    f"<tr><td><code>{escape(spec['name'])}</code></td>"
                    f"<td>{escape(spec['label'])}</td><td>{default_display}</td>"
                    f"<td>{escape(help_text)}</td></tr>"
                )
            parts.append("</table>")
        return "".join(parts)

    def _stages_html(self):
        parts = [
            "<h2>What each stage computes</h2>",
            "<p>Short physics context for each pipeline stage; see the full "
            "Pipeline Overview document (button below) for full derivations.</p>",
        ]
        for stage in self.pipeline:
            parts.append(
                f"<h3>{escape(stage['label'])}</h3>"
                f"<p>{escape(self.stage_physics.get(stage['id'], ''))}</p>"
            )
        return "".join(parts)

    @staticmethod
    def _about_html():
        return """
        <h2>About this panel</h2>
        <p>This is a control panel over the LiNaK pipelines. It runs each
        stage as a <code>linak --pipeline</code> subprocess and displays the
        JSON, CSV, and HTML files it produces. It does not reimplement the
        physics. LiNaK is free software under the GNU GPL v3 or later.</p>
        <h3>Output folder</h3>
        <p>Results go to LiNaK's per-user data folder unless another folder is
        chosen with <i>File &gt; Open output folder...</i> (<i>Use default
        output folder</i> goes back). The bundled data and example results
        are copied into the folder in use; existing files are never
        overwritten, so after an update re-run a stage to refresh its plot
        or delete the old file first. A LiNaK source checkout keeps its
        outputs under <code>.linak/</code>.</p>
        <h3>Species IDs and new species</h3>
        <p>Bare element symbols such as <code>Na</code> and <code>Cs</code>
        represent neutral atoms. Ion IDs such as <code>Na_c1</code>,
        <code>Mg_c1</code>, and <code>Na_a1</code> are accepted when registered.
        Choosing an atom with no bundled data (for example Mg) opens a dialog
        to import its NIST levels and lines and type its numbers.
        <i>Edit &gt; Species properties...</i> shows one element's built-in
        values and data files and flags what is missing.</p>
        <h3>Grotrian diagram</h3>
        <p>The stage needs ORCA data (TD-DFT or SOC) for the species; run the
        ORCA stage first. The page shows levels in columns by orbital l, with
        fine-structure bars where J is known. Use the panel on the left to
        filter by spectral region, level class, f-value source and grade, n,
        wavelength, minimum f, and ground or excited lower level; click a
        section title to fold it. Click a line or table row to select it,
        shift-click to add, double-click to isolate, right-click for
        <i>View more</i> (every value for that transition, including the
        extrapolation model f with its 68% and 95% bands). Click a level or
        an ORCA marker to pin it. <i>Export</i> saves the diagram as SVG, PNG
        or PDF on a light background, or the matching transitions as CSV.
        ORCA markers are drawn at their exact energy, in the column of the
        nearest NIST multiplet.</p>
        <h3>Extrapolated f-values</h3>
        <p>For the supported alkalis the Lifetimes stage can extrapolate the
        ns&rarr;np oscillator strengths to high n (<code>--extrapolation</code>,
        set up in <i>Edit &gt; Preferences &gt; Extrapolator</i>). <i>Show</i>
        records them next to the measured values; <i>Use</i> fills the gaps,
        tagged <code>extrapolated</code>, and the polarizability and blackbody
        stages follow. Run the later stages with the same setting.</p>
        <h3>Tables and variables</h3>
        <p><i>Edit &gt; Preferences &gt; Tables and variables</i> lists the
        built-in constants and data files. Edits are kept in the
        <code>tables/</code> folder of the output folder and never replace
        the bundled values; Restore Defaults brings them back. Add... and
        Remove work on the open lists.</p>
        <h3>Save as</h3>
        <p>The <i>Save as</i> field under a stage's output options files that
        run's results under a name, so another run with different settings
        does not overwrite them.</p>
        <h3>Offline plot rendering</h3>
        <p>Plots render offline with the bundled plotly.js. The GUI never
        modifies the report files in the output folder. Exports from the
        viewer ask where to save.</p>
        <h3>ORCA stages</h3>
        <p>The ORCA stages require ORCA to be installed and callable as
        <code>orca</code> on the system PATH, or named by the
        <code>ORCA_EXE</code> environment variable. SOC and NEVPT2 calculations can
        take a long time. Use Stop if a running subprocess must be cancelled.</p>
        <h3>Console, find and tests</h3>
        <p>The command preview shows the command that will be executed. The
        Console tab shows stdout and stderr from the running subprocess.
        Ctrl+F opens a find bar in the JSON viewer, the data editor and the
        tables page. <i>Run &gt; Tests...</i> runs the test suite with
        progress and details.</p>
        <h3>Custom dataset editor</h3>
        <p>The Edit menu provides editors for JSON and CSV datasets. JSON is
        validated before saving, and CSV headers and row widths are checked.</p>
        <h3>Licence and credits</h3>
        <p>GPL-3.0-or-later. Plotly.js (MIT) is bundled for the offline plots.
        ARC (BSD-3-Clause) is an optional dependency. ORCA is separate software
        with its own licence. Cite LiNaK and your data sources (NIST ASD and
        others); see <code>REFERENCES.md</code>.</p>
        """
