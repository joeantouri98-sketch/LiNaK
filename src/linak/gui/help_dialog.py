"""Help and pipeline reference dialog for the LiNaK desktop app."""

from __future__ import annotations

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
            "flag of that script. Leaving a field blank simply omits the flag, "
            "so the script's own default applies.</p>",
        ]
        for stage in self.pipeline:
            parts.append(
                f"<h3>{stage['label']} &nbsp; <code>{stage['script']}</code></h3>"
            )
            parts.append(f"<p>{stage['desc']}</p>")
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
                if kind == "flag":
                    default_display = "off"
                elif kind == "choice":
                    default_display = default
                    for choice in spec["choices"]:
                        if isinstance(choice, tuple) and choice[0] == default:
                            default_display = choice[1]
                            break
                else:
                    default_display = default if default else "<i>(blank)</i>"
                help_text = spec.get("help") or spec["label"]
                parts.append(
                    f"<tr><td><code>{spec['name']}</code></td>"
                    f"<td>{spec['label']}</td><td>{default_display}</td>"
                    f"<td>{help_text}</td></tr>"
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
                f"<h3>{stage['label']}</h3>"
                f"<p>{self.stage_physics.get(stage['id'], '')}</p>"
            )
        return "".join(parts)

    @staticmethod
    def _about_html():
        return """
        <h2>About this panel</h2>
        <p>This is a thin control panel over your existing LiNaK scripts. It
        runs them as subprocesses and displays the JSON, CSV, and HTML files
        they produce. It does not reimplement the physics.</p>
        <h3>Species IDs</h3>
        <p>Bare element symbols such as <code>Na</code> and <code>Cs</code>
        represent neutral atoms. Ion IDs such as <code>Na_c1</code>,
        <code>Mg_c1</code>, and <code>Na_a1</code> are accepted when registered.</p>
        <h3>Offline plot rendering</h3>
        <p>Plots can be rendered using the bundled local
        <code>vendor/plotly.min.js</code> file. The GUI does not modify original
        files under <code>resources/plots/</code>.</p>
        <h3>ORCA stages</h3>
        <p>The ORCA stages require ORCA to be installed and callable as
        <code>orca</code> on the system PATH. SOC and NEVPT2 calculations can
        take a long time. Use Stop if a running subprocess must be cancelled.</p>
        <h3>Grotrian diagram prerequisite</h3>
        <p>The Grotrian diagram requires ORCA data to already exist for the
        selected species. Run the ORCA stage first.</p>
        <h3>Console and command preview</h3>
        <p>The command preview shows the command that will be executed. The
        Console tab shows stdout and stderr from the running subprocess.</p>
        <h3>Custom dataset editor</h3>
        <p>The Edit menu provides editors for JSON and CSV datasets. JSON is
        validated before saving, and CSV headers and row widths are checked.</p>
        """
