<script>const HF_TABLE_DATA = $!tbl_rows_js;const HF_CLOCK = $!clock_js;
        function hfFmt(n, digits) {
          if (n === null || n === undefined) return '-';
          return Number(n).toLocaleString(undefined, {
            minimumFractionDigits: digits, maximumFractionDigits: digits
          });
        }
        function hfBuildTable() {
          const panel = document.getElementById('hf-const-overlay');
          let html = '<table><thead><tr>'
            + '<th>State</th><th>A (MHz)</th><th>B (MHz)</th><th>C (kHz)</th>'
            + '<th>g_J</th><th>g_J source</th><th>g_F per F</th>'
            + '</tr></thead><tbody>';
          HF_TABLE_DATA.forEach(r => {
            html += `<tr><td class="state-col">${r.state}</td>`
              + `<td>${r.A}</td><td>${r.B}</td><td>${r.C}</td>`
              + `<td>${r.gJ}</td><td>${r.src}</td><td>${r.gF}</td></tr>`;
          });
          html += '</tbody></table>';
          panel.innerHTML = html;
        }
        function hfBuildClock() {
          const panel = document.getElementById('hf-clock-overlay');
          if (!HF_CLOCK.present) {
            panel.innerHTML = '<div class="hf-clock-note">No ground-state '
              + 'microwave clock interval (need I>0 and at least two F levels).</div>';
            return;
          }
          panel.innerHTML = `
            <div class="hf-clock-card">
              <div class="hf-clock-box accent">
                <div class="hf-clock-k">${HF_CLOCK.isotope} · ${HF_CLOCK.state}
                  · F=${HF_CLOCK.F_upper} ↔ F=${HF_CLOCK.F_lower}</div>
                <div class="hf-clock-v gold">${hfFmt(HF_CLOCK.frequency_MHz, 6)} MHz</div>
              </div>
              <div class="hf-clock-box">
                <div class="hf-clock-k">Frequency (GHz)</div>
                <div class="hf-clock-v">${hfFmt(HF_CLOCK.frequency_GHz, 9)} GHz</div>
              </div>
              <div class="hf-clock-box">
                <div class="hf-clock-k">Frequency (Hz)</div>
                <div class="hf-clock-v">${hfFmt(HF_CLOCK.frequency_Hz, 0)} Hz</div>
              </div>
              <div class="hf-clock-box">
                <div class="hf-clock-k">Wavelength</div>
                <div class="hf-clock-v">${hfFmt(HF_CLOCK.wavelength_cm, 4)} cm</div>
              </div>
            </div>
            <div class="hf-clock-note">${HF_CLOCK.note}
              Stored in data_json as special.microwave_clock. The m_F=0
              component is first-order B-insensitive (clock transition);
              full Zeeman / Breit-Rabi curves are in the companion zeeman HTML.</div>`;
        }
        function hfOnToggleChange(value) {
          const gd         = document.querySelector('.plotly-graph-div');
          const constPanel = document.getElementById('hf-const-overlay');
          const clockPanel = document.getElementById('hf-clock-overlay');
          const showBar = (value === 'splittings');
          Plotly.restyle(gd, {visible: showBar}, [$!bar_trace_idx]);
          constPanel.style.display = (value === 'constants') ? 'block' : 'none';
          clockPanel.style.display = (value === 'clock') ? 'block' : 'none';
        }
        hfBuildTable();
        hfBuildClock();
        </script>