"""
basis_probe.py — finds which basis sets work for Fr in your ORCA build
Run from your ORCA working directory: python tools/basis_probe.py

SECURITY: Uses argument lists and resolves ORCA from
          ORCA_EXE env var or system PATH.
"""
import os
import subprocess
import tempfile

from linak.integrations.orca_runner import resolve_orca_executable

CANDIDATES = [
    "def2-TZVP",
    "def2-SVP",
    "SARC-DKH-TZVP",
    "SARC-ZORA-TZVP",
    "SARC2-DKH-QZVP",
    "SARC2-ZORA-QZVP",
    "x2c-TZVPall",
    "x2c-SVPall",
    "x2c-TZVPall-2c",
    "ZORA-def2-TZVP",
    "dhf-TZVP",
    "dhf-TZVPP",
    "dhf-SVP",
    "jorge-TZP-DKH",
    "jorge-DZP-DKH",
]


def test_basis(basis, orca_executable):
    inp = f"""! UKS PBE TightSCF

%basis
  NewGTO Fr "{basis}" end
end

%scf
  MaxIter 3
end

* xyz 0 2
Fr 0.0 0.0 0.0
*
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.inp',
                                     delete=False, dir='.') as f:
        f.write(inp)
        fname = f.name

    outname = fname.replace('.inp', '.out')
    try:
        with open(outname, 'w', encoding='utf-8', errors='ignore') as out_handle:
            result = subprocess.run(
                [orca_executable, fname],
                shell=False,
                stdout=out_handle,
                stderr=subprocess.STDOUT,
                timeout=60,
                check=False,
            )
        
        with open(outname, encoding='utf-8', errors='ignore') as f:
            out = f.read()
        
        if 'Basis not recognized' in out or 'not available for this element' in out:
            return 'NOT AVAILABLE'
        elif 'UNRECOGNIZED OR DUPLICATED KEYWORD' in out:
            return 'KEYWORD ERROR'
        elif 'INPUT ERROR' in out:
            return 'INPUT ERROR'
        elif 'SCF CONVERGED' in out or 'TOTAL SCF ENERGY' in out or 'MaxIter' in out:
            return 'WORKS'
        elif 'BASIS SET INFORMATION' in out or 'Contracted Basis' in out:
            return 'WORKS (basis loaded)'
        else:
            return f'UNKNOWN (exit {result.returncode})'
    except subprocess.TimeoutExpired:
        return 'TIMEOUT (probably running = WORKS)'
    finally:
        for fn in [fname, outname,
                   fname.replace('.inp', '.gbw'),
                   fname.replace('.inp', '.prop')]:
            try:
                os.remove(fn)
            except OSError:
                pass

def main():
    orca_executable = resolve_orca_executable()
    print("Testing basis sets for Fr in ORCA")
    print(f"ORCA executable: {orca_executable}")
    print(f"{'Basis':30s}  Result")
    print("-" * 55)

    working = []
    for basis in CANDIDATES:
        result = test_basis(basis, orca_executable)
        print(f"  {basis:30s}  {result}")
        if "WORKS" in result or "TIMEOUT" in result:
            working.append(basis)

    print()
    if working:
        print(f"Working basis sets for Fr: {working}")
        print(f"\nRecommended: {working[0]}")
    else:
        print("No working basis found - check ORCA path or try orca_exportbasis.exe Fr")


if __name__ == "__main__":
    main()