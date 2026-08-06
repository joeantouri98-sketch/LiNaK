"""
find_fr_basis.py — finds which basis sets work for Fr in your ORCA build
Run from your Orca working directory: python find_fr_basis.py
"""
import subprocess, os, tempfile

ORCA = r"C:\ORCA_6.1.1\orca.exe"

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

def test_basis(basis):
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
        r = subprocess.run(f'"{ORCA}" {fname} > {outname} 2>&1',
                           shell=True, timeout=60)
        # Read output and check for basis error vs actual SCF attempt
        with open(outname, errors='ignore') as f:
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
            return f'UNKNOWN (exit {r.returncode})'
    except subprocess.TimeoutExpired:
        return 'TIMEOUT (probably running = WORKS)'
    finally:
        for fn in [fname, outname,
                   fname.replace('.inp', '.gbw'),
                   fname.replace('.inp', '.prop')]:
            try: os.remove(fn)
            except: pass

print(f"Testing basis sets for Fr in ORCA 6.1.1")
print(f"{'Basis':30s}  Result")
print("-" * 55)

working = []
for b in CANDIDATES:
    result = test_basis(b)
    print(f"  {b:30s}  {result}")
    if 'WORKS' in result or 'TIMEOUT' in result:
        working.append(b)

print()
if working:
    print(f"Working basis sets for Fr: {working}")
    print(f"\nRecommended: {working[0]}")
else:
    print("No working basis found — check ORCA path or try orca_exportbasis.exe Fr")
