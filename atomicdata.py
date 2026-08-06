#!/usr/bin/env python3
"""
Atomic Data from Mendeleev and Astropy - FIXED VERSION

Shows how to extract atomic data with proper error handling.

Usage: python atomicdata.py <ELEMENT>
"""

import sys
import json
import os

# ============================================================================
# MENDELEEV Package Demo (FIXED)
# ============================================================================

def demo_mendeleev(element_symbol):
    """
    Mendeleev package - with proper error handling
    """
    
    print("\n" + "="*70)
    print("MENDELEEV PACKAGE")
    print("="*70)
    
    try:
        import mendeleev
    except ImportError:
        print("❌ Mendeleev not installed")
        print("   Install with: pip install mendeleev")
        return None
    
    try:
        # Get element
        element = mendeleev.element(element_symbol)
        
        print(f"\n✓ Found data for {element.name} ({element.symbol})")
        print(f"  Atomic number: {element.atomic_number}")
        print(f"  Atomic mass: {element.atomic_weight:.3f} u")
        
        # Electronic configuration (handle different attribute names)
        config = None
        if hasattr(element, 'electronic_configuration'):
            config = element.electronic_configuration
        elif hasattr(element, 'ec'):
            config = element.ec
        elif hasattr(element, 'econf'):
            config = element.econf
        
        if config:
            print(f"  Ground state: {config}")
        else:
            print(f"  Ground state: Not available")
        
        # Periodic table position
        print(f"\n📍 Periodic Table Position:")
        print(f"  Period: {element.period}")
        print(f"  Group: {element.group_id if element.group_id else 'N/A'}")
        print(f"  Block: {element.block}")
        
        # Classify element type
        element_type = classify_element(element)
        print(f"  Type: {element_type}")
        
        # Ionization energies (multiple)
        print(f"\n📊 Ionization Energies:")
        if hasattr(element, 'ionenergies') and element.ionenergies:
            ies = element.ionenergies
            for i in sorted(ies.keys())[:5]:  # First 5
                print(f"  IE{i}: {ies[i]:.3f} eV")
        else:
            print("  No ionization energy data")
        
        # Electron affinity
        if hasattr(element, 'electron_affinity') and element.electron_affinity:
            print(f"\n⚡ Electron affinity: {element.electron_affinity:.3f} eV")
        
        # Other properties
        print(f"\n🔬 Physical Properties:")
        if element.atomic_radius:
            print(f"  Atomic radius: {element.atomic_radius} pm")
        if element.en_pauling:
            print(f"  Electronegativity (Pauling): {element.en_pauling}")
        if element.melting_point:
            print(f"  Melting point: {element.melting_point:.1f} K")
        if element.boiling_point:
            print(f"  Boiling point: {element.boiling_point:.1f} K")
        if element.density:
            print(f"  Density: {element.density:.3f} g/cm³")
        
        # Get first ionization energy for our purposes
        ie = None
        if hasattr(element, 'ionenergies') and element.ionenergies:
            ie = element.ionenergies.get(1)
        
        return {
            'element': element.symbol,
            'name': element.name,
            'atomic_number': element.atomic_number,
            'ionization_energy_eV': ie,
            'configuration': config,
            'group': element.group_id,
            'period': element.period,
            'block': element.block,
            'type': element_type,
            'source': 'Mendeleev package'
        }
        
    except Exception as e:
        print(f"❌ Error with Mendeleev: {e}")
        import traceback
        traceback.print_exc()
        return None


def classify_element(element):
    """Classify element into categories."""
    
    # Alkali metals (Group 1, excluding H)
    if element.group_id == 1 and element.atomic_number > 1:
        return "Alkali Metal (Group 1)"
    
    # Alkaline earth metals (Group 2)
    if element.group_id == 2:
        return "Alkaline Earth Metal (Group 2)"
    
    # Transition metals (Groups 3-12)
    if element.group_id and 3 <= element.group_id <= 12:
        return f"Transition Metal (Group {element.group_id})"
    
    # Halogens (Group 17)
    if element.group_id == 17:
        return "Halogen (Group 17)"
    
    # Noble gases (Group 18)
    if element.group_id == 18:
        return "Noble Gas (Group 18)"
    
    # p-block elements
    if element.block == 'p':
        if element.group_id:
            return f"p-block Element (Group {element.group_id})"
        else:
            return "p-block Element"
    
    # s-block
    if element.block == 's':
        return "s-block Element"
    
    # d-block
    if element.block == 'd':
        return "d-block Element"
    
    # f-block
    if element.block == 'f':
        if element.atomic_number in range(57, 72):
            return "Lanthanide (f-block)"
        elif element.atomic_number in range(89, 104):
            return "Actinide (f-block)"
        else:
            return "f-block Element"
    
    return "Unknown"


# ============================================================================
# PERIODICTABLE Package Demo (FIXED)
# ============================================================================

def demo_periodictable(element_symbol):
    """
    periodictable package - FIXED
    """
    
    print("\n" + "="*70)
    print("PERIODICTABLE PACKAGE")
    print("="*70)
    
    try:
        import periodictable
    except ImportError:
        print("❌ periodictable not installed")
        print("   Install with: pip install periodictable")
        return None
    
    try:
        # Get element - handle case sensitivity
        # periodictable uses element name or symbol
        try:
            # Try as attribute (e.g., periodictable.Br)
            element = getattr(periodictable, element_symbol)
        except AttributeError:
            # Try lowercase (e.g., periodictable.br)
            try:
                element = getattr(periodictable, element_symbol.lower())
            except AttributeError:
                # Try by symbol lookup
                element = periodictable.elements.symbol(element_symbol)
        
        print(f"\n✓ Found data for {element.name} ({element.symbol})")
        print(f"  Atomic number: {element.number}")
        print(f"  Atomic mass: {element.mass:.3f} u")
        
        if element.density:
            print(f"  Density: {element.density:.3f} g/cm³")
        
        # Ions
        if hasattr(element, 'ions') and element.ions:
            print(f"\n⚛️  Common ions:")
            for ion in element.ions:
                print(f"  {element.symbol}{ion:+d}")
        
        return {
            'element': element.symbol,
            'atomic_number': element.number,
            'mass': element.mass,
            'source': 'periodictable package'
        }
        
    except Exception as e:
        print(f"❌ Error with periodictable: {e}")
        return None


# ============================================================================
# ASTROPY Package Demo (FIXED)
# ============================================================================

def demo_astropy(element_symbol):
    """
    Astropy - FIXED
    """
    
    print("\n" + "="*70)
    print("ASTROPY PACKAGE")
    print("="*70)
    
    try:
        from astropy import units as u
        from astropy import constants as const
    except ImportError:
        print("❌ Astropy not installed")
        print("   Install with: pip install astropy")
        return None
    
    try:
        print(f"\n✓ Astropy loaded successfully")
        print(f"\n📐 Physical Constants Available:")
        
        # Rydberg constant (FIXED conversion)
        rydberg_hz = const.Ryd * const.c  # Convert to Hz
        rydberg_ev = (rydberg_hz * const.h).to(u.eV)
        print(f"  Rydberg constant: {rydberg_ev.value:.6f} eV")
        
        print(f"  Electron mass: {const.m_e}")
        print(f"  Planck constant: {const.h}")
        print(f"  Speed of light: {const.c}")
        
        # Example wavelength to energy conversion
        print(f"\n📊 Example Unit Conversion:")
        wavelength = 589.3 * u.nm
        energy = wavelength.to(u.eV, equivalencies=u.spectral())
        print(f"  589.3 nm → {energy.value:.3f} eV (Na D-line)")
        
        print(f"\n⚠️  Astropy doesn't have atomic level database")
        print(f"   Best for: constants, units, spectral conversions")
        
        return {
            'package': 'astropy',
            'rydberg_eV': rydberg_ev.value
        }
        
    except Exception as e:
        print(f"❌ Error with Astropy: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# PYVALEM Package Demo
# ============================================================================

def demo_pyvalem(element_symbol):
    """PyValem - atomic notation parser"""
    
    print("\n" + "="*70)
    print("PYVALEM PACKAGE")
    print("="*70)
    
    try:
        from pyvalem.states import AtomicTermSymbol
    except ImportError:
        print("❌ PyValem not installed")
        print("   Install with: pip install pyvalem")
        return None
    
    try:
        print(f"\n✓ PyValem loaded")
        print(f"\n📝 PyValem parses spectroscopic notation:")
        
        examples = ['2P_3/2', '1S_0', '3P_2', '4S_3/2']
        for ex in examples:
            try:
                term = AtomicTermSymbol(ex)
                print(f"  {ex:10} → S={term.S}, L={term.L}, J={term.J}")
            except:
                pass
        
        print(f"\n⚠️  PyValem is for notation parsing only")
        
        return None
        
    except Exception as e:
        print(f"❌ Error with PyValem: {e}")
        return None


# ============================================================================
# RECOMMENDATION ENGINE
# ============================================================================

def get_recommendations(element_data):
    """Provide specific recommendations based on element type."""
    
    print("\n" + "="*70)
    print("💡 RECOMMENDATIONS FOR YOUR ELEMENT")
    print("="*70)
    
    if not element_data or 'type' not in element_data:
        print("\n⚠️  Element type unknown")
        return
    
    element_type = element_data['type']
    element_symbol = element_data['element']
    group = element_data.get('group')
    
    print(f"\nElement: {element_symbol} - {element_data.get('name', 'Unknown')}")
    print(f"Type: {element_type}")
    
    # Alkali metals
    if "Alkali Metal" in element_type:
        print(f"\n✅ EXCELLENT! Alkali atoms work perfectly with quantum defect theory")
        print(f"   Recommended approach:")
        print(f"   1. python generate_rydberg_series.py {element_symbol}")
        print(f"   2. python plotryd.py {element_symbol}")
        print(f"   This gives you the complete Rydberg series analytically!")
    
    # Alkaline earth metals
    elif "Alkaline Earth" in element_type:
        print(f"\n⚠️  Alkaline earth metals have 2 valence electrons")
        print(f"   Quantum defect theory is complex (multiple series)")
        print(f"   Recommended approaches:")
        print(f"   1. TDDFT (ORCA) for low-lying excited states")
        print(f"   2. NIST manual download for experimental data")
        print(f"   3. MCSCF/CASSCF for accurate multi-electron states")
    
    # Halogens
    elif "Halogen" in element_type:
        print(f"\n⚠️  Halogens have complex electronic structure (7 valence e⁻)")
        print(f"   Quantum defect theory won't work well")
        print(f"   Recommended approaches:")
        print(f"   1. NIST manual download (has extensive halogen data)")
        print(f"   2. TDDFT (ORCA) for excited states")
        print(f"   3. Use built-in NIST data if available:")
        
        from fetch_nist import NIST_DATA
        if element_symbol in NIST_DATA:
            print(f"      ✓ {element_symbol} has built-in NIST data!")
            print(f"      Run: python fetch_nist_working.py {element_symbol}")
        else:
            print(f"      ✗ No built-in data for {element_symbol}")
    
    # Transition metals
    elif "Transition Metal" in element_type:
        print(f"\n❌ Transition metals are very complex (d-electrons)")
        print(f"   Quantum defect theory DOES NOT WORK")
        print(f"   Recommended approaches:")
        print(f"   1. TDDFT (ORCA) - but may struggle with d-d transitions")
        print(f"   2. CASSCF/CASPT2 for accurate multi-reference states")
        print(f"   3. NIST manual download for experimental data")
        print(f"   4. Ligand field theory for coordination complexes")
    
    # Noble gases
    elif "Noble Gas" in element_type:
        print(f"\n⚠️  Noble gases have closed-shell ground states")
        print(f"   Excitations involve core electrons - complex!")
        print(f"   Recommended approaches:")
        print(f"   1. NIST manual download for experimental data")
        print(f"   2. High-level TDDFT or EOM-CCSD")
        print(f"   3. Experimental spectroscopy data")
    
    # p-block elements
    elif "p-block" in element_type:
        print(f"\n⚠️  p-block elements have multiple valence electrons")
        print(f"   Complexity depends on specific element")
        print(f"   Recommended approaches:")
        print(f"   1. NIST manual download (often has data)")
        print(f"   2. TDDFT (ORCA) for excited states")
        
        from fetch_nist_working import NIST_DATA
        if element_symbol in NIST_DATA:
            print(f"      ✓ {element_symbol} has built-in NIST data!")
            print(f"      Run: python fetch_nist_working.py {element_symbol}")
    
    # f-block
    elif "f-block" in element_type or "Lanthanide" in element_type or "Actinide" in element_type:
        print(f"\n❌ Lanthanides/Actinides are EXTREMELY complex (f-electrons)")
        print(f"   Quantum defect theory DOES NOT WORK")
        print(f"   Even TDDFT struggles with f-f transitions")
        print(f"   Recommended approaches:")
        print(f"   1. Experimental data only (NIST manual)")
        print(f"   2. Multi-reference methods (CASSCF/NEVPT2)")
        print(f"   3. Crystal field theory for solid-state")
    
    # Default
    else:
        print(f"\n⚠️  Element classification unclear")
        print(f"   Try NIST manual download or TDDFT")


# ============================================================================
# MAIN
# ============================================================================

def main():
    
    if len(sys.argv) < 2:
        print("""
Usage: python atomicdata_fixed.py <ELEMENT>

This script shows atomic data from multiple packages
and provides specific recommendations for your element.

Examples:
  python atomicdata_fixed.py Na    # Alkali - quantum defect works!
  python atomicdata_fixed.py Br    # Halogen - need NIST or TDDFT
  python atomicdata_fixed.py Fe    # Transition metal - very complex
""")
        sys.exit(1)
    
    element = sys.argv[1].strip()
    
    print("="*70)
    print(f"COMPREHENSIVE ATOMIC DATA - {element}")
    print("="*70)
    
    # Get data from all sources
    mendeleev_data = demo_mendeleev(element)
    pt_data = demo_periodictable(element)
    astropy_data = demo_astropy(element)
    pyvalem_data = demo_pyvalem(element)
    
    # Provide recommendations
    get_recommendations(mendeleev_data)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if mendeleev_data:
        print(f"Element: {mendeleev_data['element']} - {mendeleev_data['name']}")
        print(f"Type: {mendeleev_data['type']}")
        print(f"Group: {mendeleev_data['group']}")
        print(f"Period: {mendeleev_data['period']}")
        print(f"Block: {mendeleev_data['block']}")
        if mendeleev_data['ionization_energy_eV']:
            print(f"Ionization energy: {mendeleev_data['ionization_energy_eV']:.3f} eV")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()