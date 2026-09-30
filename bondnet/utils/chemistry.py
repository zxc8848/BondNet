"""
Valence tables and chemistry utilities for BondNet.
"""

# Standard organic valence (most common for drug-like molecules)
VALENCE_TABLE = {
    1:  1,   # H
    5:  3,   # B
    6:  4,   # C
    7:  3,   # N
    8:  2,   # O
    9:  1,   # F
    14: 4,   # Si
    15: 3,   # P
    16: 2,   # S
    17: 1,   # Cl
    35: 1,   # Br
    53: 1,   # I
}

# Aromatic contribution per bond type (for valence checking)
BOND_ORDER_MAP = {0: 1, 1: 2, 2: 3, 3: 1.5}  # single/double/triple/aromatic

# Atomic number to symbol
ELEM_SYMBOLS = {
    1: 'H', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F',
    14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I',
}


def get_expected_valence(atomic_number: int) -> int:
    """Return typical valence for the given atomic number in organic chemistry."""
    return VALENCE_TABLE.get(int(atomic_number), 4)


def get_valences_for_atoms(elems) -> list:
    """
    Return expected valence for each atom in the list/tensor of atomic numbers.

    Args:
        elems: iterable of atomic numbers (int or tensor).

    Returns:
        list of int valences.
    """
    return [get_expected_valence(int(z)) for z in elems]
