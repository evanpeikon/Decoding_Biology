def nucleotide_counter(dna_seq):
    base_count = {'A': 0, 'T': 0, 'C': 0, 'G': 0}
    for base in dna_seq:
        if base not in base_count:
            raise ValueError(f"Invalid base encountered: {base}")  # Raise error for invalid base
        base_count[base] += 1
    return base_count

def test_nucleotide_counter():
    # Test case 1: Typical input
    assert nucleotide_counter("ATCGATCG") == {'A': 2, 'T': 2, 'C': 2, 'G': 2}, "Test case 1 failed"

    # Test case 2: Edge case with empty input
    assert nucleotide_counter("") == {'A': 0, 'T': 0, 'C': 0, 'G': 0}, "Test case 2 failed"

    # Test case 3: Input with unexpected characters
    try:
        nucleotide_counter("ATCX")
        raise AssertionError("Test case 3 failed - function did not handle invalid input")
    except ValueError:
        pass  # Expected behavior since 'X' is not a valid base

# Run the tests
test_nucleotide_counter()
