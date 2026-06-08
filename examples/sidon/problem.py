def generate_sidon_set() -> list[int]:
    """
    Return a Sidon set: a list of distinct integers in [1, 10000]
    where all pairwise sums a+b (with a <= b) are unique.
    Maximize the size of the returned list.
    """
    # crucible:region start name=solution
    return [1, 2, 5, 11]
    # crucible:region end
