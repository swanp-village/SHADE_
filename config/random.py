from numpy.random import PCG64DXSM, Generator, SeedSequence


def get_differential_evolution_rng(seedsequence: SeedSequence) -> Generator:
    return Generator(PCG64DXSM(seedsequence).jumped(1))


def get_multi_start_local_search_rng(seedsequence: SeedSequence) -> Generator:
    return Generator(PCG64DXSM(seedsequence).jumped(2))


def get_ring_rng(seedsequence: SeedSequence) -> Generator:
    return Generator(PCG64DXSM(seedsequence).jumped(3))


def get_analyzer_rng(seedsequence: SeedSequence, jumps: int) -> Generator:
    return Generator(PCG64DXSM(seedsequence).jumped(4 * jumps))
