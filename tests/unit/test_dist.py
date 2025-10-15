from mfcl.utils import dist as mdist


def test_dist_single_process_defaults():
    assert mdist.is_main_process()
    assert mdist.get_rank() == 0
    assert mdist.get_world_size() == 1
    # barrier should be no-op
    mdist.barrier()
