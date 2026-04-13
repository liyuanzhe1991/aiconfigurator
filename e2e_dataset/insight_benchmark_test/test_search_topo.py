from kunlun_commons.machines.info import GB200, A100
from insight_benchmark.api.topo import search_device_from_topology


def test_search_topo():
    topo = search_device_from_topology(GB200.AG88N4C2T2, 3)
    assert len(topo["selected_devices"]) == 4

    # scale out
    topo = search_device_from_topology(GB200.AG88N4C2T2, 8)
    assert len(topo["selected_devices"]) == 7

    topo = search_device_from_topology(A100.TG78M2V7X6_FORK, 4)
    assert len(topo["selected_devices"]) == 8

    topo = search_device_from_topology(A100.TG78M2V7X6_FORK, 6)
    assert len(topo["selected_devices"]) == 11


if __name__ == "__main__":
    test_search_topo()
