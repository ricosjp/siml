import pkg_resources

if 'torch-geometric' in [
        p.key for p in pkg_resources.working_set]:  # pylint: disable=E1133
    from .cluster_gcn import ClusterGCN  # NOQA
    from .gcnii import GCNII  # NOQA
    from .gin import GIN  # NOQA
