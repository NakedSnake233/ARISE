"""Load the declared exact Drain kernel; never change the frozen parser package."""
from __future__ import annotations
from contextlib import contextmanager
import importlib.util
import sysconfig

from .common import ROOT

def artifact():
    return ROOT / "cache/re2_drain_kernel" / ("_arise_drain" + sysconfig.get_config_var("EXT_SUFFIX"))


def kernel():
    path = artifact(); spec = importlib.util.spec_from_file_location("_arise_drain", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


@contextmanager
def accelerated():
    from drain3.drain import Drain
    original = Drain.fast_match; compiled = kernel()

    def fast_match(self, cluster_ids, tokens, sim_th, include_params):
        if type(self.id_to_cluster) is not dict:
            return original(self, cluster_ids, tokens, sim_th, include_params)
        return compiled.fast_match(cluster_ids, tokens, self.id_to_cluster, self.param_str, sim_th, include_params)

    Drain.fast_match = fast_match
    try: yield
    finally: Drain.fast_match = original
