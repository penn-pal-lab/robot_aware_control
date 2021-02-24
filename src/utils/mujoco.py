"""Module for caching Python modules related to simulation."""

import sys
import os

_MUJOCO_PY_MODULE = None

_GLFW_MODULE = None


def get_mujoco_py():
    """Returns the mujoco_py module."""
    global _MUJOCO_PY_MODULE
    if _MUJOCO_PY_MODULE:
        return _MUJOCO_PY_MODULE
    try:
        import mujoco_py

        # Override the warning function.
        from mujoco_py.builder import cymj

        cymj.set_warning_callback(_mj_warning_fn)
    except ImportError:
        print(
            "Failed to import mujoco_py. Ensure that mujoco_py (using MuJoCo "
            "v1.50) is installed.",
            file=sys.stderr,
        )
        sys.exit(1)
    _MUJOCO_PY_MODULE = mujoco_py
    return mujoco_py


def get_mujoco_py_mjlib():
    """Returns the mujoco_py mjlib module."""

    class MjlibDelegate:
        """Wrapper that forwards mjlib calls."""

        def __init__(self, lib):
            self._lib = lib

        def __getattr__(self, name: str):
            if name.startswith("mj"):
                return getattr(self._lib, "_" + name)
            raise AttributeError(name)

    return MjlibDelegate(get_mujoco_py().cymj)


def _mj_warning_fn(warn_data: bytes):
    """Warning function override for mujoco_py."""
    print(
        "WARNING: Mujoco simulation is unstable (has NaNs): {}".format(
            warn_data.decode()
        )
    )

def init_mjrender_device(config):
    """
    Decide which device to render on for mujoco.
    -1 is CPU.
    """
    if config.gpu is None:
        config.render_device = -1
    else: # use the GPU
        # if slurm job, need to get the SLURM GPUs from env var
        if "SLURM_JOB_GPUS" in os.environ or "SLURM_STEP_GPUS" in os.environ:
            if "SLURM_JOB_GPUS" in os.environ:
                gpus = os.environ["SLURM_JOB_GPUS"].split(",")
            elif "SLURM_STEP_GPUS" in os.environ:
                gpus = os.environ["SLURM_STEP_GPUS"].split(",")
            gpus = [int(i) for i in gpus]
            config.render_device = gpus[config.gpu]
        else:
            config.render_device = config.gpu
    print("MjRender Device:", config.render_device)

def get_mjrender_device(initial_render_device=None):
    """
    Decide which device to render on for mujoco.
    -1 is CPU.
    """
    if initial_render_device is None:
        return -1
    else: # use the GPU
        # if slurm job, need to get the SLURM GPUs from env var
        if "SLURM_JOB_GPUS" in os.environ or "SLURM_STEP_GPUS" in os.environ:
            if "SLURM_JOB_GPUS" in os.environ:
                gpus = os.environ["SLURM_JOB_GPUS"].split(",")
            elif "SLURM_STEP_GPUS" in os.environ:
                gpus = os.environ["SLURM_STEP_GPUS"].split(",")
            gpus = [int(i) for i in gpus]
            initial_render_device = gpus[initial_render_device]
    print("MjRender Device:", initial_render_device)
    return initial_render_device