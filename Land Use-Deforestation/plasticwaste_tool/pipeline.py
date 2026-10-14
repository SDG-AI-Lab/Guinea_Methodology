"""
FDI pipeline orchestration -- compatibility shim.

MOVED. The implementation now lives in `plasticwaste_tool.fdi_tool.pipeline`,
alongside the other Floating Debris Index modules, so that each of the three
DPWMS components is an importable package of its own:

    fdi_tool/               component 1 -- floating debris on water
    dumpsite_tool/          component 2 -- waste accumulation sites on land
    characterisation_tool/  component 3 -- the plastic-characterisation gates

This file is kept as a re-export so that callers written against the previous
flat layout -- `plasticwaste_analysis.ipynb` among them -- continue to work
without modification. There is no second copy of the code: this module IS
`fdi_tool.pipeline`, re-bound under its old name.

New code should import `plasticwaste_tool.fdi_tool.pipeline` directly.
"""

from .fdi_tool.pipeline import *          # noqa: F401,F403
from .fdi_tool import pipeline as _impl

# `import *` skips underscore-prefixed names and anything absent from
# __all__ (which these modules do not define), so rebind the module's full
# namespace to keep the shim a faithful alias rather than an approximation.
globals().update({k: v for k, v in vars(_impl).items()
                   if not k.startswith("__")})

__all__ = [k for k in vars(_impl) if not k.startswith("_")]


if __name__ == "__main__":
    # `python -m plasticwaste_tool.pipeline` is the documented offline
    # self-test entry point (Appendix A of the methodology document), so it
    # has to keep working now that the implementation sits one level down.
    import logging

    logging.basicConfig(level=logging.WARNING)
    _impl._synthetic_test()
