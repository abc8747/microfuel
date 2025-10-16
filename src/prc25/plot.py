from __future__ import annotations

import os
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from matplotlib.figure import Figure


class MPL:
    @classmethod
    def _init_style(cls, dark: bool = False) -> None:
        if dark:
            plt.style.use("dark_background")
        if os.getenv("WEB"):
            mpl.use("webagg")

    @classmethod
    def default_fig(cls, *args, **kwargs) -> Figure:
        cls._init_style(dark=False)
        if "figsize" not in kwargs:
            kwargs["figsize"] = (12, 7)
        fig = plt.figure(*args, **kwargs)
        fig.set_layout_engine("tight")
        return fig
