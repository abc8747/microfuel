import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from . import PATH_ROOT

class MPL:
    C = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
    PATH_ASSETS = PATH_ROOT / "docs" / "assets"

    @classmethod
    def _init_style(cls, dark: bool = False) -> None:
        if dark:
            plt.style.use("dark_background")
        if os.getenv("WEB"):
            mpl.use("webagg")

    @classmethod
    def default_fig(cls, *args, **kwargs) -> plt.Figure:
        cls._init_style(dark=False)
        if "figsize" not in kwargs:
            kwargs["figsize"] = (12, 7)
        fig = plt.figure(*args, **kwargs)
        fig.set_layout_engine("tight")
        return fig
