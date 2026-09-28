"""The mouse visual areas the ventral/dorsal analyses run over, and how each one is named,
colored and placed on the anatomical hierarchy.

These are shared between the compute side (RSM/ventral_dorsal_contrast.py, which needs the area
order and the two reference areas to know what to fetch and what to compare against) and the
plotting side (plotting/plot_ventral_dorsal*.py, which needs the colors and hierarchy scores to
draw them).
"""

AREA_ORDER = ["VISp", "VISl", "VISpm", "VISal", "VISam"]

# Wang et al. 2011 and Wang et al. 2012: the most ventral and most dorsal areas
# of mouse visual cortex, used as reference points for every other area's contrast.
VENTRAL_REF_AREA = "VISl"
DORSAL_REF_AREA = "VISam"

AREA_COLORS = {
    "VISp": "#1f77b4",
    "VISl": "#ff7f0e",
    "VISal": "#2ca02c",
    "VISpm": "#d62728",
    "VISam": "#9467bd",
}

# Harris et al. 2019, "Hierarchical organization of cortical and thalamic connectivity" (Nature
# 575:195-202), Supplementary Table 9: hierarchy score per area, combining cortico-cortical,
# thalamocortical, and corticothalamic connectivity evidence (the "Cre_conf CC+TC+CT" column).
ANATOMICAL_HIERARCHY_SCORE = {
    "VISp": -0.420983039,
    "VISl": -0.085552019,
    "VISal": 0.022612886,
    "VISpm": 0.112409304,
    "VISam": 0.274742343,
}
