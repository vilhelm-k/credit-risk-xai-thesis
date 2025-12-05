"""
Thesis-quality plotting configuration for academic figures.

This module provides matplotlib/seaborn styling optimized for:
- LaTeX documents (Computer Modern fonts)
- Journal of Finance style (clean, professional appearance)
- 12pt thesis font with footnotesize captions

Figure Width Guidelines for LaTeX:
---------------------------------
- Full-width standalone:    0.85-0.90\textwidth  (~5.5in for standard margins)
- Side-by-side subfigures:  0.48\textwidth each   (~3.1in each)
- Single column:            0.70-0.75\textwidth   (~4.5in)
- Wide figures:             1.0\textwidth         (~6.5in)

The figure sizes here are calibrated so that when imported at these widths,
font sizes appear consistent with 12pt body text and \footnotesize captions.
"""

import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# Color Palette (professional, colorblind-friendly)
# =============================================================================

# Primary colors for model comparison
COLORS = {
    'lgbm': '#2c7bb6',      # Steel blue - primary model
    'logit': '#d7191c',     # Red-orange - baseline model
    'neutral': '#404040',   # Dark gray for reference lines
    'grid': '#e0e0e0',      # Light gray for grid
    'accent': '#fdae61',    # Orange accent
}

# Sequential palette for heatmaps (white to dark blue)
CMAP_SEQUENTIAL = 'Blues'

# Diverging palette for signed values
CMAP_DIVERGING = 'RdBu_r'


# =============================================================================
# Figure Sizes (in inches, calibrated for LaTeX import)
# =============================================================================

# Standard figure sizes that produce good results at common LaTeX widths
# Heights are calculated assuming ~1.3-1.5 aspect ratio for most plots

FIGSIZE = {
    # For 0.85-0.90\textwidth (standalone figures)
    'standalone': (5.5, 3.8),
    'standalone_tall': (5.5, 5.0),
    'standalone_wide': (6.5, 3.5),

    # For 0.48\textwidth subfigures (side-by-side)
    'subfigure': (3.2, 2.6),
    'subfigure_square': (3.2, 3.0),

    # For 0.70-0.75\textwidth (medium width)
    'medium': (4.5, 3.2),
    'medium_tall': (4.5, 4.5),

    # For full-width complex figures
    'full': (6.5, 4.5),
    'full_tall': (6.5, 6.0),

    # Multi-panel layouts
    'panel_2x1': (6.5, 3.0),
    'panel_1x2': (6.5, 2.8),
    'panel_2x2': (6.5, 5.5),
}


# =============================================================================
# Font Sizes (calibrated for 12pt thesis with footnotesize captions)
# =============================================================================

# When figures are scaled in LaTeX, these sizes produce readable text:
# - At 0.48\textwidth: tick labels ~8pt, axis labels ~9pt
# - At 0.85\textwidth: tick labels ~7-8pt, axis labels ~8-9pt

FONTSIZES = {
    # For standalone/medium figures
    'title': 10,
    'axis_label': 9,
    'tick_label': 8,
    'legend': 8,
    'annotation': 8,

    # For subfigures (slightly larger to compensate for scaling)
    'subfig_title': 11,
    'subfig_axis_label': 10,
    'subfig_tick_label': 9,
    'subfig_legend': 9,
}


# =============================================================================
# Style Configuration
# =============================================================================

def set_thesis_style(use_tex: bool = True) -> None:
    """
    Configure matplotlib for thesis-quality figures.

    Parameters
    ----------
    use_tex : bool, default True
        Whether to use LaTeX for text rendering. Set to False if LaTeX
        is not installed or for faster rendering during development.

    Notes
    -----
    Call this function once at the start of your notebook/script.
    """
    # Use seaborn's clean white style as base
    sns.set_style("ticks", {
        'axes.edgecolor': '#404040',
        'axes.linewidth': 0.8,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.minor.size': 2,
        'ytick.minor.size': 2,
    })

    # Core matplotlib configuration
    config = {
        # Font configuration for LaTeX compatibility
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman', 'Times New Roman', 'DejaVu Serif'],
        'font.size': FONTSIZES['axis_label'],

        # Font sizes
        'axes.titlesize': FONTSIZES['title'],
        'axes.labelsize': FONTSIZES['axis_label'],
        'xtick.labelsize': FONTSIZES['tick_label'],
        'ytick.labelsize': FONTSIZES['tick_label'],
        'legend.fontsize': FONTSIZES['legend'],

        # Figure defaults
        'figure.figsize': FIGSIZE['standalone'],
        'figure.dpi': 150,
        'figure.facecolor': 'white',
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,

        # Line and marker defaults
        'lines.linewidth': 1.5,
        'lines.markersize': 5,

        # Grid configuration (subtle, dashed)
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        'axes.axisbelow': True,

        # Legend
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '#cccccc',
        'legend.fancybox': False,

        # Axes
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.prop_cycle': plt.cycler('color', [COLORS['lgbm'], COLORS['logit'], COLORS['accent']]),
    }

    # LaTeX rendering (optional, requires LaTeX installation)
    if use_tex:
        config.update({
            'text.usetex': True,
            'text.latex.preamble': r'\usepackage{amsmath}\usepackage{amssymb}',
        })
    else:
        config['text.usetex'] = False

    plt.rcParams.update(config)


def set_subfigure_style() -> None:
    """
    Adjust font sizes for subfigure panels (0.48 textwidth).

    Call this before creating figures that will be used as subfigures.
    """
    plt.rcParams.update({
        'axes.titlesize': FONTSIZES['subfig_title'],
        'axes.labelsize': FONTSIZES['subfig_axis_label'],
        'xtick.labelsize': FONTSIZES['subfig_tick_label'],
        'ytick.labelsize': FONTSIZES['subfig_tick_label'],
        'legend.fontsize': FONTSIZES['subfig_legend'],
        'figure.figsize': FIGSIZE['subfigure'],
    })


def reset_to_standalone_style() -> None:
    """Reset to default standalone figure styling."""
    plt.rcParams.update({
        'axes.titlesize': FONTSIZES['title'],
        'axes.labelsize': FONTSIZES['axis_label'],
        'xtick.labelsize': FONTSIZES['tick_label'],
        'ytick.labelsize': FONTSIZES['tick_label'],
        'legend.fontsize': FONTSIZES['legend'],
        'figure.figsize': FIGSIZE['standalone'],
    })


# =============================================================================
# Utility Functions
# =============================================================================

def despine(ax=None, trim: bool = False):
    """
    Remove top and right spines from axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes, optional
        Axes to despine. If None, uses current axes.
    trim : bool, default False
        If True, limit spines to data range.
    """
    sns.despine(ax=ax, trim=trim)


def format_percentage(ax, axis: str = 'y', decimals: int = 0):
    """
    Format axis tick labels as percentages.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to format.
    axis : str, {'x', 'y', 'both'}
        Which axis to format.
    decimals : int
        Number of decimal places.
    """
    from matplotlib.ticker import PercentFormatter
    formatter = PercentFormatter(xmax=1, decimals=decimals)

    if axis in ('y', 'both'):
        ax.yaxis.set_major_formatter(formatter)
    if axis in ('x', 'both'):
        ax.xaxis.set_major_formatter(formatter)


def add_identity_line(ax, **kwargs):
    """
    Add a 45-degree identity line (y=x) to the axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to add the line to.
    **kwargs
        Additional arguments passed to ax.plot().
    """
    defaults = {'color': COLORS['neutral'], 'linestyle': '--', 'linewidth': 1, 'alpha': 0.7, 'zorder': 0}
    defaults.update(kwargs)
    ax.plot([0, 1], [0, 1], **defaults)


def save_figure(fig, path, **kwargs):
    """
    Save figure with thesis-appropriate settings.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    path : str or Path
        Output path (should end in .pdf for best quality).
    **kwargs
        Additional arguments passed to fig.savefig().
    """
    defaults = {
        'format': 'pdf',
        'bbox_inches': 'tight',
        'pad_inches': 0.02,
        'dpi': 300,
    }
    defaults.update(kwargs)
    fig.savefig(path, **defaults)
