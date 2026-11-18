"""
har3d_plotstyle.py -- one place that defines publication figure style, so every
figure in the paper is visually consistent and colourblind-safe.

Imported by the figure stages. Calling apply() sets matplotlib rcParams for
vector-friendly, embedded-font, readable-at-column-width output. PALETTE is
Wong's 8-colour colourblind-safe set (Nature Methods 2011), which reads
correctly under deuteranopia, protanopia and tritanopia and in greyscale.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

# Wong 2011 colourblind-safe palette.
PALETTE = {
    'black':  '#000000', 'orange': '#E69F00', 'skyblue': '#56B4E9',
    'green':  '#009E73', 'yellow': '#F0E442', 'blue':    '#0072B2',
    'vermillion': '#D55E00', 'purple': '#CC79A7',
}
# Fixed, meaningful assignment reused across every figure so a colour means the
# same thing throughout the paper.
ROLE = {
    'observed': PALETTE['blue'], 'null': '#7F7F7F', 'excess': PALETTE['vermillion'],
    'action3d': PALETTE['blue'], 'daily': PALETTE['orange'],
    'utkinect': PALETTE['green'], 'ntu60': PALETTE['purple'],
    'gap_fill': PALETTE['vermillion'], 'bar': PALETTE['skyblue'],
}
CYCLE = [PALETTE[k] for k in ('blue', 'vermillion', 'green', 'orange',
                              'purple', 'skyblue', 'yellow', 'black')]


def apply():
    """Set journal-quality rcParams. Fonts are embedded as editable text in the
    PDF (Type-42 / TrueType) rather than outlines, which most journals require."""
    mpl.rcParams.update({
        'figure.dpi': 150,
        'savefig.dpi': 400,
        'savefig.bbox': 'tight',
        'pdf.fonttype': 42,          # embed TrueType so text stays selectable
        'ps.fonttype': 42,
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 7.5,
        'legend.frameon': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.8,
        'lines.linewidth': 1.4,
        'lines.markersize': 5,
        'axes.prop_cycle': mpl.cycler(color=CYCLE),
        'figure.constrained_layout.use': True,
    })


def save(fig, path_noext, formats=('pdf', 'png')):
    """Save one figure in every requested format. PDF is the vector master a
    journal typesets from; PNG is the preview."""
    out = []
    for ext in formats:
        p = f'{path_noext}.{ext}'
        fig.savefig(p)
        out.append(p)
    return out