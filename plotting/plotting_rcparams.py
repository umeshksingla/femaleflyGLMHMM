import matplotlib

rcParams = matplotlib.rcParams
# -- Axes --
rcParams['axes.spines.bottom'] = True
rcParams['axes.spines.left'] = True
rcParams['axes.spines.right'] = False
rcParams['axes.spines.top'] = False
rcParams['axes.grid'] = False
rcParams['axes.grid.axis'] = 'y'
rcParams['grid.color'] = 'black'
rcParams['grid.linewidth'] = 0.5
rcParams['axes.axisbelow'] = True
rcParams['axes.linewidth'] = 2
rcParams['axes.ymargin'] = 0
rcParams["axes.labelsize"] = 12
rcParams["xtick.labelsize"] = 12
rcParams["ytick.labelsize"] = 12
rcParams["legend.fontsize"] = 12

# -- Ticks and tick labels --
rcParams['axes.edgecolor'] = 'black'
rcParams['xtick.bottom'] = True
rcParams['ytick.left'] = True
rcParams['xtick.color'] = 'black'
rcParams['ytick.color'] = 'black'
rcParams['xtick.major.width'] = 1
rcParams['ytick.major.width'] = 1
rcParams['xtick.major.size'] = 4
rcParams['ytick.major.size'] = 4

# -- Fonts --
rcParams['font.size'] = 16  # Panel label
# rcParams['font.family'] = 'Arial'
# rcParams['font.sans-serif'] = 'Arial'
rcParams['text.color'] = 'black'
rcParams['axes.labelcolor'] = 'black'

# -- Figure size --
# rcParams['figure.figsize'] = (6, 4)
# rcParams['figure.dpi'] = 300

# -- Saving Options --
# rcParams['savefig.bbox'] = 'tight'
# rcParams['pdf.fonttype'] = 42
# rcParams['ps.fonttype'] = 42
# rcParams['savefig.transparent'] = True

# -- Plot Styles --
rcParams['lines.linewidth'] = 1.5
