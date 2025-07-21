# %%
import time
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import seaborn as sns
from caveclient import CAVEclient
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.sparse import csr_array
from scipy.stats import rankdata
from tqdm_joblib import tqdm_joblib

version = 1169
client = CAVEclient("v1dd", version=version)

# %%
data_path = Path(f"/Users/ben.pedigo/capocaccia/data/v1dd/{version}")

cell_info = pd.read_feather(data_path / f"soma_cell_type_{version}.feather").set_index(
    "pt_root_id"
)
cell_info["pt_position_x"] = cell_info["pt_position"].apply(lambda x: x[0])
cell_info["pt_position_y"] = cell_info["pt_position"].apply(lambda x: x[1])
cell_info["pt_position_z"] = cell_info["pt_position"].apply(lambda x: x[2])

proofread_axon_root_ids = np.load(data_path / f"proofread_axon_list_{version}.npy")
proofread_dendrite_root_ids = np.load(
    data_path / f"proofread_dendrite_list_{version}.npy"
)
proofread_root_ids = np.intersect1d(
    proofread_axon_root_ids, proofread_dendrite_root_ids
)
proofread_root_ids = cell_info.index.intersection(proofread_root_ids)

proofread_cell_info = cell_info.loc[proofread_root_ids]

proofread_cell_info["cell_type_simple"] = (
    proofread_cell_info["cell_type"].str.split("_").str[0]
)

# %%
root_ids = proofread_cell_info.index
root_chunks = np.array_split(root_ids, 10)
synapses = []


def get_synapses(source_root_chunk, target_root_chunk):
    synapse_chunk = client.materialize.tables.synapse_target_predictions_ssa(
        pre_pt_root_id=source_root_chunk, post_pt_root_id=target_root_chunk
    ).query(split_positions=True, desired_resolution=[1, 1, 1], log_warning=False)
    synapse_chunk["pre_pt_root_id"] = synapse_chunk["pre_pt_root_id"].astype(int)
    synapse_chunk["post_pt_root_id"] = synapse_chunk["post_pt_root_id"].astype(int)
    return synapse_chunk


currtime = time.time()

with tqdm_joblib(desc="Loading synapses", total=len(root_chunks) ** 2):
    synapses = Parallel(n_jobs=-1)(
        delayed(get_synapses)(source_chunk, target_chunk)
        for source_chunk in root_chunks
        for target_chunk in root_chunks
    )
print(f"{time.time() - currtime:.3f} seconds elapsed.")

synapses = pd.concat(synapses, axis=0)


# %%


def clear_axis(axis):
    axis.set_xticks([])
    axis.set_yticks([])


def get_relative_measurement(ax, main_ax, measurement="height"):
    """
    Get the height of an axis's bounding box, as a fraction of main_ax's height.
    """
    fig = ax.figure
    fig.canvas.draw()  # ensure renderer is up-to-date
    renderer = fig.canvas.get_renderer()

    # Get height in inches for the label axis and the main axis
    label_bbox = ax.get_tightbbox(renderer=renderer)
    main_bbox = main_ax.get_tightbbox(renderer=renderer)
    if measurement == "height":
        label_height_inches = label_bbox.height
        main_height_inches = main_bbox.height
    elif measurement == "width":
        label_height_inches = label_bbox.width
        main_height_inches = main_bbox.width

    return label_height_inches / main_height_inches


class AxisGrid:
    def __init__(
        self,
        ax,
        spines=True,
    ):
        fig = ax.figure
        divider = make_axes_locatable(ax)

        self.spines = spines

        self.fig = fig
        self.ax = ax
        self.divider = divider
        self.top_axs = []
        self.left_axs = []
        self.bottom_axs = []
        self.right_axs = []
        self.side_axs = {
            "top": self.top_axs,
            "bottom": self.bottom_axs,
            "left": self.left_axs,
            "right": self.right_axs,
        }

    @property
    def all_top_axs(self):
        return [self.ax] + self.top_axs

    @property
    def all_bottom_axs(self):
        return [self.ax] + self.bottom_axs

    @property
    def all_left_axs(self):
        return [self.ax] + self.left_axs

    @property
    def all_right_axs(self):
        return [self.ax] + self.right_axs

    def append_axes(self, side, size="10%", pad="auto", **kwargs) -> plt.Axes:
        # NOTE: old way was using shared axes, but labels kept getting annoying
        # kws = {}
        # if side in ["top", "bottom"]:
        #     kws["sharex"] = self.ax
        # elif side in ["left", "right"]:
        #     kws["sharey"] = self.ax

        if pad == "auto":
            if len(self.side_axs[side]) > 0:
                last_ax = self.side_axs[side][-1]
                measurement = "height" if side in ["top", "bottom"] else "width"
                pad = get_relative_measurement(last_ax, self.ax, measurement)
            else:
                pad = 0.0

        # NOTE: this was VERY fragile, could not figure out how to do it in right in
        # float or in manual axes_size like pad = axes_size.from_any(
        #   pad, fraction_ref=axes_size.AxesX(self.ax)
        # )
        pad = f"{pad * 100}%"
        ax = self.divider.append_axes(side, size=size, pad=pad, **kwargs)

        clear_axis(ax)
        ax.tick_params(
            which="both",
            length=0,
        )

        if side in ["top", "bottom"]:
            ax.set_xlim(self.ax.get_xlim())
        elif side in ["left", "right"]:
            ax.set_ylim(self.ax.get_ylim())

        self.side_axs[side].append(ax)
        return ax

    def set_title(self, title, **kwargs):
        for ax in self.all_top_axs:
            ax.set_title("", **kwargs)
        text = self.all_top_axs[-1].set_title(title, **kwargs)
        return text

    def set_xlabel(self, xlabel, **kwargs):
        for ax in self.all_bottom_axs:
            ax.set_xlabel("", **kwargs)
        # NOTE a bit of an abuse of notation here but putting xlabel on the top
        text = self.all_top_axs[-1].set_title(xlabel, **kwargs)
        return text

    def set_ylabel(self, ylabel, **kwargs):
        for ax in self.all_left_axs:
            ax.set_ylabel("", **kwargs)
        text = self.all_left_axs[-1].set_ylabel(ylabel, **kwargs)
        return text

    def set_corner_title(self, title, **kwargs):
        """
        Set a title in the top left corner of the grid.
        """
        ax = self.top_axs[-1]
        text = ax.set_ylabel(title, ha="right", rotation=0, **kwargs)
        return text


def draw_bracket(ax, start, end, axis="x", color="black"):
    lx = np.linspace(-np.pi / 2.0 + 0.05, np.pi / 2.0 - 0.05, 500)
    tan = np.tan(lx)
    curve = np.hstack((tan[::-1], tan))
    x = np.linspace(start, end, 1000)
    if axis == "x":
        ax.plot(x, -curve, color=color)
    elif axis == "y":
        ax.plot(curve, x, color=color)


def draw_color_box(ax, start, end, axis="x", color="black", alpha=0.5, lw=0.5):
    if axis == "x":
        rect = plt.Rectangle(
            (start, 0),
            end - start,
            1,
            color=color,
        )
        ax.axvline(start, lw=0.5, alpha=0.5, color="black", zorder=2)
    elif axis == "y":
        rect = plt.Rectangle(
            (0, start),
            1,
            end - start,
            color=color,
        )
        ax.axhline(start, lw=0.5, alpha=0.5, color="black", zorder=2)
    ax.add_patch(rect)


def adjacencyplot(
    adjacency: Union[np.ndarray, csr_array],
    nodes: pd.DataFrame = None,
    groupby=None,
    sortby=None,
    group_element="color",
    group_axis_size="1%",
    node_palette=None,
    edge_palette="Greys",
    ax=None,
    figsize=(8, 8),
    size_by_weight=True,
    hue_by_weight=True,
    sizes=(1, 10),
    s=0.1,
    edge_linewidth=0.05,
    label_fontsize="small",
    **kwargs,
):
    if nodes is None:
        nodes = pd.DataFrame(index=np.arange(adjacency.shape[0]))
    nodes = nodes.reset_index().copy()
    if sortby is not None:
        if isinstance(sortby, str):
            sortby = [sortby]
    else:
        sortby = []
    if groupby is not None:
        if isinstance(groupby, str):
            groupby = [groupby]
    else:
        groupby = []
    sort_by = groupby + sortby

    nodes = nodes.sort_values(sort_by)
    nodes["position"] = np.arange(len(nodes))

    sources, targets = np.nonzero(adjacency)
    data = adjacency[sources, targets]

    # remap sources and targets to node positions
    sources = nodes.loc[sources]["position"].values
    targets = nodes.loc[targets]["position"].values

    ranked_data = rankdata(data, method="average") / len(data)

    if size_by_weight:
        size = data
    else:
        size = None

    if hue_by_weight:
        hue = ranked_data
    else:
        hue = None

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    sns.scatterplot(
        y=sources,
        x=targets,
        size=size,
        hue=hue,
        hue_norm=(0.2, 1),
        ax=ax,
        legend=False,
        s=s,
        sizes=sizes,
        palette=edge_palette,
        linewidth=edge_linewidth,
        **kwargs,
    )

    ax.axis("square")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, adjacency.shape[0] - 0.5)
    ax.set_ylim(adjacency.shape[1] - 0.5, -0.5)

    grid = AxisGrid(ax)

    # add groupby indicators starting from last to first
    for i, level in enumerate(groupby[::-1]):
        cax_left = grid.append_axes(
            "left", size=group_axis_size, pad="auto", zorder=len(sort_by) - i
        )
        cax_top = grid.append_axes(
            "top", size=group_axis_size, pad="auto", zorder=len(sort_by) - i
        )
        if group_element == "bracket":
            cax_left.spines[["top", "bottom", "left", "right"]].set_visible(False)
            cax_top.spines[["top", "bottom", "left", "right"]].set_visible(False)

        means = nodes.groupby(level)["position"].mean().rename("mean")
        starts = nodes.groupby(level)["position"].min().rename("start")
        ends = nodes.groupby(level)["position"].max().rename("end")
        info = pd.concat([starts, ends], axis=1)

        for group_name, (start, end) in info.iterrows():
            if group_element == "color":
                draw_color_box(
                    cax_left, start, end, axis="y", color=node_palette[group_name]
                )
                draw_color_box(
                    cax_top, start, end, axis="x", color=node_palette[group_name]
                )

            elif group_element == "bracket":
                draw_bracket(
                    cax_left, start, end, axis="y", color=node_palette[group_name]
                )
                draw_bracket(
                    cax_top, start, end, axis="x", color=node_palette[group_name]
                )

            ax.axhline(start, lw=0.5, alpha=0.5, color="black", zorder=-1)
            ax.axvline(start, lw=0.5, alpha=0.5, color="black", zorder=-1)

            if end == (len(nodes) - 1):
                ax.axhline(len(nodes), lw=0.5, alpha=0.5, color="black", clip_on=False)
                ax.axvline(len(nodes), lw=0.5, alpha=0.5, color="black", clip_on=False)

        cax_left.set_yticks(means.values)
        ticklabels = cax_left.set_yticklabels(means.index, rotation=0, fontsize=8)
        for label, color in zip(ticklabels, means.index.map(node_palette)):
            label.set_color(color)

        cax_top.set_xticks(means.values)
        cax_top.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        ticklabels = cax_top.set_xticklabels(
            means.index, rotation=45, fontsize=label_fontsize, ha="left"
        )
        for label, color in zip(ticklabels, means.index.map(node_palette)):
            label.set_color(color)
            label.set_in_layout(True)

    return ax, grid


def make_adjacency(synapses, cell_table, aggfunc="sum_size") -> csr_array:
    if aggfunc == "sum_size":
        edges = (
            synapses.groupby(["pre_pt_root_id", "post_pt_root_id"])["size"]
            .sum()
            .rename("weight")
            .reset_index()
        )
    elif aggfunc == "count":
        edges = (
            synapses.groupby(["pre_pt_root_id", "post_pt_root_id"])
            .count()
            .rename("weight")
            .reset_index()
        )
    edges["source_index"] = cell_table.index.get_indexer(edges["pre_pt_root_id"])
    edges["target_index"] = cell_table.index.get_indexer(edges["post_pt_root_id"])
    adjacency = csr_array(
        (edges["weight"], (edges["source_index"], edges["target_index"])),
        shape=(len(cell_table), len(cell_table)),
    )
    return adjacency


# %%
# sort nodes by their root_id label
proofread_cell_info = proofread_cell_info.sort_index()
adjacency = make_adjacency(synapses, proofread_cell_info, aggfunc="sum_size")
ax, grid = adjacencyplot(adjacency)
grid.set_ylabel("Presynaptic cell", fontsize="medium")
grid.set_xlabel("Postsynaptic cell", fontsize="medium")

# %%
# sort by depth
ax, grid = adjacencyplot(adjacency, nodes=proofread_cell_info, sortby="pt_position_y")
grid.set_ylabel("Presynaptic cell", fontsize="medium")
grid.set_xlabel("Postsynaptic cell", fontsize="medium")

# %%

node_hue = "cell_type_simple"


n_e_classes = len(
    proofread_cell_info.query("cell_type_coarse == 'E'")[node_hue].unique()
)
n_i_classes = len(
    proofread_cell_info.query("cell_type_coarse == 'I'")[node_hue].unique()
)

e_colors = sns.cubehelix_palette(
    start=0.4, rot=0.3, light=0.85, hue=1.0, dark=0.4, gamma=1.3, n_colors=n_e_classes
)

i_colors = sns.cubehelix_palette(
    start=0.3, rot=-0.4, light=0.75, dark=0.2, hue=1.0, gamma=1.3, n_colors=n_i_classes
)

cell_type_palette = dict(
    zip(
        proofread_cell_info.sort_values(["cell_type_coarse", node_hue])[
            node_hue
        ].unique(),
        e_colors + i_colors,
    )
)

cell_type_palette["E"] = np.array(list(e_colors)).mean(axis=0)
cell_type_palette["I"] = np.array(list(i_colors)).mean(axis=0)

ax, grid = adjacencyplot(
    adjacency,
    nodes=proofread_cell_info,
    groupby=["cell_type_coarse", node_hue],
    sortby=["pt_position_y"],
    node_palette=cell_type_palette,
)
grid.set_ylabel("Presynaptic cell", fontsize="large")
grid.set_xlabel("Postsynaptic cell", fontsize="large")

# %%
fig, axs = plt.subplots(1, 3, figsize=(24, 8))
connection_types = ["spine", "shaft", "soma"]
for i, connection_type in enumerate(connection_types):
    connection_type_adjacency = make_adjacency(
        synapses.query(f"tag == '{connection_type}'"),
        proofread_cell_info,
        aggfunc="sum_size",
    )
    ax = axs[i]
    ax, grid = adjacencyplot(
        connection_type_adjacency,
        nodes=proofread_cell_info,
        groupby=["cell_type_coarse", node_hue],
        sortby=["pt_position_y"],
        node_palette=cell_type_palette,
        ax=ax,
    )
    grid.set_corner_title(
        f"{connection_type.capitalize()}",
        fontsize=24,
    )
    grid.set_ylabel("Presynaptic cell", fontsize="large")
    grid.set_xlabel("Postsynaptic cell", fontsize="large")


# %%
def make_edges(synapses, cell_table, map_columns=None):
    edges = (
        synapses.groupby(["pre_pt_root_id", "post_pt_root_id"])["size"]
        .agg(["count", "sum"])
        .reset_index()
        .rename(
            columns={
                "count": "n_synapses",
                "sum": "sum_synapse_size",
                "pre_pt_root_id": "source",
                "post_pt_root_id": "target",
            }
        )
    )
    if map_columns is None:
        map_columns = []

    for map_column in map_columns:
        if map_column not in cell_table.columns:
            raise ValueError(f"Column {map_column} not found in cell_table.")
        edges["source_" + map_column] = edges["source"].map(cell_table[map_column])
        edges["target_" + map_column] = edges["target"].map(cell_table[map_column])

    return edges

edges = make_edges(
    synapses,
    proofread_cell_info,
    map_columns=["cell_type_coarse", "cell_type_simple"],
)
edges

# %%

groupby = "cell_type_simple"

categories = proofread_cell_info.sort_values(["cell_type_coarse", groupby])[
    groupby
].unique()

group_edges = (
    edges.groupby([f"source_{groupby}", f"target_{groupby}"])
    .size()
    .unstack(fill_value=0)
)

group_sizes = proofread_cell_info.groupby(groupby).size()

possible_group_edges = np.outer(group_sizes, group_sizes) - np.diag(group_sizes)
possible_group_edges = pd.DataFrame(
    possible_group_edges,
    index=group_sizes.index,
    columns=group_sizes.index,
)
p_connection = group_edges / possible_group_edges
p_connection = p_connection.reindex(columns=categories, index=categories, fill_value=0)

fig, ax = plt.subplots(figsize=(6, 6))
sns.heatmap(
    p_connection, cmap="Greens", ax=ax, annot=True, fmt=".2f", square=True, cbar=False
)
ax.xaxis.tick_top()
ax.xaxis.set_label_position("top")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="left")
ax.set_xlabel("Postsynaptic cell type", fontsize="large")
ax.set_ylabel("Presynaptic cell type", fontsize="large")

# %%
mean_edge_weight = (
    edges.groupby([f"source_{groupby}", f"target_{groupby}"])["n_synapses"]
    .mean()
    .unstack()
)
mean_edge_weight = mean_edge_weight.reindex(columns=categories, index=categories)
fig, ax = plt.subplots(figsize=(6, 6))
sns.heatmap(
    mean_edge_weight,
    cmap="Greens",
    ax=ax,
    annot=True,
    fmt=".2f",
    square=True,
    cbar=False,
)
ax.xaxis.tick_top()
ax.xaxis.set_label_position("top")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="left")
ax.set_xlabel("Postsynaptic cell type", fontsize="large")
ax.set_ylabel("Presynaptic cell type", fontsize="large")

