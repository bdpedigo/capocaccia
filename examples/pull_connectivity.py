# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
import seaborn as sns
from caveclient import CAVEclient
from mpl_toolkits.axes_grid1 import make_axes_locatable

data_path = Path(__file__).parent.parent / "data"

DATASTACK = "minnie65_public"  # the minnie dataset that we'll work with
VERSION = 1300  # the version/timestamp slice of the dataset
# "server" for pulling data remotely, "local" for pulling from saved files, "github" for pulling from github
DATA_SOURCE = "local"
query_params = dict(split_positions=True, desired_resolution=[1, 1, 1])

if DATA_SOURCE == "server":
    client = CAVEclient(DATASTACK, version=VERSION)

    cell_table = client.materialize.query_view("aibs_cell_info", **query_params)
    cell_table = cell_table.loc[
        cell_table["pt_root_id"].drop_duplicates(keep=False).index
    ]
    cell_table.set_index("pt_root_id", inplace=True)
    cell_table.to_csv(data_path / "cell_info.csv.gz")
elif DATA_SOURCE == "local":
    cell_table = pd.read_csv(data_path / "cell_info.csv.gz", index_col=0)
elif DATA_SOURCE == "github":
    # TODO
    cell_table = pd.read_csv("")

cell_table

# %%

column_cell_table = cell_table.query(
    "cell_type_source == 'allen_v1_column_types_slanted_ref'"
).copy()

column_root_ids = column_cell_table.index

column_cell_table


# %%

sns.set_context("talk", font_scale=1)
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

fig, ax = plt.subplots(figsize=(8, 3))

sns.scatterplot(
    data=cell_table,
    x="pt_position_x",
    y="pt_position_z",
    # hue="cell_type_source",
    color="grey",
    s=1,
    alpha=1,
    ax=ax,
)

sns.scatterplot(
    data=column_cell_table,
    x="pt_position_x",
    y="pt_position_z",
    hue="broad_type",
    s=5,
    alpha=1,
    ax=ax,
)
ax.axis("equal")
sns.move_legend(
    ax, "upper left", bbox_to_anchor=(1, 1), title="Broad Type", markerscale=5
)

# %%

fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(
    data=cell_table,
    x="pt_position_x",
    y="pt_position_y",
    # hue="broad_type",
    color="dimgrey",
    s=1,
    alpha=1,
    ax=ax,
)

sns.scatterplot(
    data=column_cell_table,
    x="pt_position_x",
    y="pt_position_y",
    hue="broad_type",
    # color="dimgrey",
    s=8,
    alpha=1,
    ax=ax,
)
ax.invert_yaxis()
ax.axis("equal")
sns.move_legend(
    ax, "upper left", bbox_to_anchor=(1, 1), title="Broad Type", markerscale=5
)


# %%


broad_type_counts = (
    cell_table.groupby("broad_type")
    .size()
    .sort_values(ascending=False)
    .rename("count")
    .to_frame()
)

fig, ax = plt.subplots(figsize=(6, 5))
sns.barplot(data=broad_type_counts, x="broad_type", y="count", ax=ax, hue="broad_type")

# %%

fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(
    data=cell_table,
    x="pt_position_x",
    y="pt_position_y",
    hue="broad_type",
    s=1,
    alpha=1,
    ax=ax,
)
ax.invert_yaxis()
ax.axis("equal")
sns.move_legend(
    ax, "upper left", bbox_to_anchor=(1, 1), title="Broad Type", markerscale=5
)

# %%

fig, axs = plt.subplots(
    1, 2, figsize=(6, 5), sharex=True, sharey=True, gridspec_kw={"wspace": 1.5}
)
ax = axs[0]
sns.scatterplot(
    data=column_cell_table.query("broad_type == 'excitatory'"),
    x="pt_position_x",
    y="pt_position_y",
    hue="cell_type",
    palette="husl",
    s=8,
    ax=ax,
)
sns.move_legend(
    ax, "upper left", bbox_to_anchor=(1, 1), title="Cell Type", markerscale=1
)

ax = axs[1]
sns.scatterplot(
    data=column_cell_table.query("broad_type == 'inhibitory'"),
    x="pt_position_x",
    y="pt_position_y",
    hue="cell_type",
    s=8,
    alpha=1,
    palette="Set1",
    ax=ax,
)
sns.move_legend(
    ax, "upper left", bbox_to_anchor=(1, 1), title="Cell Type", markerscale=1
)
ax.invert_yaxis()


# %%

if DATA_SOURCE == "server":
    column_synapse_table = client.materialize.synapse_query(
        pre_ids=column_root_ids, post_ids=column_root_ids, **query_params
    )
    column_synapse_table.set_index("id", inplace=True)
    column_synapse_table = column_synapse_table.drop(
        columns=["created", "superceded_id", "valid"]
    )
    column_synapse_table["pre_pt_level2_id"] = client.chunkedgraph.get_roots(
        column_synapse_table["pre_pt_supervoxel_id"], stop_layer=2
    )
    column_synapse_table["post_pt_level2_id"] = client.chunkedgraph.get_roots(
        column_synapse_table["post_pt_supervoxel_id"], stop_layer=2
    )
    column_synapse_table.to_csv(data_path / "column_synapses.csv.gz")
elif DATA_SOURCE == "local":
    column_synapse_table = pd.read_csv(
        data_path / "column_synapses.csv.gz", index_col=0
    )
elif DATA_SOURCE == "github":
    # TODO
    column_synapse_table = pd.read_csv("")

column_synapse_table

# %%

fig, ax = plt.subplots(
    1, 1, figsize=(4, 8), sharex=True, sharey=True, gridspec_kw={"wspace": 1.5}
)

sns.scatterplot(
    data=column_synapse_table,
    x="ctr_pt_position_x",
    y="ctr_pt_position_y",
    hue=column_synapse_table["pre_pt_root_id"].map(cell_table["cell_type"]),
    s=1,
    ax=ax,
    alpha=0.25,
)
sns.move_legend(
    ax,
    "upper left",
    bbox_to_anchor=(1, 1),
    title="Presynaptic\ncell type",
    markerscale=10,
)
ax.invert_yaxis()
ax.axis("equal")

# %%

sum_size_adjacency = (
    column_synapse_table.groupby(["pre_pt_root_id", "post_pt_root_id"])["size"]
    .sum()
    .unstack(fill_value=0)
    .reindex(index=column_root_ids, columns=column_root_ids, fill_value=0)
)

count_adjacency = (
    column_synapse_table.groupby(["pre_pt_root_id", "post_pt_root_id"])
    .size()
    .unstack(fill_value=0)
    .reindex(index=column_root_ids, columns=column_root_ids, fill_value=0)
)

column_cell_table["in_degree"] = count_adjacency.sum(axis=0)
column_cell_table["out_degree"] = count_adjacency.sum(axis=1)
column_cell_table["total_degree"] = (
    column_cell_table["in_degree"] + column_cell_table["out_degree"]
)
column_cell_table["in_size_sum"] = sum_size_adjacency.sum(axis=0)
column_cell_table["out_size_sum"] = sum_size_adjacency.sum(axis=1)
column_cell_table["total_size_sum"] = (
    column_cell_table["in_size_sum"] + column_cell_table["out_size_sum"]
)

# %%


def sparse_adjacency_plot(
    adjacency: pd.DataFrame,
    nodes: pd.DataFrame = None,
    sort_by=None,
    group_by=None,
    ascending=True,
    palette=None,
    ax=None,
    figsize=(8, 8),
    s=0.1,
    **kwargs,
):
    if nodes is not None:
        nodes = nodes.copy()

    if nodes is not None and sort_by is not None:
        nodes.sort_values(sort_by, ascending=ascending, inplace=True)
        sorted_index = nodes.index
        adjacency = adjacency.reindex(
            index=sorted_index, columns=sorted_index, fill_value=0
        )

    if nodes is not None:
        nodes["position"] = np.arange(len(nodes))

    adjacency = adjacency.values
    sources, targets = np.nonzero(adjacency)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    sns.scatterplot(
        x=targets,
        y=sources,
        color="black",
        ax=ax,
        legend=False,
        s=s,
        **kwargs,
    )

    ax.invert_yaxis()

    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, adjacency.shape[0] - 0.5)
    ax.set_ylim(adjacency.shape[1] - 0.5, -0.5)

    divider = make_axes_locatable(ax)
    cax_left = divider.append_axes("left", size="5%", pad=0.01)
    cax_left.set_xticks([])
    cax_left.set_yticks([])
    cax_left.spines[["top", "right", "left", "bottom"]].set_visible(False)
    cax_left.set_ylim(ax.get_ylim())

    cax_top = divider.append_axes("top", size="5%", pad=0.01)
    cax_top.set_xticks([])
    cax_top.set_yticks([])
    cax_top.spines[["top", "right", "left", "bottom"]].set_visible(False)
    cax_top.set_xlim(ax.get_xlim())
    cax_top.invert_yaxis()

    if group_by is not None:
        n_groups = len(group_by)
        color_width = 1 / n_groups - 0.2 / n_groups
        for i, level in enumerate(group_by):
            starts = nodes.groupby(level)["position"].min().rename("start")
            ends = nodes.groupby(level)["position"].max().rename("end")
            info = pd.concat([starts, ends], axis=1)

            for group_name, (start, end) in info.iterrows():
                # add rectangle to the left for each group
                cax_left.add_patch(
                    plt.Rectangle(
                        (i / n_groups, start),
                        color_width,
                        end - start,
                        color=palette[group_name],
                        # color="black",
                        # alpha=0.2,
                        # lw=2,
                    )
                )
                # add rectangle to the top for each group
                cax_top.add_patch(
                    plt.Rectangle(
                        (start, i / n_groups),
                        end - start,
                        color_width,
                        color=palette[group_name],
                        # color="black",
                        # alpha=0.2,
                        # lw=2,
                    )
                )
                ax.axhline(start, lw=0.5, alpha=0.5, color="black", zorder=-1)
                ax.axvline(start, lw=0.5, alpha=0.5, color="black", zorder=-1)

            ax.axhline(len(nodes), lw=0.5, alpha=0.5, color="black", clip_on=False)
            ax.axvline(len(nodes), lw=0.5, alpha=0.5, color="black", clip_on=False)

    # left_ax = ax if cax_left is None else cax_left
    cax_left.set_ylabel("Presynaptic cell")
    cax_top.set_title("Postsynaptic cell")

    return ax


palette = dict(
    zip(
        column_cell_table["cell_type"].unique(),
        sns.color_palette("husl", n_colors=column_cell_table["cell_type"].nunique()),
    )
)
palette["excitatory"] = "lightgrey"
palette["inhibitory"] = "black"

sparse_adjacency_plot(count_adjacency)

sparse_adjacency_plot(
    count_adjacency,
    nodes=column_cell_table,
    sort_by=["broad_type", "cell_type"],
    group_by=["broad_type", "cell_type"],
    ascending=True,
    palette=palette,
)

# %%
sparse_adjacency_plot(
    count_adjacency,
    nodes=column_cell_table,
    sort_by=["broad_type", "cell_type", "pt_position_y"],
    group_by=["cell_type"],
    ascending=[True, True, False],
    palette=palette,
    s=1,
)

# %%
source_ilocs, target_ilocs = np.nonzero(count_adjacency)

edges = pd.DataFrame(
    {
        "source": column_cell_table.index[source_ilocs],
        "target": column_cell_table.index[target_ilocs],
        "count": count_adjacency.values[source_ilocs, target_ilocs],
        "sum_synapse_size": sum_size_adjacency.values[source_ilocs, target_ilocs],
    }
)
edges["source_broad_type"] = edges["source"].map(column_cell_table["broad_type"])
edges["target_broad_type"] = edges["target"].map(column_cell_table["broad_type"])
edges["source_cell_type"] = edges["source"].map(column_cell_table["cell_type"])
edges["target_cell_type"] = edges["target"].map(column_cell_table["cell_type"])
edges["source_mtype"] = edges["source"].map(column_cell_table["mtype"])
edges["target_mtype"] = edges["target"].map(column_cell_table["mtype"])

# %%

categorization = "cell_type"

categories = column_cell_table.sort_values(["broad_type", "cell_type"])[
    categorization
].unique()

# %%
group_edges = (
    edges.groupby(["source_cell_type", "target_cell_type"]).size().unstack(fill_value=0)
)

group_sizes = column_cell_table.groupby("cell_type").size()
possible_group_edges = np.outer(group_sizes, group_sizes) - np.diag(group_sizes)
possible_group_edges = pd.DataFrame(
    possible_group_edges,
    index=group_sizes.index,
    columns=group_sizes.index,
)
p_connection = group_edges / possible_group_edges
p_connection = p_connection.reindex(columns=categories, index=categories, fill_value=0)

fig, ax = plt.subplots(figsize=(12, 12))
sns.heatmap(
    p_connection, cmap="Reds", ax=ax, annot=True, fmt=".2f", square=True, cbar=False
)
ax.xaxis.tick_top()
ax.xaxis.set_label_position("top")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="left")
ax.set_xlabel("Postsynaptic cell type")
ax.set_ylabel("Presynaptic cell type")

# %%
fig, ax = plt.subplots(figsize=(12, 12))

sns.heatmap(
    group_edges,
    cmap="Reds",
    annot=True,
    ax=ax,
    annot_kws={"size": 8},
    fmt=".0f",
    square=True,
    cbar=False,
)

# %%

rows = []

for source_type in categories:
    for target_type in categories:
        forward_edges = (
            edges.query(
                f"(source_{categorization} == @source_type and target_{categorization} == @target_type)"
            )
            .set_index(["source", "target"])
            .index
        )
        reverse_edges = (
            edges.query(
                f"(source_{categorization} == @target_type and target_{categorization} == @source_type)"
            )
            .set_index(["source", "target"])
            .index
        )
        possible_reverse_edges = (
            edges.query(
                f"(source_{categorization} == @source_type and target_{categorization} == @target_type)"
            )
            .set_index(["target", "source"])
            .index.rename(["source", "target"])
        )

        n_reciprocal = len(possible_reverse_edges.intersection(reverse_edges))
        n_total = len(possible_reverse_edges)
        if n_total == 0:
            reciprocity = np.nan
        else:
            reciprocity = n_reciprocal / n_total
        rows.append(
            {
                "source": source_type,
                "target": target_type,
                "reciprocity": reciprocity,
                "n_reciprocal": n_reciprocal,
                "n_total": n_total,
            }
        )

results = pd.DataFrame(rows)

results_square = results.pivot(
    index="source", columns="target", values="reciprocity"
).reindex(
    index=categories,
    columns=categories,
    fill_value=0,
)

fig, ax = plt.subplots(figsize=(15, 15))

sns.heatmap(
    results_square,
    cmap="Blues",
    cbar_kws={"label": "Reciprocity"},
    annot=True,
    fmt=".1f",
    linewidths=0.5,
    linecolor="black",
    ax=ax,
    square=True,
    cbar=False,
)
ax.xaxis.tick_top()
ax.xaxis.set_label_position("top")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

# %%
source_type = "5P-ET"
target_type = "MC"
source_root = (
    edges.query(
        f"(source_{categorization} == @source_type and target_{categorization} == @target_type)"
    )
    .groupby("source")
    .size()
    .idxmax()
    .item()
)
target_roots = column_cell_table.query("cell_type == @target_type").index

# %%
# root_id = 864691135562001633  # example basket cell
root_id = 864691135503182685
client = CAVEclient(DATASTACK, version=VERSION)
skeleton_dict = client.skeleton.get_skeleton(root_id)

# %%
pre_synapses = client.materialize.synapse_query(
    pre_ids=root_id,
    **query_params,
)
post_synapses = client.materialize.synapse_query(
    post_ids=root_id,
    **query_params,
)
# %%
pre_synapses["post_cell_type"] = (
    pre_synapses["post_pt_root_id"].map(cell_table["cell_type"]).fillna("unknown")
)


# %%

vertices = skeleton_dict["vertices"]
edges = skeleton_dict["edges"]
lines = np.column_stack((np.full((len(edges), 1), 2), edges))

skeleton_polydata = pv.PolyData(vertices, lines=lines)

plotter = pv.Plotter()

plotter.add_mesh(skeleton_polydata, color="black", line_width=1)

plotter.add_points(
    pre_synapses[
        ["ctr_pt_position_x", "ctr_pt_position_y", "ctr_pt_position_z"]
    ].values,
    color="coral",
    # scalars=pre_synapses["post_cell_type"],
    cmap="tab20",
    point_size=3,
)

plotter.add_points(
    post_synapses[
        ["ctr_pt_position_x", "ctr_pt_position_y", "ctr_pt_position_z"]
    ].values,
    color="lightblue",
    point_size=3,
)
plotter.enable_fly_to_right_click()
plotter.show()

# %%
column_synapse_table["pre_cell_type"] = (
    column_synapse_table["pre_pt_root_id"]
    .map(cell_table["cell_type"])
    .fillna("unknown")
)
column_synapse_table["post_cell_type"] = (
    column_synapse_table["post_pt_root_id"]
    .map(cell_table["cell_type"])
    .fillna("unknown")
)
column_synapse_table["pre_broad_type"] = (
    column_synapse_table["pre_pt_root_id"]
    .map(cell_table["broad_type"])
    .fillna("unknown")
)
column_synapse_table["post_broad_type"] = (
    column_synapse_table["post_pt_root_id"]
    .map(cell_table["broad_type"])
    .fillna("unknown")
)

source_type = "23P"
target_type = "23P"
sns.histplot(
    x=column_synapse_table.query(
        "pre_cell_type == @source_type and post_cell_type == @target_type"
    )["size"],
    log_scale=True,
    bins=100,
)

# %%
n_groups = len(column_synapse_table["post_cell_type"].unique())
fig, axs = plt.subplots(
    n_groups, 2, figsize=(8, 16), constrained_layout=True, sharex="col"
)
for i, (group, sub_synapses) in enumerate(
    column_synapse_table.groupby("post_cell_type")
):
    ax = axs[i, 0]
    sns.kdeplot(
        data=sub_synapses,
        x="size",
        hue="pre_broad_type",
        log_scale=True,
        # bins=50,
        ax=ax,
        # color="coral",
        legend=False,
        common_norm=False,
    )
    # ax.set_title(group)
    ax.set_xlabel("Synapse size")
    ax.set_ylabel("Count")
    ax.spines["left"].set_visible(False)
    ax.set_yticks([])
    ax.set_ylabel(group)

    sub_edges = edges.query("target_cell_type == @group")
    ax = axs[i, 1]
    sns.histplot(
        data=sub_edges,
        x="count",
        hue="source_broad_type",
        log_scale=False,
        # bins=50,
        ax=ax,
        # color="coral",
        legend=False,
        common_norm=False,
        discrete=True,
    )
    ax.set_xlim(0, 10)
    ax.set_xlabel("Count")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_ylabel("")


# %%
