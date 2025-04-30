# %%
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from caveclient import CAVEclient
from joblib import Parallel, delayed
from scipy.sparse import csr_array
from scipy.spatial import Delaunay
from tqdm_joblib import tqdm_joblib

sns.set_context("talk", font_scale=1)
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

DATASTACK = "minnie65_public"
VERSION = 1300

data_path = Path(__file__).parent.parent / "data"

query_params = dict(split_positions=True, desired_resolution=[1, 1, 1])

cell_table = pd.read_csv(data_path / "cell_info.csv.gz", index_col=0)

cell_table.query(
    "(cell_type_source == 'allen_v1_column_types_slanted_ref') and (broad_type =='excitatory')",
    inplace=True,
)

# %%

root_ids = cell_table.index

synapse_table = pd.read_csv(data_path / "column_synapses.csv.gz", index_col=0)

synapse_table = synapse_table.query(
    "pre_pt_root_id in @root_ids and post_pt_root_id in @root_ids"
).copy()


# %%


def get_skeleton(root_id, cache_only=False):
    client = CAVEclient(DATASTACK, version=VERSION)

    skeleton_path = data_path / f"skeletons/{root_id}.npz"
    if skeleton_path.exists() and not cache_only:
        skeleton = np.load(skeleton_path)
        vertices = skeleton["vertices"]
        edges = skeleton["edges"]
        level2_to_skeleton_map = skeleton["level2_to_skeleton_map"]
        return vertices, edges, level2_to_skeleton_map
    else:
        skeleton = client.skeleton.get_skeleton(root_id)

        edges = skeleton["edges"]
        vertices = skeleton["vertices"]
        level2_to_skeleton_map = skeleton["mesh_to_skel_map"]
        level2_ids = skeleton["lvl2_ids"]
        level2_to_skeleton_map = np.stack([level2_ids, level2_to_skeleton_map], axis=1)

        np.savez_compressed(
            skeleton_path,
            vertices=vertices,
            edges=edges,
            level2_to_skeleton_map=level2_to_skeleton_map,
        )
        return vertices, edges, level2_to_skeleton_map


if False:
    # turn on to pull skeletons from CAVE once
    with tqdm_joblib(total=len(cell_table.index)):
        Parallel(n_jobs=-1)(
            delayed(get_skeleton)(root_id) for root_id in cell_table.index
        )

# %%

# a bunch of mapping wrangling: synapse to level2 ID, level2 ID to skeleton ID,
# skeleton ID to vertex ID in a big collection of all skeleton vertices

all_vertices = []
all_maps = []
for root_id in cell_table.index:
    vertices, edges, level2_to_skeleton_map = get_skeleton(root_id=root_id)

    vertices = pd.DataFrame(
        vertices,
        columns=["x", "y", "z"],
        index=pd.MultiIndex.from_product(
            [[root_id], range(len(vertices))], names=["root_id", "skeleton_id"]
        ),
    )

    level2_to_skeleton_map = pd.DataFrame(
        level2_to_skeleton_map, columns=["level2_id", "skeleton_id"]
    )
    level2_to_skeleton_map["root_id"] = root_id
    all_vertices.append(vertices)
    all_maps.append(level2_to_skeleton_map)

# combine vertices from all skeletons
all_vertices = pd.concat(all_vertices)
all_vertices["vertex_id"] = np.arange(len(all_vertices))
all_maps = pd.concat(all_maps)
all_maps["vertex_id"] = all_vertices.loc[
    all_maps.set_index(["root_id", "skeleton_id"]).index
]["vertex_id"].values

# track where synapses are in the big collection of vertices
synapse_table["post_pt_vertex_id"] = (
    synapse_table["post_pt_level2_id"]
    .map(all_maps.set_index("level2_id")["vertex_id"])
    .values
)

synapse_table["pre_pt_vertex_id"] = (
    synapse_table["pre_pt_level2_id"]
    .map(all_maps.set_index("level2_id")["vertex_id"])
    .values
)

# use convention of source < target
synapse_table["source_vertex_id"] = synapse_table[
    ["pre_pt_vertex_id", "post_pt_vertex_id"]
].min(axis=1)
synapse_table["target_vertex_id"] = synapse_table[
    ["pre_pt_vertex_id", "post_pt_vertex_id"]
].max(axis=1)

# %%

currtime = time.time()

d = Delaunay(all_vertices[["x", "y", "z"]].values)
print(f"{time.time() - currtime:.3f} seconds elapsed.")

# %%

simplices = d.simplices

edges = np.concatenate(
    [
        simplices[:, [0, 1]],
        simplices[:, [1, 2]],
        simplices[:, [2, 3]],
        simplices[:, [3, 0]],
    ],
    axis=0,
)

# %%
# edges = np.unique(np.sort(edges, axis=1), axis=0)
adj = csr_array(
    (np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
    shape=(len(all_vertices), len(all_vertices)),
)
adj = adj + adj.T
edges = np.stack(np.nonzero(adj)).T
edges = pd.DataFrame(
    edges, columns=["source_vertex_id", "target_vertex_id"], index=np.arange(len(edges))
)
edges.query("source_vertex_id <= target_vertex_id", inplace=True)

# %%
vertices = all_vertices[["x", "y", "z"]].values

edge_lengths = np.linalg.norm(
    vertices[edges.loc[:, "source_vertex_id"], :]
    - vertices[edges.loc[:, "target_vertex_id"], :],
    axis=1,
)
edges["length"] = edge_lengths

sns.histplot(x=edge_lengths, log_scale=True)

# %%
edge_index = edges.set_index(["source_vertex_id", "target_vertex_id"]).index

# %%
synapse_index = synapse_table.set_index(["source_vertex_id", "target_vertex_id"]).index

# %%
is_in_delaunay = synapse_index.isin(edge_index)
print(is_in_delaunay.mean())

# %%
