# %%
from pathlib import Path

import pandas as pd
from caveclient import CAVEclient

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
client = CAVEclient(DATASTACK, version=117)

column_synapse_table_old = client.materialize.query_table(
    "synapses_pni_2", filter_in_dict=dict(id=column_synapse_table.index)
)
column_synapse_table_old.set_index("id", inplace=True)
# %%

column_synapse_table["old_pre_pt_root_id"] = column_synapse_table_old["pre_pt_root_id"]

# %%

from sklearn.neighbors import NearestNeighbors

for old_root_id, old_root_data in column_synapse_table.groupby("old_pre_pt_root_id"):
    if len(old_root_data) < 5:
        continue
    points = old_root_data[
        ["ctr_pt_position_x", "ctr_pt_position_y", "ctr_pt_position_z"]
    ].values
    neighbors = NearestNeighbors(n_neighbors=min(len(points), 5)).fit(points)
    distances, indices = neighbors.kneighbors(points)
    break

# %%
