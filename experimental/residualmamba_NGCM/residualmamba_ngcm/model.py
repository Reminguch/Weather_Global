"""Fresh GraphCast mesh + unchanged v24 attach_temporal + zero 203-channel head."""
import dataclasses
import haiku as hk
import jax
import numpy as np
import xarray as xr
from graphcast import graphcast as gc
from src.models.mamba.v24_Ilya.model import attach_temporal
from .features import grid_nodes,from_grid_nodes
from .artifacts import sha256


def read_model_config(path,expected_sha256):
    if not expected_sha256 or sha256(path)!=expected_sha256:
        raise ValueError('GC-small checkpoint SHA256 mismatch')
    # np.load is lazy: inspect only ModelConfig entries; never load any weights.
    with np.load(path,allow_pickle=False) as archive:
        values = {k.split(':',1)[1]:archive[k].item() for k in archive.files if k.startswith('model_config:')}
    if not values:
        raise ValueError('GC checkpoint has no ModelConfig')
    return gc.ModelConfig(**values)


class ResidualGraph(gc.GraphCast):
    def __init__(self,base,config,latitude,longitude):
        model_config = dataclasses.replace(base,mesh_size=config.mesh_size,latent_size=config.width,
            gnn_msg_steps=config.msg_steps,radius_query_fraction_edge_length=.6)
        task = gc.TaskConfig(input_variables=(),target_variables=tuple(f'residual_{i}' for i in range(203)),
                             forcing_variables=(),pressure_levels=(),input_duration='12h')
        super().__init__(model_config,task)
        attach_temporal(self,config.architecture())
        self._maybe_init(xr.Dataset(coords={'lat':np.rad2deg(latitude),'lon':np.rad2deg(longitude)}))
        self.geometry = {'grid_nodes':self._num_grid_nodes,'mesh_nodes':self._num_mesh_nodes,'edges':{}}
        for graph in (self._grid2mesh_graph_structure,self._mesh_graph_structure,self._mesh2grid_graph_structure):
            for key,edge in graph.edges.items():
                n = int(np.asarray(graph.nodes[key.node_sets[1]].n_node).sum())
                degree = np.bincount(edge.indices.receivers,minlength=n)
                if degree.shape!=(n,) or not np.all(degree>0):
                    raise ValueError(f'Disconnected GraphCast receiver: {key.name}')
                self.geometry['edges'][key.name] = {'count':len(edge.indices.receivers),'min_degree':int(degree.min())}

    def __call__(self,features):
        mesh,grid = self._run_grid2mesh_gnn(grid_nodes(features))
        mesh = self._run_mesh_gnn(mesh,is_training=True)
        output = self._run_mesh2grid_gnn(mesh,grid)
        output = hk.Linear(203,w_init=hk.initializers.Constant(0),b_init=hk.initializers.Constant(0),
                           name='temporal_residual_head')(output)
        fields = from_grid_nodes(output,len(self._grid_lon),len(self._grid_lat))
        return fields.transpose(2,0,1).reshape(7,29,len(self._grid_lon),len(self._grid_lat))


def make_model(base,config,latitude,longitude):
    def forward(features):
        return ResidualGraph(base,config,latitude,longitude)(features)
    return hk.transform_with_state(forward)


def zero_memory(state):
    import jax.numpy as jnp
    return jax.tree_util.tree_map(jnp.zeros_like,state)
