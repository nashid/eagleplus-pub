# This file was copied from torchrec sharding and modified
# https://github.com/pytorch/torchrec/blob/main/examples/sharding/sharding.ipynb

from copyreg import pickle
import os
import torch
import torchrec
import pickle
from torchsnapshot import Snapshot

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29510"

from torchrec.distributed.planner.types import ParameterConstraints
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.types import ShardingType
from typing import Dict

# define model using embeddingbag
large_table_cnt = 2
small_table_cnt = 2
large_tables=[
    torchrec.EmbeddingBagConfig(
        name="large_table_" + str(i),
        embedding_dim=8,
        num_embeddings=4096,
        feature_names=["large_table_feature_" + str(i)],
        pooling=torchrec.PoolingType.SUM,
    ) for i in range(large_table_cnt)
]
small_tables=[
    torchrec.EmbeddingBagConfig(
        name="small_table_" + str(i),
        embedding_dim=8,
        num_embeddings=1024,
        feature_names=["small_table_feature_" + str(i)],
        pooling=torchrec.PoolingType.SUM,
    ) for i in range(small_table_cnt)
]

# define model as embeddingbagcollection of 4 tables/embeddingbags
ebc = torchrec.EmbeddingBagCollection(
    device="meta",
    tables=large_tables + small_tables
)

EBCmodel = ebc
print(ebc)

from torchrec.models.dlrm import DLRM, DenseArch
ebc = DLRM(
   embedding_bag_collection=ebc,
   dense_in_features=1024,
   dense_arch_layer_sizes=[8],
   over_arch_layer_sizes=[5, 1],
)
B = 20
D = 3
ebc = DenseArch(10, layer_sizes=[15, D])

# import torchvision.models as models
# ebc = models.resnet18(pretrained=True)

# from torchrec.models.deepfm import SimpleDeepFMNN
# ebc = SimpleDeepFMNN(
#     num_dense_features=1024, embedding_bag_collection=ebc, hidden_layer_size=20, deep_fm_dimension=5
# )

# generate cosntraints for model sharding
def gen_constraints(sharding_type: ShardingType = ShardingType.TABLE_WISE) -> Dict[str, ParameterConstraints]:
    large_table_constraints = {
        "large_table_" + str(i): ParameterConstraints(
            sharding_types=[sharding_type.value],
        ) for i in range(large_table_cnt)
    }
    small_table_constraints = {
        "small_table_" + str(i): ParameterConstraints(
         sharding_types=[sharding_type.value],
        ) for i in range(small_table_cnt)
    }
    constraints = {**large_table_constraints, **small_table_constraints}
    return constraints

# execute model on a single rank
def single_rank_execution(
    rank: int,
    world_size: int,
    constraints: Dict[str, ParameterConstraints],
    module: torch.nn.Module,
    backend: str,
    distributed = True,
    save_model = False,
    model_saving_path = None,
    load_model = False,
    model_loading_path = None,
    result_path = None
) -> None:
    import os
    import torch
    import torch.distributed as dist
    from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
    from torchrec.distributed.model_parallel import DistributedModelParallel
    from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
    from torchrec.distributed.types import ModuleSharder, ShardingEnv, ShardingPlan
    from typing import cast

    # initialize process setting
    def init_distributed_single_host(
        rank: int,
        world_size: int,
        backend: str,
        # pyre-fixme[11]: Annotation `ProcessGroup` is not defined as a type.
    ) -> dist.ProcessGroup:
        os.environ["RANK"] = f"{rank}"
        os.environ["WORLD_SIZE"] = f"{world_size}"
        dist.init_process_group(rank=rank, world_size=world_size, backend=backend)
        return dist.group.WORLD

    if backend == "nccl":
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        device_str = "cuda"
    else:
        device = torch.device("cpu")
        device_str = "cpu"
    topology = Topology(world_size=world_size, compute_device=device_str)
    pg = init_distributed_single_host(rank, world_size, backend)
    planner = EmbeddingShardingPlanner(
        topology=topology,
        constraints=constraints,
    )
    sharders = [cast(ModuleSharder[torch.nn.Module], EmbeddingBagCollectionSharder())]
    plan: ShardingPlan = planner.collective_plan(module, sharders, pg)
    
    # wrap model with DistributedModelParallel, which will shard the model and place on different devices
    sharded_model = DistributedModelParallel(
        module,
        env=ShardingEnv.from_process_group(pg),
        plan=plan,
        sharders=sharders,
        device=device,
    )
    print(f"rank:{rank},sharding plan: {plan}")

    app_state = {"model": sharded_model}

    if save_model:
        assert model_saving_path is not None, "model saving path is None."
        snapshot = Snapshot.take(path=model_saving_path, app_state=app_state)
    
    if load_model:
        assert model_loading_path is not None, "model loading path is None."
        dist.barrier()
        snapshot = Snapshot(path=model_loading_path)
        snapshot.restore(app_state)
    
    if not distributed:
        sharded_model = sharded_model.module

    sharded_model.eval()

    # inference with random input
    mb = torchrec.KeyedJaggedTensor(
        keys = ["large_table_feature_0", "small_table_feature_0", "large_table_feature_1", "small_table_feature_1"],
        values = torch.tensor([101, 202, 303, 404, 505, 606, 101, 202, 303, 404, 505, 606]),
        lengths = torch.tensor([2, 0, 1, 1, 1, 1, 2, 0, 1, 1, 1, 1], dtype=torch.int64),
    )
    mb = torch.range(0, B*10-1, dtype=torch.float32).view(B, 10)/(B*10-1)
    # mb = torch.rand(1, 3, 224, 224).to(device)
    # feature = torch.rand((3, 1024))
    print("mb: ", mb)
    result = sharded_model(mb)
    # result = sharded_model(mb).to_dict()
    # result = sharded_model(feature, mb)
    print(result)
    
    with open(result_path + "_rank_{}.p".format(rank), "wb") as f:
        pickle.dump(result, f)

    return sharded_model

import multiprocess

# execute model on multiple ranks
def spmd_sharing_simulation(
    sharding_type: ShardingType = ShardingType.TABLE_WISE,
    world_size = 2,
    distributed = True,
    save_model = False,
    model_saving_path = None,
    load_model = False,
    model_loading_path = None,
    result_path = None
):
    ctx = multiprocess.get_context("spawn")
    processes = []
    for rank in range(world_size):
        p = ctx.Process(
            target=single_rank_execution,
            args=(
                rank,
                world_size,
                gen_constraints(sharding_type),
                ebc,
                # "nccl",
                "gloo",

                distributed,
                save_model,
                model_saving_path,
                load_model,
                model_loading_path,
                result_path
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        assert 0 == p.exitcode

if __name__ == "__main__":
    spmd_sharing_simulation(ShardingType.TABLE_WISE, 2, True, True, "modelsnapshot", False, None, result_path="result_tablewise_ws_2")
    spmd_sharing_simulation(ShardingType.TABLE_WISE, 1, True, False, None, True, "modelsnapshot", result_path="result_tablewise_ws_1")
    spmd_sharing_simulation(ShardingType.TABLE_WISE, 1, False, False, None, True, "modelsnapshot", result_path="result_tablewise_nondistributed")

    # result_path = "result_tablewise_ws_2"
    # rank = 1
    # with open(result_path + "_rank_{}.p".format(rank), "rb") as f:
    #     result_2 = pickle.load(f)
    #     print(result_2)
    
    # result_path = "result_tablewise_ws_1"
    # rank = 0
    # with open(result_path + "_rank_{}.p".format(rank), "rb") as f:
    #     result_3 = pickle.load(f)
    #     print(result_3)

    #     result_path = "result_tablewise_nondistributed"
    # rank = 0
    # with open(result_path + "_rank_{}.p".format(rank), "rb") as f:
    #     result_4 = pickle.load(f)
    #     print(result_4)
    
    # print(result_2 == result_3)
