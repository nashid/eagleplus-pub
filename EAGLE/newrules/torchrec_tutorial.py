# This is tutorial code copied from torchrec tutorial and run on a single rank.
# https://github.com/pytorch/torchrec/blob/main/Torchrec_Introduction.ipynb

# import sys
# sys.path = ['', '~/anaconda3/envs/py3', '~/anaconda3/envs/py3/lib/python3.10', '~/anaconda3/envs/py3/lib/python3.10/lib-dynload', '~/anaconda3/envs/py3/lib/python3.10/site-packages']

import os
import torch
import torchrec
import torch.distributed as dist

os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"
OPTIMIZE = False

print(torch.cuda.is_available())
# Note - you will need a V100 or A100 to run tutorial as as!
# If using an older GPU (such as colab free K80), 
# you will need to compile fbgemm with the appripriate CUDA architecture
# or run with "gloo" on CPUs exi
dist.init_process_group(backend="gloo")
print("good")

ebc = torchrec.EmbeddingBagCollection(
    device="cuda",
    tables=[
        torchrec.EmbeddingBagConfig(
            name="product_table",
            embedding_dim=64,
            num_embeddings=4096,
            feature_names=["product"],
            pooling=torchrec.PoolingType.SUM,
        ),
        torchrec.EmbeddingBagConfig(
            name="user_table",
            embedding_dim=64,
            num_embeddings=4096,
            feature_names=["user"],
            pooling=torchrec.PoolingType.SUM,
        )
    ]
)

if OPTIMIZE:
    from torchrec.modules.fused_embedding_modules import fuse_embedding_optimizer
    ebc = fuse_embedding_optimizer(
        ebc,
        optimizer_type=torch.optim.SGD,
        optimizer_kwargs={"lr": 0.02},
        device=torch.device("meta"),
    )

model = torchrec.distributed.DistributedModelParallel(ebc, device=torch.device("cuda"))
print(model)
print(model.plan)

product_eb = torch.nn.EmbeddingBag(4096, 64)
result = product_eb(input=torch.tensor([101, 202, 303]), offsets=torch.tensor([0, 2, 2]))
print(result)

mb = torchrec.KeyedJaggedTensor(
    keys = ["product", "user"],
    values = torch.tensor([101, 202, 303, 404, 505, 606]).cuda(),
    lengths = torch.tensor([2, 0, 1, 1, 1, 1], dtype=torch.int64).cuda(),
)

print(mb.to(torch.device("cpu")))

pooled_embeddings = model(mb).to_dict()
print("product embeddings", pooled_embeddings["product"])
print("user embeddings", pooled_embeddings["user"])

SAVE = True
CHECKPOINT_PATH = "./model_state_dict_tutorial"
if SAVE:
    # All processes should see same parameters as they all start from same
    # random parameters and gradients are synchronized in backward passes.
    # Therefore, saving it in one process is sufficient.
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print("model saved")

dist.barrier()
# configure map_location properly
# map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
map_location = "cpu"
print(map_location)
module = torchrec.distributed.DistributedModelParallel(ebc, device=torch.device("cuda"))
CHECKPOINT_PATH = "./model_state_dict_tutorial"
module.load_state_dict(torch.load(CHECKPOINT_PATH, map_location))
print("model loaded")
pooled_embeddings = module(mb).to_dict()
print("product embeddings", pooled_embeddings["product"])
print("user embeddings", pooled_embeddings["user"])

# ebc.cuda()
result = ebc(mb).to_dict()
print(result)