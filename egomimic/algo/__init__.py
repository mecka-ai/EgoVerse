from egomimic.algo.act import ACT as ACT

# from egomimic.algo.pi import PI
from egomimic.algo.algo import Algo as Algo
from egomimic.algo.hpt import HPT as HPT

# OAT algos depend on the external/oat submodule, which is only checked out when
# a run is launched with init_submodules=true (tokenizer training). For every
# other model (HPT, ACT) oat is intentionally absent, so importing it eagerly
# here would break the whole algo package — and thus HPT/ACT loading — with a
# ModuleNotFoundError('oat'). Guard it: these names are only available when oat
# is present. Hydra targets the oat modules by full path
# (egomimic.algo.oat_tokenizer.OATTokenizerTrainer), so the tokenizer run does
# not rely on these re-exports.
try:
    from egomimic.algo.autoregressive import (
        AutoregressivePolicy as AutoregressivePolicy,
    )
    from egomimic.algo.autoregressive import (
        oattok_from_egomimic_lightning_ckpt as oattok_from_egomimic_lightning_ckpt,
    )
    from egomimic.algo.oat_tokenizer import OATTokenizerTrainer as OATTokenizerTrainer
except ImportError:
    pass  # oat submodule not initialized (init_submodules=false) — non-oat run

try:
    from egomimic.algo.quest_tokenizer import (
        QuestTokenizerTrainer as QuestTokenizerTrainer,
    )
except ImportError:
    pass  # quest submodule not initialized (init_submodules=quest) — non-quest run
