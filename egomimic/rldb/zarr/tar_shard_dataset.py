# Re-export from zarr_dataset_multi for backward compatibility.
from egomimic.rldb.zarr.zarr_dataset_multi import (  # noqa: F401
    TarShardMultiDataset,
    TarShardMultiDataset as TarShardIterableDataset,
)
