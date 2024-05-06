from .._compatibility import is_compatible_with_meta

if is_compatible_with_meta():
    from .opcount import flop_mapping as flop_mapping
    from .profiler import profile_function, profile_method, profile_module
    from .shard_utils import (
        calculate_bwd_time as calculate_bwd_time,
    )
    from .shard_utils import (
        calculate_fwd_in,
        calculate_fwd_out,
        calculate_fwd_tmp,
    )
    from .shard_utils import (
        calculate_fwd_time as calculate_fwd_time,
    )
    from .tensor import MetaTensor as MetaTensor
else:
    from .experimental import (
        calculate_fwd_in as calculate_fwd_in,
    )
    from .experimental import (
        calculate_fwd_out as calculate_fwd_out,
    )
    from .experimental import (
        calculate_fwd_tmp as calculate_fwd_tmp,
    )
    from .experimental import (
        meta_profiler_function as meta_profiler_function,
    )
    from .experimental import (
        meta_profiler_module as meta_profiler_module,
    )
    from .experimental import (
        profile_function as profile_function,
    )
    from .experimental import (
        profile_method as profile_method,
    )
    from .experimental import (
        profile_module as profile_module,
    )

from .dataflow import GraphInfo as GraphInfo
from .memory_utils import activation_size as activation_size
from .memory_utils import is_inplace as is_inplace
from .memory_utils import parameter_size as parameter_size
