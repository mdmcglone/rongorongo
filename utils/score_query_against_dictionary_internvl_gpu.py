from __future__ import annotations

import mlx.core as mx

from score_query_against_dictionary_internvl import main


if __name__ == "__main__":
    mx.set_default_device(mx.gpu)
    main()
