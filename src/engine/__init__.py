"""Engine: the core estimation layer of the buyback cost and benefit channels.

Three modules, one estimated coefficient each:

* :mod:`src.engine.absorption`, the absorption cost ``lambda_B``, from the
  end-of-day price change around each buyback operation;
* :mod:`src.engine.issuance`, the supply elasticity ``lambda_I``, from the
  par-yield change around each auction;
* :mod:`src.engine.auction_passthrough`, the pass-through ``rho`` from recent
  buyback intensity to auction outcomes.

Each returns a :class:`src.common.schema.Estimate` in raw units. The analysis
layer consumes these; the engine reads panels and writes nothing.
"""
