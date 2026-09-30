"""Data layer: reads the transformed input data shipped in data/ (never downloads).

Two loader modules read the parquet files, one for the public Treasury layers and
one for the licensed CRSP layer, and :mod:`src.data.panel` assembles them into
the four analysis panels of :mod:`src.common.schema`.
"""
