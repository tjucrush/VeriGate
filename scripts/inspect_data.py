"""Inspect supplied datasets without importing the training stack."""

import hashlib
import json
from pathlib import Path


def inspect_datasets():
    try:
        import pyarrow.parquet as parquet
    except ImportError:
        parquet = None
    records = []
    for path in sorted((Path(__file__).resolve().parents[1] / "datasets").glob("*.parquet")):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        record = {"file": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()}
        if parquet is not None:
            metadata = parquet.ParquetFile(path)
            record.update(rows=metadata.metadata.num_rows, columns=metadata.schema.names)
        records.append(record)
    return records


if __name__ == "__main__":
    print(json.dumps(inspect_datasets(), indent=2))
