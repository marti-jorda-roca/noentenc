from pathlib import Path

import numpy as np
import onnxruntime as ort


def create_session(path: Path, num_threads: int | None) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.inter_op_num_threads = 1
    if num_threads is not None:
        options.intra_op_num_threads = num_threads
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def pad(sequences: list[list[int]], pad_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Right-pad token id sequences into `(input_ids, attention_mask)`."""
    width = max(len(s) for s in sequences)
    input_ids = np.full((len(sequences), width), pad_id, dtype=np.int64)
    attention_mask = np.zeros((len(sequences), width), dtype=np.int64)
    for row, sequence in enumerate(sequences):
        input_ids[row, : len(sequence)] = sequence
        attention_mask[row, : len(sequence)] = 1
    return input_ids, attention_mask
