import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from src.streaming import StreamSimulator


def test_stream_initial_state():
    sim = StreamSimulator(n_per_batch=100)
    assert sim.status == "idle"
    assert sim.cursor == 0
    assert sim.n_processed == 0
    assert sim.n_anomalous == 0
    assert sim.total_rows > 0


def test_stream_step_advances_cursor():
    sim = StreamSimulator(n_per_batch=100)
    sim.start()
    before = sim.cursor
    batch = sim.step()
    assert len(batch) == 100
    assert sim.cursor == before + 100
    assert sim.n_processed == 100
    assert sim.n_normal + sim.n_anomalous == 100


def test_stream_pause_and_reset():
    sim = StreamSimulator(n_per_batch=100)
    sim.start()
    sim.step()
    sim.pause()
    assert sim.status == "paused"
    # step does nothing while paused
    sim.step()
    assert sim.n_processed == 100
    sim.reset()
    assert sim.status == "idle"
    assert sim.cursor == 0
    assert sim.n_processed == 0


def test_stream_completes_small_slice():
    sim = StreamSimulator(n_per_batch=100)
    sim.df = sim.df.iloc[:250].reset_index(drop=True)
    sim.run_until_complete()
    assert sim.status == "completed"
    assert sim.n_processed == 250
    assert sim.progress_pct == 100.0