"""FROZEN passthrough strategy. Wave 2 opens an editable region here to implement
the batching multiplexer, speculative-decoding orchestration, and prefix cache.
For Wave 0/1 the harness ignores this module; it exists so the immutable structure
matches the spec and Wave 2 can open it without moving files.
"""