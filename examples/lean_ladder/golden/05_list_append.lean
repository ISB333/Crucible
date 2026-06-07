-- Rung 5: concrete list append computes.
theorem list_append_toy : [1, 2] ++ [3] = [1, 2, 3] := by
  -- crucible:region start name=proof
  rfl
  -- crucible:region end
