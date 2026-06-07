-- Rung 3: a < a + 1.
theorem succ_gt_toy (a : Nat) : a < a + 1 := by
  -- crucible:region start name=proof
  omega
  -- crucible:region end
