-- Rung 1: commutativity of Nat addition.
theorem add_comm_toy (a b : Nat) : a + b = b + a := by
  -- crucible:region start name=proof
  exact Nat.add_comm a b
  -- crucible:region end
