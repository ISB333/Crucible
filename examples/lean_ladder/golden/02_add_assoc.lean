-- Rung 2: associativity of Nat addition.
theorem add_assoc_toy (a b c : Nat) : (a + b) + c = a + (b + c) := by
  -- crucible:region start name=proof
  exact Nat.add_assoc a b c
  -- crucible:region end
