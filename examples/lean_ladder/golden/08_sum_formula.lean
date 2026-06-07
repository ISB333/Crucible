-- Rung 8: Gauss sum closed form for a CUSTOM recursive sum. Nonlinear
-- arithmetic with no `ring` tactic available (bare Lean core).
def sumTo : Nat → Nat
  | 0 => 0
  | n + 1 => sumTo n + (n + 1)

theorem sum_formula (n : Nat) : 2 * sumTo n = n * (n + 1) := by
  -- crucible:region start name=proof
  induction n with
  | zero => rfl
  | succ k ih =>
    simp only [sumTo, Nat.mul_add, Nat.add_mul, Nat.mul_one, Nat.one_mul] at ih ⊢
    omega
  -- crucible:region end
