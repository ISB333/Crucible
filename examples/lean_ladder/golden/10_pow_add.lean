-- Rung 10: exponent addition law for a CUSTOM pow. Needs induction plus
-- associativity/commutativity juggling of Nat.mul without `ring`.
def pow : Nat → Nat → Nat
  | _, 0 => 1
  | a, n + 1 => a * pow a n

theorem pow_add (a m n : Nat) : pow a (m + n) = pow a m * pow a n := by
  -- crucible:region start name=proof
  induction n with
  | zero => simp [pow]
  | succ k ih =>
    have h : m + (k + 1) = (m + k) + 1 := rfl
    rw [h]
    simp [pow, ih, Nat.mul_comm, Nat.mul_assoc, Nat.mul_left_comm]
  -- crucible:region end
