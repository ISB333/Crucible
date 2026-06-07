-- Rung 14: the Ackermann function dominates its second argument. Classic
-- nested double induction: outer on m generalizing n, inner on n, both
-- inductive hypotheses needed simultaneously.
def ack : Nat → Nat → Nat
  | 0, n => n + 1
  | m + 1, 0 => ack m 1
  | m + 1, n + 1 => ack m (ack (m + 1) n)

theorem ack_gt (m n : Nat) : n < ack m n := by
  -- crucible:region start name=proof
  induction m generalizing n with
  | zero => simp [ack]
  | succ k ih =>
    induction n with
    | zero =>
      have h := ih 1
      simp only [ack]
      omega
    | succ n' ihn =>
      have h1 := ih (ack (k + 1) n')
      simp only [ack] at *
      omega
  -- crucible:region end
