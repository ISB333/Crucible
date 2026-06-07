-- Rung 7: reverse is an involution — over CUSTOM app/rev, so no library
-- lemma applies. Requires discovering helper lemmas (app_nil, app_assoc,
-- rev_app) inside the proof.
def app : List Nat → List Nat → List Nat
  | [], ys => ys
  | x :: xs, ys => x :: app xs ys

def rev : List Nat → List Nat
  | [] => []
  | x :: xs => app (rev xs) [x]

theorem rev_involution (xs : List Nat) : rev (rev xs) = xs := by
  -- crucible:region start name=proof
  sorry
  -- crucible:region end
