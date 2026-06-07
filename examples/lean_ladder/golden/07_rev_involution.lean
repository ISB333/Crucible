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
  have app_nil : ∀ (l : List Nat), app l [] = l := by
    intro l
    induction l with
    | nil => rfl
    | cons x xs ih => simp [app, ih]
  have app_assoc : ∀ (a b c : List Nat), app (app a b) c = app a (app b c) := by
    intro a b c
    induction a with
    | nil => rfl
    | cons x xs ih => simp [app, ih]
  have rev_app : ∀ (a b : List Nat), rev (app a b) = app (rev b) (rev a) := by
    intro a b
    induction a with
    | nil => simp [app, rev, app_nil]
    | cons x xs ih => simp [app, rev, ih, app_assoc]
  induction xs with
  | nil => rfl
  | cons x xs ih => simp [rev, rev_app, app, ih]
  -- crucible:region end
