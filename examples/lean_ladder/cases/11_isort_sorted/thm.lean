-- Rung 11: insertion sort produces a sorted list. The key invariant
-- (insert preserves sortedness) must be discovered and proved as a helper
-- with nested case analysis.
def insert' : Nat → List Nat → List Nat
  | x, [] => [x]
  | x, y :: ys => if x ≤ y then x :: y :: ys else y :: insert' x ys

def isort : List Nat → List Nat
  | [] => []
  | x :: xs => insert' x (isort xs)

def sorted : List Nat → Bool
  | x :: y :: ys => x ≤ y && sorted (y :: ys)
  | _ => true

theorem isort_sorted (xs : List Nat) : sorted (isort xs) = true := by
  -- crucible:region start name=proof
  sorry
  -- crucible:region end
