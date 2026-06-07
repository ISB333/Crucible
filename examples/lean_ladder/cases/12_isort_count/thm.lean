-- Rung 12: insertion sort is a permutation, stated via element counts.
-- Needs an if-splitting helper lemma about insert' and count.
def insert' : Nat → List Nat → List Nat
  | x, [] => [x]
  | x, y :: ys => if x ≤ y then x :: y :: ys else y :: insert' x ys

def isort : List Nat → List Nat
  | [] => []
  | x :: xs => insert' x (isort xs)

def count : Nat → List Nat → Nat
  | _, [] => 0
  | y, x :: xs => (if x = y then 1 else 0) + count y xs

theorem isort_count (y : Nat) (xs : List Nat) :
    count y (isort xs) = count y xs := by
  -- crucible:region start name=proof
  sorry
  -- crucible:region end
