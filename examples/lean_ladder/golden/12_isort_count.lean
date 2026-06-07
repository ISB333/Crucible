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
  have ins : ∀ (x : Nat) (l : List Nat),
      count y (insert' x l) = (if x = y then 1 else 0) + count y l := by
    intro x l
    induction l with
    | nil => simp [insert', count]
    | cons z zs ih =>
      by_cases hxz : x ≤ z
      · simp [insert', hxz, count]
      · simp [insert', hxz, count, ih]
        omega
  induction xs with
  | nil => rfl
  | cons x xs ih =>
    simp only [isort, count, ins, ih]
  -- crucible:region end
