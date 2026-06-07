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
  have ins : ∀ (x : Nat) (l : List Nat), sorted l = true → sorted (insert' x l) = true := by
    intro x l
    induction l with
    | nil => intro _; rfl
    | cons y ys ih =>
      intro h
      by_cases hxy : x ≤ y
      · simp [insert', hxy, sorted, h]
      · cases ys with
        | nil =>
          simp [insert', hxy, sorted]
          omega
        | cons z zs =>
          simp [sorted] at h
          have h2 := ih h.2
          by_cases hxz : x ≤ z
          · simp [insert', hxy, hxz, sorted, h.2]
            omega
          · simp [insert', hxy, hxz, sorted] at h2 ⊢
            exact ⟨h.1, h2⟩
  induction xs with
  | nil => rfl
  | cons x xs ih =>
    simp only [isort]
    exact ins x (isort xs) ih
  -- crucible:region end
