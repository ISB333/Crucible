-- Rung 9: binary-tree mirror is an involution AND preserves size.
-- Conjunction goal over a custom inductive type.
inductive Tree where
  | leaf : Tree
  | node : Tree → Nat → Tree → Tree

def mirror : Tree → Tree
  | .leaf => .leaf
  | .node l v r => .node (mirror r) v (mirror l)

def size : Tree → Nat
  | .leaf => 0
  | .node l _ r => size l + size r + 1

theorem mirror_involution (t : Tree) :
    mirror (mirror t) = t ∧ size (mirror t) = size t := by
  -- crucible:region start name=proof
  induction t with
  | leaf => exact ⟨rfl, rfl⟩
  | node l v r ihl ihr =>
    refine ⟨?_, ?_⟩
    · simp [mirror, ihl.1, ihr.1]
    · simp [mirror, size, ihl.2, ihr.2]
      omega
  -- crucible:region end
