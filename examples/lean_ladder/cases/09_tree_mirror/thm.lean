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
  sorry
  -- crucible:region end
