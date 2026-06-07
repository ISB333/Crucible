-- Rung 13: compiler correctness — a stack machine executing compiled code
-- computes the expression's value. The stated theorem is NOT directly
-- provable by induction; the prover must invent the generalization over an
-- arbitrary instruction continuation and stack.
inductive Expr where
  | const : Nat → Expr
  | plus : Expr → Expr → Expr
  | times : Expr → Expr → Expr

def eval : Expr → Nat
  | .const n => n
  | .plus a b => eval a + eval b
  | .times a b => eval a * eval b

inductive Instr where
  | push : Nat → Instr
  | add : Instr
  | mul : Instr

def exec : List Instr → List Nat → List Nat
  | [], s => s
  | .push n :: is, s => exec is (n :: s)
  | .add :: is, a :: b :: s => exec is ((b + a) :: s)
  | .mul :: is, a :: b :: s => exec is ((b * a) :: s)
  | _ :: _, _ => []  -- stack underflow

def compile : Expr → List Instr
  | .const n => [.push n]
  | .plus a b => compile a ++ compile b ++ [.add]
  | .times a b => compile a ++ compile b ++ [.mul]

theorem compile_correct (e : Expr) : exec (compile e) [] = [eval e] := by
  -- crucible:region start name=proof
  sorry
  -- crucible:region end
