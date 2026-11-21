"""Allowlisted arithmetic; no shell, filesystem, network, or device-control tools."""
import ast,operator,math
OPS={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv,ast.FloorDiv:operator.floordiv,ast.Mod:operator.mod,ast.Pow:operator.pow,ast.USub:operator.neg,ast.UAdd:operator.pos}
def calculate(expression):
 tree=ast.parse(str(expression),mode="eval")
 def visit(n):
  if isinstance(n,ast.Expression): return visit(n.body)
  if isinstance(n,ast.Constant) and isinstance(n.value,(int,float)) and not isinstance(n.value,bool): return n.value
  if isinstance(n,ast.UnaryOp) and type(n.op) in OPS: return OPS[type(n.op)](visit(n.operand))
  if isinstance(n,ast.BinOp) and type(n.op) in OPS:
   a,b=visit(n.left),visit(n.right)
   if isinstance(n.op,ast.Pow) and abs(b)>8: raise ValueError("Exponent limit exceeded")
   result=OPS[type(n.op)](a,b)
   if not math.isfinite(float(result)) or abs(result)>1e12: raise ValueError("Result limit exceeded")
   return result
  raise ValueError("Only basic arithmetic is allowed")
 return visit(tree)
