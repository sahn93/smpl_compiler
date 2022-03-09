from typing import List, Tuple, Dict, Optional
import ssa


class ASTNode:
    def compile(self, ir: ssa.SSA):
        raise NotImplementedError


class Computation(ASTNode):
    def __init__(self, vars, funcs, stmts):
        self.vars = vars
        self.funcs = funcs
        self.stmts = stmts

    def compile(self, ir: ssa.SSA) -> None:
        main_func = ir.get_new_function_block("main")
        ir.set_current_block(main_func)
        for func in self.funcs:
            func.compile(ir)
        for stmt in self.stmts:
            stmt.compile(ir)
        ir.current_block.emit(ssa.Operation.END)


class Expression(ASTNode):
    def compile(self, ir: ssa.SSA) -> ssa.Operand:
        return None


class FunctionCall(ASTNode):

    def __init__(self, ident: str, *arg_exprs: Expression):
        self.ident = ident
        self.arg_exprs: Tuple[Expression] = arg_exprs

    def compile(self, ir: ssa.SSA):
        arg_operands = []
        for arg_expr in self.arg_exprs:
            arg_operands.append(arg_expr.compile(ir))

        ir.current_block.emit(ssa.Operation.CALL, ssa.FuncCallOp(self.ident, *arg_operands, ir))
