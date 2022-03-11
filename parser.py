import re
from typing import List, Tuple, Dict, Optional
from enum import Enum
from functools import reduce
import os
import ssa


class SmplLexerError(Exception):
    def __init__(self, lexer):
        line = lexer.code_lines[lexer.curr_row]
        underlined = f"{lexer.filename} {lexer.curr_row}:{lexer.curr_col}" \
                     f"   {line[:lexer.curr_col]}\033[4m{line[lexer.curr_col:]}\033[0m"
        msg = f"\nUnexpected token at:\n{underlined}"
        super().__init__(msg)


class Lexeme(Enum):
    MAIN = re.compile(r'main\s+')
    VOID = re.compile(r'void\s+')
    FUNC = re.compile(r'function\s+')
    LET = re.compile(r'let\s+')
    CALL = re.compile(r'call\s+')
    IF = re.compile(r'if\s+')
    THEN = re.compile(r'then\s+')
    ELSE = re.compile(r'else\s+')
    FI = 'fi'
    WHILE = re.compile(r'while\s+')
    DO = re.compile(r'do\s+')
    OD = 'od'
    RETURN = re.compile(r'return\s+')
    VAR = re.compile(r'var\s+')
    ARRAY = re.compile(r'array\s+')
    LPAREN = '('
    RPAREN = ')'
    LBRACE = '{'
    RBRACE = '}'
    LBRACKET = '['
    RBRACKET = ']'
    PERIOD = '.'
    COMMA = ','
    SEMICOLON = ';'
    LARROW = '<-'
    REL_OP = re.compile(r'==|!=|<=|<|>=|>')
    ASTERISK = '*'
    SLASH = '/'
    PLUS = '+'
    MINUS = '-'
    IDENT = re.compile(r"[a-zA-Z][a-zA-Z0-9]*")
    NUMBER = re.compile(r"[0-9]+")
    WHITESPACE = re.compile(r"\s+")


class Token:
    def __init__(self, lexeme, value, row, col):
        self.lexeme = lexeme
        self.value = value
        self.row = row
        self.col = col

    def __str__(self):
        return self.value


class Lexer:
    def __init__(self, filepath):
        with open(filepath, 'r') as f:
            self.code_lines = f.readlines()
        self.filename = os.path.basename(filepath)
        self.curr_row = 0
        self.curr_col = 0

    def next(self) -> Optional[Token]:
        remaining_code = self.code_lines[self.curr_row][self.curr_col:]
        if not remaining_code:
            return None
        for lexeme in Lexeme:
            pat = lexeme.value
            if isinstance(pat, re.Pattern):
                match = pat.match(remaining_code)
                if match:
                    token = Token(lexeme, match[0].strip(), self.curr_row, self.curr_col)
                    self.curr_col += len(match[0])
                    if self.curr_col == len(self.code_lines[self.curr_row]):
                        self.curr_col = 0
                        self.curr_row += 1

                    if lexeme == Lexeme.WHITESPACE:
                        return self.next()
                    else:
                        return token
            else:
                if remaining_code.startswith(pat):
                    token = Token(lexeme, pat, self.curr_row, self.curr_col)
                    self.curr_col += len(pat)
                    return token

        raise SmplLexerError(self)


class SmplParserError(Exception):
    def __init__(self, lexer: Lexer, msg: str):
        line = lexer.code_lines[lexer.curr_row]
        underlined = f"{lexer.filename} {lexer.curr_row}:{lexer.curr_col}" \
                     f"   {line[:lexer.curr_col]}\033[4m{line[lexer.curr_col:]}\033[0m"
        msg = f"\n{underlined}{msg}"
        super().__init__(msg)


class Parser:
    def __init__(self, filepath, ir: ssa.SSA):
        self.lexer = Lexer(filepath)
        self.curr = self.lexer.next()
        self.ir = ir

    def next(self):
        self.curr = self.lexer.next()

    def error(self, msg: str):
        raise SmplParserError(self.lexer, msg)

    def parse(self):
        self.computation()

    def consume_if(self, *lexemes: Lexeme) -> Token:
        if self.curr.lexeme in lexemes:
            consumed_token = self.curr
            self.next()
            return consumed_token
        else:
            self.error(f"Expected {lexemes}, got {self.curr.lexeme}")

    def computation(self) -> None:
        main_func = self.ir.get_new_function_block("main")
        self.ir.set_current_block(main_func)

        self.consume_if(Lexeme.MAIN)
        while self.curr.lexeme in [Lexeme.VAR, Lexeme.ARRAY]:
            self.var_decl()
        while self.curr.lexeme in [Lexeme.VOID, Lexeme.FUNC]:
            self.func_decl()
        self.consume_if(Lexeme.LBRACE)
        self.stat_sequence()
        self.consume_if(Lexeme.RBRACE)
        self.consume_if(Lexeme.PERIOD)

    def stat_sequence(self):
        self.statement()
        while self.curr.lexeme == Lexeme.SEMICOLON:
            self.consume_if(Lexeme.SEMICOLON)
            self.statement()
        if self.curr.lexeme == Lexeme.SEMICOLON:
            self.consume_if(Lexeme.SEMICOLON)

    def statement(self):
        if self.curr.lexeme == Lexeme.LET:
            return self.assignment()
        elif self.curr.lexeme == Lexeme.CALL:
            return self.func_call()
        elif self.curr.lexeme == Lexeme.IF:
            return self.if_statement()
        # elif self.curr.lexeme == Lexeme.WHILE:
        #     return self.while_statement()
        # elif self.curr.lexeme == Lexeme.RETURN:
        #     return self.return_statement()
        else:
            self.error(f"Expected statement, got {self.curr.lexeme}")

    def designator(self) -> Tuple[ssa.Variable, ssa.Operand]:
        ident = self.consume_if(Lexeme.IDENT)
        var = self.ir.current_block.sym_table[ident.value]
        # operand: None (uninitialized int) or InstructionOp (initialized int) or VarAddressOp (array)
        operand = var.operand
        idx_pos = 0
        while self.curr.lexeme == Lexeme.LBRACKET:
            idx_pos += 1
            self.consume_if(Lexeme.LBRACKET)
            idx_op = self.expression()
            # Calculate offset for this index
            if idx_op != ssa.ImmediateOp(0):
                offset_mul_op = self.ir.emit(
                    ssa.Operation.MUL, idx_op, reduce(lambda a, b: a*b, var.dims[idx_pos:]))
                operand = self.ir.emit(ssa.Operation.ADDA, operand, offset_mul_op)
            self.consume_if(Lexeme.RBRACKET)
        return var, operand

    def assignment(self) -> None:
        self.consume_if(Lexeme.LET)
        lhs_var, lhs = self.designator()
        self.consume_if(Lexeme.LARROW)
        rhs = self.expression()
        if lhs_var.dims is None:    # Integer, update the value of the symbol table
            lhs_var.operand = rhs
        else:   # Array, lhs is an address, store rhs in the address.
            self.ir.current_block.emit(ssa.Operation.STORE, rhs, lhs)

    def var_decl(self) -> None:
        """
        Add global variables to the main block's symbol table.
        :return: None
        """
        dims = self.type_decl()
        self.ir.current_block.decl_var(self.ident(), dims)
        while self.curr.lexeme == Lexeme.COMMA:
            self.consume_if(Lexeme.COMMA)
            self.ir.current_block.decl_var(self.ident(), dims)
        self.consume_if(Lexeme.SEMICOLON)

    def ident(self) -> str:
        return self.consume_if(Lexeme.IDENT).value

    def number(self) -> int:
        return int(self.consume_if(Lexeme.NUMBER).value)

    def type_decl(self) -> Optional[List[int]]:
        type_token = self.consume_if(Lexeme.VAR, Lexeme.ARRAY)
        dims = None
        if type_token.lexeme == Lexeme.ARRAY:
            dims = []
            self.consume_if(Lexeme.LBRACKET)
            dims.append(self.number())
            self.consume_if(Lexeme.RBRACKET)
            while self.curr.lexeme == Lexeme.LBRACKET:
                self.consume_if(Lexeme.LBRACKET)
                dims.append(self.number())
                self.consume_if(Lexeme.RBRACKET)
        return dims

    def func_decl(self) -> None:
        is_void = False
        if self.curr.lexeme == Lexeme.VOID:
            is_void = True
            self.consume_if(Lexeme.VOID)
        self.consume_if(Lexeme.FUNC)
        ident = self.ident()
        params = self.formal_param()
        self.consume_if(Lexeme.SEMICOLON)
        local_variables, stmts = self.func_body()
        self.consume_if(Lexeme.SEMICOLON)

    def func_body(self):
        # TODO
        return None, None

    def formal_param(self):
        self.consume_if(Lexeme.LPAREN)
        params = [self.ident()]
        while self.curr.lexeme == Lexeme.COMMA:
            self.consume_if(Lexeme.COMMA)
            params.append(self.ident())
        self.consume_if(Lexeme.RPAREN)
        return params

    def if_statement(self) -> None:
        self.consume_if(Lexeme.IF)
        # Create 3 Basic Blocks
        orig_block = self.ir.current_block
        fall_through_block = self.ir.get_new_basicblock(ssa.BasicBlockType.FALL_THROUGH)

        # emit incomplete branch instruction to the original block
        cond_branch_op = self.relation()  # Second operand is not added yet
        self.consume_if(Lexeme.THEN)

        # Move and compile fall-through block
        self.ir.set_current_block(fall_through_block)
        self.stat_sequence()
        join_branch_op = self.ir.emit(ssa.Operation.BRA)  # Branch operand is not added yet
        self.ir.set_current_block(orig_block)

        # Move and compile branch block
        if self.curr.lexeme == Lexeme.ELSE:
            self.consume_if(Lexeme.ELSE)

            # Create branch block
            branch_block = self.ir.get_new_basicblock(ssa.BasicBlockType.BRANCH)
            orig_block.branch_block = branch_block
            cond_branch_op.instr.operands.append(ssa.BasicBlockOp(branch_block.basic_block_id))

            self.ir.set_current_block(branch_block)
            self.stat_sequence()
            self.ir.set_current_block(orig_block)

            # Create Join block
            join_block = self.ir.get_new_basicblock(ssa.BasicBlockType.JOIN)
            fall_through_block.branch_block = join_block
            branch_block.fall_through_block = join_block
            join_block_op = ssa.BasicBlockOp(join_block.basic_block_id)
            join_branch_op.instr.operands.append(join_block_op)

            rhs_block = branch_block  # Set rhs block for phi function
        else:
            # Without then statements
            join_block = self.ir.get_new_basicblock(ssa.BasicBlockType.JOIN)
            fall_through_block.branch_block = join_block
            orig_block.branch_block = join_block
            join_block_op = ssa.BasicBlockOp(join_block.basic_block_id)
            join_branch_op.instr.operands.append(join_block_op)
            cond_branch_op.instr.operands.append(join_block_op)

            rhs_block = orig_block  # Set rhs block for phi function

        # Move to join block and add phi functions
        self.ir.set_current_block(join_block)
        # Add phi functions for modified variables
        for ident, var in join_block.sym_table.items():
            lhs_var = fall_through_block.sym_table[ident]
            rhs_var = rhs_block.sym_table[ident]
            if lhs_var != rhs_var:
                phi_op = self.ir.emit(ssa.Operation.PHI, lhs_var.operand, rhs_var.operand)
                self.ir.current_block.sym_table[ident].operand = phi_op

        self.consume_if(Lexeme.FI)

    def relation(self) -> ssa.InstructionOp:
        lhs = self.expression()
        rel_token = self.consume_if(Lexeme.REL_OP)
        rhs = self.expression()
        cmp = self.ir.emit(ssa.Operation.CMP, lhs, rhs)
        if rel_token.value == "==":
            # beq
            return self.ir.emit(ssa.Operation.BEQ, cmp)
        elif rel_token.value == "!=":
            # bne
            return self.ir.emit(ssa.Operation.BNE, cmp)
        elif rel_token.value == "<":
            # rhs is equal or less than lhs -> bge
            # Undefined yet. Revisit after compiling fall-through and add the branch
            return self.ir.emit(ssa.Operation.BGE, cmp)
        elif rel_token.value == "<=":
            # bgt
            return self.ir.emit(ssa.Operation.BGT, cmp)
        elif rel_token.value == ">":
            # blt
            return self.ir.emit(ssa.Operation.BLT, cmp)
        elif rel_token.value == ">=":
            # ble
            return self.ir.emit(ssa.Operation.BLE, cmp)

    # def while_statement(self):
    #     pass
    #
    # def return_statement(self):
    #     pass

    def expression(self) -> ssa.Operand:
        lhs = self.term()
        while self.curr.lexeme in [Lexeme.PLUS, Lexeme.MINUS]:
            token = self.consume_if(Lexeme.PLUS, Lexeme.MINUS)
            rhs = self.term()
            if token.lexeme == Lexeme.PLUS:
                lhs = self.ir.emit(ssa.Operation.ADD, lhs, rhs)
            elif token.lexeme == Lexeme.MINUS:
                lhs = self.ir.emit(ssa.Operation.SUB, lhs, rhs)
        return lhs

    def term(self) -> ssa.Operand:
        lhs = self.factor()
        while self.curr.lexeme in [Lexeme.ASTERISK, Lexeme.SLASH]:
            token = self.consume_if(Lexeme.ASTERISK, Lexeme.SLASH)
            rhs = self.factor()
            if token.lexeme == Lexeme.ASTERISK:
                lhs = self.ir.emit(ssa.Operation.MUL, lhs, rhs)
            elif token.lexeme == Lexeme.SLASH:
                lhs = self.ir.emit(ssa.Operation.DIV, lhs, rhs)
        return lhs

    def factor(self) -> ssa.Operand:
        if self.curr.lexeme == Lexeme.IDENT:
            _, operand = self.designator()
        elif self.curr.lexeme == Lexeme.NUMBER:
            operand = ssa.ImmediateOp(self.number())
        elif self.curr.lexeme == Lexeme.LPAREN:
            self.consume_if(Lexeme.LPAREN)
            operand = self.expression()
            self.consume_if(Lexeme.RPAREN)
        elif self.curr.lexeme == Lexeme.CALL:
            operand = self.func_call()
        else:
            raise SmplParserError
        return operand

    def func_call(self) -> ssa.Operand:
        self.consume_if(Lexeme.CALL)
        ident = self.consume_if(Lexeme.IDENT)
        operands = []

        if self.curr.lexeme == Lexeme.LPAREN:
            self.consume_if(Lexeme.LPAREN)
            if self.curr.lexeme != Lexeme.RPAREN:   # Parse func params
                operands.append(self.expression())
                while self.curr.lexeme == Lexeme.COMMA:
                    self.consume_if(Lexeme.COMMA)
                    operands.append(self.expression())
            self.consume_if(Lexeme.RPAREN)

        if ident.value == "InputNum":
            return self.ir.emit(ssa.Operation.READ)
        elif ident.value == "OutputNum":
            return self.ir.emit(ssa.Operation.WRITE, *operands)
        elif ident.value == "OutputNewLine":
            return self.ir.emit(ssa.Operation.WRITE_NL)

        return ssa.FuncCallOp(self.ir, ident.value, *operands)


"""
main
var a, b, c, d, e; {
    let a <- call InputNum();
    let b <- a;
    let c <- b;
    let d <- b + c;
    let e <- a + b;
    if a < 0 then let d <- d + e; let a <- d else let d <- e fi;
    call OutputNum(a)
}.
"""
