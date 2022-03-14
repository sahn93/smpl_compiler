import re
from typing import List, Tuple, Dict, Optional
from enum import Enum
from functools import reduce
from copy import copy
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
    ARRAY = 'array'
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

    def error(self, msg: str = ""):
        raise SmplParserError(self.lexer, msg)

    def parse(self):
        self.computation()

    def consume_if(self, *lexemes: Lexeme) -> Token:
        if self.curr.lexeme in lexemes:
            consumed_token = self.curr
            self.next()
            return consumed_token
        else:
            self.error(f"Expected {[lexeme.name for lexeme in lexemes]}, got {self.curr.lexeme.name}")

    def computation(self) -> None:
        """
        Parse and compile a smpl code at the same time.
        :return: None
        """
        # Init main function.
        main_func = self.ir.get_new_function_block("main", [])
        self.ir.set_current_block(main_func)
        self.consume_if(Lexeme.MAIN)

        # Add variables and compile functions.
        while self.curr.lexeme in [Lexeme.VAR, Lexeme.ARRAY]:
            self.var_decl()
        while self.curr.lexeme in [Lexeme.VOID, Lexeme.FUNC]:
            self.func_decl()

        # Restore context to the main function after compiling the user declared functions.
        self.ir.set_current_block(main_func)

        # Compile function body
        self.consume_if(Lexeme.LBRACE)
        self.stat_sequence()
        self.consume_if(Lexeme.RBRACE)
        self.consume_if(Lexeme.PERIOD)
        self.ir.emit(ssa.Operation.END)

    def stat_sequence(self):
        self.statement()
        while self.curr.lexeme == Lexeme.SEMICOLON:
            self.consume_if(Lexeme.SEMICOLON)
            if self.curr.lexeme in [Lexeme.LET, Lexeme.CALL, Lexeme.IF, Lexeme.WHILE, Lexeme.RETURN]:
                stat_op = self.statement()
                if isinstance(stat_op, ssa.InstructionOp) and stat_op.instr.operation == ssa.Operation.BRA:
                    self.ir.current_block.consume_dead_code = True
        self.ir.current_block.consume_dead_code = False
        if self.curr.lexeme == Lexeme.SEMICOLON:  # Last semicolon is optional.
            self.consume_if(Lexeme.SEMICOLON)

    def statement(self) -> Optional[ssa.Operand]:
        if self.curr.lexeme == Lexeme.LET:
            return self.assignment()
        elif self.curr.lexeme == Lexeme.CALL:
            return self.func_call()  # returns an InstructionOp or a FuncCallOp
        elif self.curr.lexeme == Lexeme.IF:
            return self.if_statement()
        elif self.curr.lexeme == Lexeme.WHILE:
            return self.while_statement()
        elif self.curr.lexeme == Lexeme.RETURN:
            return self.return_statement()  # returns an InstructionOp
        else:
            self.error(f"Expected statement, got {self.curr.lexeme.name}")

    def designator(self) -> Tuple[ssa.Variable, ssa.Operand]:
        ident = self.consume_if(Lexeme.IDENT).value
        var = self.ir.current_block.sym_table[ident]
        # operand: None (uninitialized int) or InstructionOp (initialized int) or VarAddressOp (array)
        var_addr = var.operand
        idx_pos = 0
        while self.curr.lexeme == Lexeme.LBRACKET:  # is Array.
            self.consume_if(Lexeme.LBRACKET)
            idx_op = self.expression()

            # Calculate offset for this index
            if len(var.dims) > idx_pos + 1:
                stride = 4 * reduce(lambda a, b: a*b, var.dims[idx_pos+1:])
            else:
                stride = 4  # bytes
            offset_op = self.ir.emit(ssa.Operation.MUL, idx_op, ssa.ImmediateOp(stride))
            var_addr = self.ir.emit(ssa.Operation.ADDA, var_addr, offset_op)
            self.consume_if(Lexeme.RBRACKET)
            idx_pos += 1
        return var, var_addr

    def assignment(self) -> None:
        self.consume_if(Lexeme.LET)
        lhs_var, lhs = self.designator()
        self.consume_if(Lexeme.LARROW)
        rhs = self.expression()
        if lhs_var.dims is None:    # Integer, update the value of the symbol table
            lhs_var.operand = rhs
            if self.ir.current_block.num_nested_while_counter > 0:
                lhs_var.operand = ssa.VariableOp(lhs_var.name, lhs_var.operand)
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
        func_name = self.ident()
        param_names = self.formal_param()

        # Move to the new function and block, compile function body
        self.ir.get_new_function_block(func_name, param_names, is_void)
        self.consume_if(Lexeme.SEMICOLON)
        self.func_body()
        self.consume_if(Lexeme.SEMICOLON)

    def func_body(self):
        while self.curr.lexeme in [Lexeme.VAR, Lexeme.ARRAY]:
            self.var_decl()
        self.consume_if(Lexeme.LBRACE)
        self.stat_sequence()
        self.consume_if(Lexeme.RBRACE)

    def formal_param(self) -> List[str]:
        self.consume_if(Lexeme.LPAREN)
        params = []
        if self.curr.lexeme == Lexeme.IDENT:
            params.append(self.ident())
            while self.curr.lexeme == Lexeme.COMMA:
                self.consume_if(Lexeme.COMMA)
                params.append(self.ident())
        self.consume_if(Lexeme.RPAREN)
        return params

    def add_phi_functions(self,
                          join_block: ssa.BasicBlock,
                          left_parent_block: ssa.BasicBlock,
                          right_parent_block: ssa.BasicBlock,
                          is_while: bool = False) -> None:
        self.ir.set_current_block(join_block)
        num_phi_instr = 0

        for ident, var in join_block.sym_table.items():
            lhs_var = left_parent_block.get_var(ident)
            rhs_var = right_parent_block.get_var(ident)
            if lhs_var.operand != rhs_var.operand:
                num_phi_instr += 1
                phi_op = join_block.emit(ssa.Operation.PHI, lhs_var.operand, rhs_var.operand)
                join_block.set_var_op(ident, phi_op)

                # While loop -> update values modified by phi in the dominated blocks
                if is_while:
                    blocks_to_update = [join_block]
                    # Update operands modified by the phi.
                    while blocks_to_update:
                        block = blocks_to_update.pop()
                        for instr in block.instrs:
                            if instr == phi_op.instr:
                                continue
                            for i, op in enumerate(instr.operands):
                                if isinstance(op, ssa.VariableOp) and op.name == ident:
                                    instr.operands[i] = phi_op
                                    block.set_var_op(ident, phi_op)
                        blocks_to_update += reversed(block.dominates)

                    # If it is the outermost loop, commit the variable value
                    if join_block.num_nested_while_counter == 1:
                        for i, op in enumerate(phi_op.instr.operands):
                            if isinstance(op, ssa.VariableOp):
                                phi_op.instr.operands[i] = op.operand

        if is_while:
            join_block.num_nested_while_counter -= 1  # Decrease the nested while loop counter
            join_block.instrs = join_block.instrs[-num_phi_instr:] + join_block.instrs[:-num_phi_instr]

    def if_statement(self) -> None:
        self.consume_if(Lexeme.IF)
        # Create fall-through block
        orig_block = self.ir.current_block
        fall_through_root_block = self.ir.get_new_basic_block(ssa.BasicBlockType.FALL_THROUGH)

        # emit incomplete branch instruction to the original block
        cond_branch_op = self.relation()  # Second operand is not added yet
        self.consume_if(Lexeme.THEN)

        # Move and compile fall-through block
        self.ir.set_current_block(fall_through_root_block)
        self.stat_sequence()
        fall_through_last_block = self.ir.current_block  # Last block started from the fall_through_root_block
        join_branch_op = self.ir.emit(ssa.Operation.BRA)  # Branch operand is not added yet
        self.ir.set_current_block(orig_block)  # Move back to the original block

        # Move and compile branch block
        if self.curr.lexeme == Lexeme.ELSE:
            self.consume_if(Lexeme.ELSE)

            # Create branch block
            branch_root_block = self.ir.get_new_basic_block(ssa.BasicBlockType.BRANCH)
            cond_branch_op.instr.operands.append(ssa.BasicBlockOp(branch_root_block.basic_block_id))
            self.ir.set_current_block(branch_root_block)
            self.stat_sequence()
            branch_last_block = self.ir.current_block

            # Create Join block
            self.ir.set_current_block(orig_block)
            join_block = self.ir.get_new_basic_block(ssa.BasicBlockType.JOIN)
            fall_through_last_block.branch_block = join_block
            branch_last_block.fall_through_block = join_block
            join_branch_op.instr.operands.append(ssa.BasicBlockOp(join_block.basic_block_id))

            rhs_block = branch_last_block  # Set rhs block for phi function
        else:
            # Without then statements
            join_block = self.ir.get_new_basic_block(ssa.BasicBlockType.JOIN)
            fall_through_last_block.branch_block = join_block
            orig_block.branch_block = join_block
            join_block_op = ssa.BasicBlockOp(join_block.basic_block_id)
            join_branch_op.instr.operands.append(join_block_op)
            cond_branch_op.instr.operands.append(join_block_op)

            rhs_block = orig_block  # Set rhs block for phi function

        # Move to join block and add phi functions
        self.add_phi_functions(join_block, fall_through_last_block, rhs_block)

        self.consume_if(Lexeme.FI)

    def while_statement(self):
        self.consume_if(Lexeme.WHILE)
        orig_block = self.ir.current_block

        # Create a join block
        join_block = self.ir.get_new_basic_block(ssa.BasicBlockType.JOIN)
        orig_block.fall_through_block = join_block
        self.ir.set_current_block(join_block)
        join_block.num_nested_while_counter += 1

        # Add cmp to the join block
        cond_branch_op = self.relation()  # Second operand is not added yet
        self.consume_if(Lexeme.DO)

        # Create a branch block(while body) which is dominated by the join block
        branch_root_block = self.ir.get_new_basic_block(ssa.BasicBlockType.BRANCH)
        cond_branch_op.instr.operands.append(ssa.BasicBlockOp(branch_root_block.basic_block_id))
        self.ir.set_current_block(branch_root_block)
        self.stat_sequence()
        branch_last_block = self.ir.current_block
        branch_last_block.fall_through_block = join_block

        # Back to the join block and create the phi functions between the orig and join blocks
        self.add_phi_functions(join_block, orig_block, branch_last_block, is_while=True)

        # Create a fall-through block followed by the while statement
        fall_through_block = self.ir.get_new_basic_block(ssa.BasicBlockType.FALL_THROUGH)
        self.ir.set_current_block(fall_through_block)

        self.consume_if(Lexeme.OD)

    def relation(self) -> ssa.InstructionOp:
        lhs = self.expression()
        rel_token = self.consume_if(Lexeme.REL_OP)
        rhs = self.expression()
        cmp = self.ir.emit(ssa.Operation.CMP, lhs, rhs)
        if rel_token.value == "==":  # beq
            return self.ir.emit(ssa.Operation.BEQ, cmp)
        elif rel_token.value == "!=":  # bne
            return self.ir.emit(ssa.Operation.BNE, cmp)
        elif rel_token.value == "<":  # rhs is equal or less than lhs -> bge
            return self.ir.emit(ssa.Operation.BGE, cmp)
        elif rel_token.value == "<=":  # bgt
            return self.ir.emit(ssa.Operation.BGT, cmp)
        elif rel_token.value == ">":  # blt
            return self.ir.emit(ssa.Operation.BLT, cmp)
        elif rel_token.value == ">=":  # ble
            return self.ir.emit(ssa.Operation.BLE, cmp)

    def return_statement(self) -> ssa.Operand:
        self.consume_if(Lexeme.RETURN)
        if self.curr.lexeme in [Lexeme.IDENT, Lexeme.NUMBER, Lexeme.LPAREN, Lexeme.CALL]:
            self.ir.current_block.decl_var("@R31")
            self.ir.current_block.sym_table["@R31"].operand = self.expression()
            return self.ir.emit(ssa.Operation.BRA, ssa.VarAddressOp("CallLoc"))
        else:
            self.error()

    def expression(self) -> ssa.Operand:
        lhs = self.term()
        while self.curr.lexeme in [Lexeme.PLUS, Lexeme.MINUS]:
            token = self.consume_if(Lexeme.PLUS, Lexeme.MINUS)
            rhs = self.term()
            if token.lexeme == Lexeme.PLUS:
                # Immediate Operation Optimization
                if isinstance(lhs, ssa.ImmediateOp) and isinstance(rhs, ssa.ImmediateOp):
                    lhs = ssa.ImmediateOp(lhs.value + rhs.value)
                else:
                    lhs = self.ir.emit(ssa.Operation.ADD, lhs, rhs)
            elif token.lexeme == Lexeme.MINUS:
                if isinstance(lhs, ssa.ImmediateOp) and isinstance(rhs, ssa.ImmediateOp):
                    lhs = ssa.ImmediateOp(lhs.value - rhs.value)
                else:
                    lhs = self.ir.emit(ssa.Operation.SUB, lhs, rhs)
        return lhs

    def term(self) -> ssa.Operand:
        lhs = self.factor()
        while self.curr.lexeme in [Lexeme.ASTERISK, Lexeme.SLASH]:
            token = self.consume_if(Lexeme.ASTERISK, Lexeme.SLASH)
            rhs = self.factor()
            # Immediate Operation Optimization
            if token.lexeme == Lexeme.ASTERISK:
                if isinstance(lhs, ssa.ImmediateOp) and isinstance(rhs, ssa.ImmediateOp):
                    lhs = ssa.ImmediateOp(lhs.value * rhs.value)
                else:
                    lhs = self.ir.emit(ssa.Operation.MUL, lhs, rhs)
            elif token.lexeme == Lexeme.SLASH:
                if isinstance(lhs, ssa.ImmediateOp) and isinstance(rhs, ssa.ImmediateOp):
                    lhs = ssa.ImmediateOp(int(lhs.value / rhs.value))
                else:
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
        """
        Call a function with arguments
        :return: FuncCallOp if return int, InstructionOp if void.
        """
        self.consume_if(Lexeme.CALL)
        ident = self.consume_if(Lexeme.IDENT).value
        args = []

        if self.curr.lexeme == Lexeme.LPAREN:
            self.consume_if(Lexeme.LPAREN)
            if self.curr.lexeme != Lexeme.RPAREN:   # Parse func params
                args.append(self.expression())
                while self.curr.lexeme == Lexeme.COMMA:
                    self.consume_if(Lexeme.COMMA)
                    args.append(self.expression())
            self.consume_if(Lexeme.RPAREN)

        # Emit a corresponding operation for predefined functions
        if ident == "InputNum":
            return self.ir.emit(ssa.Operation.READ)
        elif ident == "OutputNum":
            return self.ir.emit(ssa.Operation.WRITE, *args)
        elif ident == "OutputNewLine":
            return self.ir.emit(ssa.Operation.WRITE_NL)

        func_call_op = ssa.FuncCallOp(self.ir, ident, *args)

        if self.ir.func_roots[ident].func.is_void:
            # If the function returns nothing, emit a call instruction.
            return self.ir.emit(ssa.Operation.CALL, func_call_op)
        else:
            # If the function has a return value, return as an operand
            return func_call_op
