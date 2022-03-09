import re
from enum import Enum
import ssa


class SmplParserError(Exception):
    pass


class SmplLexerError(Exception):
    def __init__(self, code_lines, curr_row, curr_col):
        line = code_lines[curr_row]
        underlined = f"{line[:curr_col]}\033[4m{line[curr_col:]}\033[0m"
        msg = f"Unexpected token at line {curr_row}: {underlined}"
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
    EQ = '=='
    INEQ = '!='
    LT = '<'
    LE = '<='
    GT = '>'
    GE = '>='
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
        self.curr_row = 0
        self.curr_col = 0

    def next(self):
        remaining_code = self.code_lines[self.curr_row][self.curr_col:]
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

        raise SmplLexerError(self.code_lines, self.curr_row, self.curr_col)


class Parser:
    def __init__(self, filepath, ir: ssa.SSA):
        self.lexer = Lexer(filepath)
        self.curr = self.lexer.next()
        self.inst_index = 1
        self.ir = ir

    def next(self):
        self.curr = self.lexer.next()

    @staticmethod
    def error(msg: str):
        raise SmplParserError(msg)

    def parse(self):
        self.computation()

    # def consume_if(self, *symbols: Lexeme):
    #     if self.curr in symbols:
    #         consumed_token = self.curr
    #         self.next()
    #         return consumed_token
    #     else:
    #         self.error(f"Expected {symbols}, got {self.curr.lexeme}")
    #
    # def computation(self) -> None:
    #     self.consume_if(Lexeme.MAIN)
    #     while self.curr.lexeme in [Lexeme.VAR, Lexeme.ARRAY]:
    #         self.var_decl()
    #
    #     while self.curr.lexeme in [Lexeme.VOID, Lexeme.FUNC]:
    #         self.func_decl()
    #
    #     self.consume_if(Lexeme.LBRACE)
    #     self.stat_sequence()
    #     self.consume_if(Lexeme.RBRACE)
    #     self.consume_if(Lexeme.PERIOD)
    #
    # def stat_sequence(self):
    #     self.statement()
    #     while self.curr.lexeme == Lexeme.SEMICOLON:
    #         self.consume_if(Lexeme.SEMICOLON)
    #         self.statement()
    #     self.consume_if(Lexeme.SEMICOLON)
    #
    # def statement(self):
    #     if self.curr.lexeme == Lexeme.LET:
    #         return self.assignment()
    #     elif self.curr.lexeme == Lexeme.CALL:
    #         return self.func_call()
    #     elif self.curr.lexeme == Lexeme.IF:
    #         return self.if_statement()
    #     elif self.curr.lexeme == Lexeme.WHILE:
    #         return self.while_statement()
    #     elif self.curr.lexeme == Lexeme.RETURN:
    #         return self.return_statement()
    #     else:
    #         self.error(f"Expected statement, got {self.curr.lexeme}")
    #
    # def assignment(self):
    #     self.consume_if(Lexeme.LET)
    #     ident, idx = self.designator()
    #     self.consume_if(Lexeme.LARROW)
    #     rhs = self.expression()
    #     if len(idx) == 0:
    #         self.curr_basicblock.sym_table[ident].value = rhs
    #     else:
    #         self.curr_basicblock.sym_table[ident].value[idx] = rhs
    #
    # def var_decl(self):
    #     dim = self.type_decl()
    #     self.curr_basicblock.sym_table[self.ident()] = Variable(dim)
    #     while self.curr.lexeme == Lexeme.COMMA:
    #         self.consume_if(Lexeme.COMMA)
    #         self.curr_basicblock.sym_table[self.ident()] = Variable(dim)
    #     self.consume_if(Lexeme.SEMICOLON)
    #
    # def ident(self):
    #     return self.consume_if(Lexeme.IDENT).value
    #
    # def number(self):
    #     return int(self.consume_if(Lexeme.NUMBER).value)
    #
    # def type_decl(self):
    #     type_token = self.consume_if(Lexeme.VAR, Lexeme.ARRAY)
    #     dims = None
    #     if type_token.lexeme == Lexeme.ARRAY:
    #         dims = []
    #         self.consume_if(Lexeme.LBRACKET)
    #         dims.append(self.number())
    #         self.consume_if(Lexeme.RBRACKET)
    #         while self.curr.lexeme == Lexeme.LBRACKET:
    #             self.consume_if(Lexeme.LBRACKET)
    #             dims.append(self.number())
    #             self.consume_if(Lexeme.RBRACKET)
    #     return dims
    #
    # def func_decl(self):
    #     is_void = False
    #     if self.curr.lexeme == Lexeme.VOID:
    #         is_void = True
    #         self.consume_if(Lexeme.VOID)
    #     self.consume_if(Lexeme.FUNC)
    #     ident = self.ident()
    #     params = self.formal_param()
    #     self.consume_if(Lexeme.SEMICOLON)
    #     local_variables, stmts = self.func_body()
    #     self.consume_if(Lexeme.SEMICOLON)
    #     # TODO: Declare a function node and return it
    #     return ident, None
    #
    # def func_body(self):
    #     # TODO
    #     return None, None
    #
    # def formal_param(self):
    #     self.consume_if(Lexeme.LPAREN)
    #     params = [self.ident()]
    #     while self.curr.lexeme == Lexeme.COMMA:
    #         self.consume_if(Lexeme.COMMA)
    #         params.append(self.ident())
    #     self.consume_if(Lexeme.RPAREN)
    #     return params
    #
    # def if_statement(self):
    #     pass
    #
    # def while_statement(self):
    #     pass
    #
    # def return_statement(self):
    #     pass
    #
    # def designator(self):
    #     ident = self.consume_if(Lexeme.IDENT)
    #     idxs = []
    #     while self.curr.lexeme == Lexeme.LBRACKET:
    #         self.consume_if(Lexeme.LBRACKET)
    #         idxs.append(self.expression())
    #         self.consume_if(Lexeme.RBRACKET)
    #     return ident.value, tuple(idxs)
    #
    # def expression(self):
    #     lhs = self.term()
    #     while self.curr.lexeme in [Lexeme.PLUS, Lexeme.MINUS]:
    #         token = self.consume_if(Lexeme.PLUS, Lexeme.MINUS)
    #         rhs = self.term()
    #         self.compute(token.lexeme, lhs, rhs)
    #     return lhs
    #
    # def term(self):
    #     lhs = self.factor()
    #     while self.curr.lexeme in [Lexeme.ASTERISK, Lexeme.SLASH]:
    #         token = self.consume_if(Lexeme.ASTERISK, Lexeme.SLASH)
    #         rhs = self.term()
    #         self.compute(token.lexeme, lhs, rhs)
    #     return lhs
    #
    # def factor(self):
    #     if self.curr.lexeme == Lexeme.IDENT:
    #         operand = self.curr_basicblock.sym_table[self.curr.value]
    #     elif self.curr.lexeme == Lexeme.NUMBER:
    #         operand = ImmediateOp(self.number())
    #     elif self.curr.lexeme == Lexeme.LPAREN:
    #         self.consume_if(Lexeme.LPAREN)
    #         operand = self.expression()
    #         self.consume_if(Lexeme.RPAREN)
    #     elif self.curr.lexeme == Lexeme.CALL:
    #         operand = self.func_call()
    #     return operand
    #
    # def func_call(self) -> Operand:
    #     self.consume_if(Lexeme.CALL)
    #     ident = self.consume_if(Lexeme.IDENT)
    #     operands = []
    #     if self.curr.lexeme == Lexeme.LPAREN:
    #         self.consume_if(Lexeme.LPAREN)
    #         operands.append(self.expression())
    #         while self.curr.lexeme == Lexeme.COMMA:
    #             self.consume_if(Lexeme.COMMA)
    #             operands.append(self.expression())
    #         self.consume_if(Lexeme.RPAREN)
    #     return FuncCallOP(ident.value, operands)
    #
    #
    # def compute(self, symbol, lhs, rhs):
    #     pass


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
