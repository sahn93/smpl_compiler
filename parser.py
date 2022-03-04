import re
from enum import Enum
from ssa import *


class SmplParserError(Exception):
    pass


class SmplLexerError(Exception):
    pass


class Symbol(Enum):
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


class Token:
    def __init__(self, symbol, string, pos):
        self.symbol = symbol
        self.string = string
        self.pos = pos


class Lexer:
    def __init__(self, filepath):
        with open(filepath, 'r') as f:
            self.code = f.read()
            self.code_len = len(self.code)
        self.pos = 0

    def next(self):
        remaining_code = self.code[self.pos:]
        for symbol in Symbol:
            pat = symbol.value
            if isinstance(pat, re.Pattern):
                match = pat.match(remaining_code)
                if match:
                    val = match[0].strip()
                    self.pos += len(val)
                    while self.pos < self.code_len and self.code[self.pos].isspace():
                        self.pos += 1
                    return Token(symbol, val.strip(), self.pos)
            else:
                if remaining_code.startswith(pat):
                    self.pos += len(pat)
                    while self.pos < self.code_len and self.code[self.pos].isspace():
                        self.pos += 1
                    return Token(symbol, pat, self.pos)

        raise SmplLexerError(self.pos, remaining_code)


class Parser:
    def __init__(self, filepath):
        self.lexer = Lexer(filepath)
        self.curr = self.lexer.next()
        self.root = BasicBlock(dict())  # init with empty value table.
        self.curr_block = self.root

    def parse(self):
        self.computation()
        return self.root

    def next(self):
        self.curr = self.lexer.next()

    @staticmethod
    def error(msg: str):
        raise SmplParserError(msg)

    def consume_if(self, *symbols: Symbol):
        if self.curr in symbols:
            consumed_token = self.curr
            self.next()
            return consumed_token
        else:
            self.error(f"Expected {symbols}, got {self.curr.symbol}")

    def computation(self):
        self.consume_if(Symbol.MAIN)
        while self.curr.symbol in [Symbol.VAR, Symbol.ARRAY]:
            self.var_decl()

        while self.curr.symbol in [Symbol.VOID, Symbol.FUNC]:
            self.func_decl()

        self.consume_if(Symbol.LBRACE)
        self.stat_sequence()
        self.consume_if(Symbol.RBRACE)
        self.consume_if(Symbol.PERIOD)

    def stat_sequence(self):
        self.statement()
        while self.curr.symbol == Symbol.SEMICOLON:
            self.consume_if(Symbol.SEMICOLON)
            self.statement()
        self.consume_if(Symbol.SEMICOLON)

    def statement(self):
        if self.curr.symbol == Symbol.LET:
            return self.assignment()
        elif self.curr.symbol == Symbol.CALL:
            return self.func_call()
        elif self.curr.symbol == Symbol.IF:
            return self.if_statement()
        elif self.curr.symbol == Symbol.WHILE:
            return self.while_statement()
        elif self.curr.symbol == Symbol.RETURN:
            return self.return_statement()
        else:
            self.error(f"Expected statement, got {self.curr.symbol}")

    def assignment(self):
        self.consume_if(Symbol.LET)
        ident, idx = self.designator()
        self.consume_if(Symbol.LARROW)
        rhs = self.expression()
        if len(idx) == 0:
            self.curr_block.sym_table[ident].value = rhs
        else:
            self.curr_block.sym_table[ident].value[idx] = rhs

    def var_decl(self):
        dim = self.type_decl()
        self.curr_block.sym_table[self.ident()] = SSAVariable(dim)
        while self.curr.symbol == Symbol.COMMA:
            self.consume_if(Symbol.COMMA)
            self.curr_block.sym_table[self.ident()] = SSAVariable(dim)
        self.consume_if(Symbol.SEMICOLON)

    def ident(self):
        return self.consume_if(Symbol.IDENT).string

    def number(self):
        return int(self.consume_if(Symbol.NUMBER).string)

    def type_decl(self):
        type_token = self.consume_if(Symbol.VAR, Symbol.ARRAY)
        dims = None
        if type_token.symbol == Symbol.ARRAY:
            dims = []
            self.consume_if(Symbol.LBRACKET)
            dims.append(self.number())
            self.consume_if(Symbol.RBRACKET)
            while self.curr.symbol == Symbol.LBRACKET:
                self.consume_if(Symbol.LBRACKET)
                dims.append(self.number())
                self.consume_if(Symbol.RBRACKET)
        return dims

    def func_decl(self):
        is_void = False
        if self.curr.symbol == Symbol.VOID:
            is_void = True
            self.consume_if(Symbol.VOID)
        self.consume_if(Symbol.FUNC)
        ident = self.ident()
        params = self.formal_param()
        self.consume_if(Symbol.SEMICOLON)
        local_variables, stmts = self.func_body()
        self.consume_if(Symbol.SEMICOLON)
        # TODO: Declare a function node and return it
        return ident, None

    def func_body(self):
        # TODO
        return None, None

    def formal_param(self):
        self.consume_if(Symbol.LPAREN)
        params = [self.ident()]
        while self.curr.symbol == Symbol.COMMA:
            self.consume_if(Symbol.COMMA)
            params.append(self.ident())
        self.consume_if(Symbol.RPAREN)
        return params

    def func_call(self):
        pass

    def if_statement(self):
        pass

    def while_statement(self):
        pass

    def return_statement(self):
        pass

    def designator(self):
        ident = self.consume_if(Symbol.IDENT)
        idxs = []
        while self.curr.symbol == Symbol.LBRACKET:
            self.consume_if(Symbol.LBRACKET)
            idxs.append(self.expression())
            self.consume_if(Symbol.RBRACKET)
        return ident, tuple(idxs)

    def expression(self) -> SSAValue:
        lhs = self.term()
        while self.curr.symbol in [Symbol.PLUS, Symbol.MINUS]:
            token = self.consume_if(Symbol.PLUS, Symbol.MINUS)
            rhs = self.term()
            self.compute(token.symbol, lhs, rhs)

        return lhs


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
