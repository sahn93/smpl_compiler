import re
from enum import Enum


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
    RET = re.compile(r'return\s+')
    VAR = re.compile(r'var\s+')
    ARR = re.compile(r'array\s+')
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

        raise SmplLexerError(len(self.code), remaining_code, self.pos)


class Parser:
    def __init__(self, filepath):
        self.lexer = Lexer(filepath)
        self.curr = self.lexer.next()
        self.lookahead = self.lexer.next()

    def next(self):
        self.curr = self.lookahead
        self.lookahead = self.lexer.next()

    @staticmethod
    def error():
        raise SmplParserError

    def consume_if(self, *types):
        if self.curr in types:
            self.next()
        else:
            self.error()

    def computation(self):
        var_table = {}
        func_table = {}
        self.consume_if(Symbol.MAIN)
        self.next()
        while self.lookahead.token in [Symbol.ARR, Symbol.VAR]:
            var_id, val = self.var_decl()
            var_table[var_id] = val
        while self.lookahead.token in [Symbol.VOID, Symbol.FUNC]:
            func_id, func = self.func_decl()
            func_table[func_id] = func

        self.consume_if(Symbol.LBRACE)
        self.statSequence()
        self.consume_if(Symbol.RBRACE)
        self.consume_if(Symbol.PERIOD)

    def var_decl(self):
        return None, None

    def func_decl(self):
        return None, None


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
