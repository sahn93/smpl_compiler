import re


class SmplParserError(SyntaxError):
    pass


class SmplLexerError(SyntaxError):
    pass


class Token:
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

    patterns = [
        MAIN,
        VOID,
        FUNC,
        LET,
        CALL,
        IF,
        THEN,
        ELSE,
        FI,
        WHILE,
        DO,
        OD,
        RET,
        VAR,
        ARR,
        LPAREN,
        RPAREN,
        LBRACE,
        RBRACE,
        LBRACKET,
        RBRACKET,
        PERIOD,
        COMMA,
        SEMICOLON,
        LARROW,
        EQ,
        INEQ,
        LT,
        LE,
        GT,
        GE,
        ASTERISK,
        SLASH,
        PLUS,
        MINUS,
        IDENT,
        NUMBER,
    ]

    def __init__(self, token, string, pos):
        self.token = token
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
        for pat in Token.patterns:
            if isinstance(pat, re.Pattern):
                match = pat.match(remaining_code)
                if match:
                    val = match[0].strip()
                    self.pos += len(val)
                    while self.pos < self.code_len and self.code[self.pos].isspace():
                        self.pos += 1
                    return Token(pat, val.strip(), self.pos)
            else:
                if remaining_code.startswith(pat):
                    self.pos += len(pat)
                    while self.pos < self.code_len and self.code[self.pos].isspace():
                        self.pos += 1
                    return Token(pat, pat, self.pos)

        raise SmplLexerError


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
        self.consume_if(Token.MAIN)
        self.next()
        while self.lookahead.token in [Token.ARR, Token.VAR]:
            var_id, val = self.var_decl()
            var_table[var_id] = val
        while self.lookahead.token in [Token.VOID, Token.FUNC]:
            func_id, func = self.func_decl()
            func_table[func_id] = func

        self.consume_if(Token.LBRACE)
        self.statSequence()
        self.consume_if(Token.RBRACE)
        self.consume_if(Token.PERIOD)

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
