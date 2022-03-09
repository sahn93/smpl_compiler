#!/usr/bin/env python3
import argparse
from parser import Parser
from ssa import SSA



class Result:
    def __init__(self, kind, val):
        self.kind = kind
        self.val = val


def main():
    parser = argparse.ArgumentParser(description="SMPL Compiler")
    parser.add_argument('smpl', metavar="smpl source file")
    args = parser.parse_args()

    ir = SSA()
    smpl_parser = Parser(args.smpl, ir)

    # Test Lexer
    # while smpl_parser.curr.lexeme != Lexeme.PERIOD:
    #     print(smpl_parser.curr.value, smpl_parser.curr.row, smpl_parser.curr.col)
    #     smpl_parser.next()
    # print(smpl_parser.curr.value)

    # smpl_parser.parse()
    print(ir.dot())


if __name__ == "__main__":
    main()
