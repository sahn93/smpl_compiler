#!/usr/bin/env python3
import argparse
from parser import *


class Result:
    def __init__(self, kind, val):
        self.kind = kind
        self.val = val


def main():
    parser = argparse.ArgumentParser(description="SMPL Compiler")
    parser.add_argument('smpl', metavar="smpl source file")
    args = parser.parse_args()

    smpl_parser = Parser(args.smpl)
    while smpl_parser.curr.symbol != Symbol.PERIOD:
        print(smpl_parser.curr.string, smpl_parser.curr.pos)
        smpl_parser.next()
    print(smpl_parser.curr.string)


if __name__ == "__main__":
    main()
