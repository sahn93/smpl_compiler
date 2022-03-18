#!/usr/bin/env python3
import argparse
from parser import Parser
from ssa import SSA
from register_allocator import RegisterAllocator


class Result:
    def __init__(self, kind, val):
        self.kind = kind
        self.val = val


def main():
    arg_parser = argparse.ArgumentParser(description="SMPL Compiler")
    arg_parser.add_argument('smpl', metavar="smpl source file")
    arg_parser.add_argument("--ssa", default=False, action="store_true", help="Produce SSA graph.")
    arg_parser.add_argument("--reg", default=False, action="store_true"
                            , help="Produce SSA graph after register assignment.")

    args = arg_parser.parse_args()

    ir = SSA()

    smpl_parser = Parser(args.smpl, ir)
    smpl_parser.parse()
    if args.ssa:
        print(ir.dot())

    if args.reg:
        register_allocator = RegisterAllocator(ir)
        register_allocator.allocate_registers()
        print(ir.dot())


if __name__ == "__main__":
    main()
