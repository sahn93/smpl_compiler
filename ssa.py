from typing import List, Tuple, Dict, Optional
from enum import Enum


class SSACompileError(Exception):
    pass


class Operation(Enum):
    NEG = "neg"
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    CMP = "cmp"
    ADDA = "adda"
    LOAD = "load"
    STORE = "store"
    PHI = "phi"
    END = "end"
    BRA = "bra"
    BNE = "bne"
    BEQ = "beq"
    BLE = "ble"
    BLT = "blt"
    BGE = "bge"
    BGT = "bgt"
    READ = "read"
    WRITE = "write"
    WRITE_NL = "writeNL"
    CALL = "call"


class Operand:
    pass


class ArgumentOp(Operand):
    def __init__(self, ident: str):
        self.name = ident

    def __str__(self):
        return f"@{self.name}"

    def __eq__(self, other):
        return isinstance(other, ArgumentOp) and self.name == other.name


class ImmediateOp(Operand):
    def __init__(self, value: int):
        self.value = value

    def __str__(self):
        return f"#{self.value}"

    def __eq__(self, other):
        return isinstance(other, ImmediateOp) and self.value == other.value


class InstructionOp(Operand):
    def __init__(self, instr_i):
        self.value = instr_i

    def __str__(self):
        return f"({self.value})"

    def __eq__(self, other):
        return isinstance(other, InstructionOp) and self.value == other.value


class Instruction:
    def __init__(self, i: int, operation: Operation, *operands: Operand):
        self.i: int = i
        self.operation: Operation = operation
        self.operands: Tuple[Operand] = operands
        self.dominator: Optional[Instruction] = None    # Previous instruction with the same kind

    def __str__(self):
        if self.operands:
            return f"{self.i}: {self.operation.value} {', '.join([str(op) for op in self.operands])}"
        else:
            return f"{self.i}: {self.operation.value}"

    def __eq__(self, other):
        return (isinstance(other, Instruction) and
                self.operation == other.operation and
                self.operands == other.operands)


class Function:
    def __init__(self, num_operands: int):
        self.instr_counter = 0
        self.block_counter = 0  # A function's root block
        self.num_operands = num_operands

    def get_basic_block_id(self):
        self.block_counter += 1
        return self.block_counter

    def get_instr_id(self):
        self.instr_counter += 1
        return self.instr_counter


class BasicBlock:
    def __init__(self, func: Function, sym_table: Dict[str, Instruction]):
        self.func: Function = func
        self.basic_block_id: int = func.get_basic_block_id()
        self.instrs: List[Instruction] = []
        self.branch: Optional[BasicBlock] = None
        self.fall_through: Optional[BasicBlock] = None
        self.is_join_block: bool = False
        self.dominates: List[BasicBlock] = []
        self.sym_table: Dict[str, Instruction] = dict(sym_table)  # Copy symbol table from the dominator

    def dot(self, subgraph_id: Optional[int] = None) -> List[str]:
        subgraph_prefix = ""
        if subgraph_id is not None:
            subgraph_prefix = f"subgraph{subgraph_id}_"

        bfs_frontiers = [self]
        bfs_nodes = set()

        while bfs_frontiers:
            node = bfs_frontiers.pop()
            bfs_nodes.add(node)
            if self.fall_through:
                bfs_frontiers.append(self.fall_through)
            if self.branch:
                bfs_frontiers.append(self.branch)

        dotgraph_blocks = []
        dotgraph_branches = []
        dotgraph_doms = []

        for node in bfs_nodes:
            dot_block, dot_branches, dot_doms = node.dot_node(subgraph_prefix)
            dotgraph_blocks.append(dot_block)
            dotgraph_branches += dot_branches
            dotgraph_doms += dot_doms

        return dotgraph_blocks + dotgraph_branches + dotgraph_doms

    def dot_node(self, subgraph_prefix: str = ""):
        join = "join\n" if self.is_join_block else ""
        instrs = "|".join([str(instr) for instr in self.instrs])
        label = f"<b>{join}BB{self.basic_block_id}| {{{str(instrs)}}}"
        dot_block = f'\t{subgraph_prefix}bb{self.basic_block_id} [shape=record, label="{label}"];'

        dot_branches = []

        if self.fall_through:
            dot_branches.append(f'{subgraph_prefix}bb{self.basic_block_id}:s'
                                f' -> {subgraph_prefix}bb{self.fall_through}:n [label="fall-through"];')

        if self.branch:
            dot_branches.append(f'{subgraph_prefix}bb{self.basic_block_id}:s'
                                f' -> {subgraph_prefix}bb{self.branch}:n [label="branch"];')

        dot_doms = []
        for dom in self.dominates:
            dot_doms.append(f'{subgraph_prefix}bb{self.basic_block_id}:b'
                            f' -> bb{dom.basic_block_id}:b [color=blue, style=dotted, label="dom"];')

        return dot_block, dot_branches, dot_doms

    def emit(self, operation: Operation, *operands: Operand):
        instr = Instruction(self.func.get_instr_id(), operation, *operands)
        self.instrs.append(instr)


class SSA:
    def __init__(self):
        self.current_block: Optional[BasicBlock] = None
        self.current_func: Optional[Function] = None
        self.func_roots: Dict[str, BasicBlock] = dict()
        self.builtin_funcs: List[str] = ["InputNum", "OutputNum", "OutputNewLine"]
        # Add built-in functions
        read_block = self.get_new_function_block("InputNum", 0)
        read_block.emit(Operation.READ)
        write_block = self.get_new_function_block("OutputNum", 1)
        write_block.emit(Operation.WRITE, ArgumentOp("x"))
        output_nl_block = self.get_new_function_block("OutputNewLine", 0)
        output_nl_block.emit(Operation.WRITE_NL)

    def set_current_block(self, block: BasicBlock):
        self.current_block = block

    def get_new_function_block(self, name: str, num_operands: int) -> BasicBlock:
        func_root_bb = BasicBlock(Function(num_operands), dict())
        self.func_roots[name] = func_root_bb
        return func_root_bb

    def get_new_basicblock(self):
        return BasicBlock(self.current_func, dict(self.current_block.sym_table))

    def dot(self) -> str:
        dot_lines = ["digraph G {"]

        if len(self.func_roots) == 4:  # There's only main function except the builtin funcs -> no subgraph
            dot_lines += self.func_roots["main"].dot(None)
        else:  # There are more than 0 user define functions -> add functions as subgraphs
            for i, (func_name, func_root_block) in enumerate(self.func_roots.items()):
                if func_name in self.builtin_funcs:
                    continue    # Skip graphs for built-in functions
                dot_subgraph = [f"\tsubgraph cluster_{i} {{"]
                for line in func_root_block.dot(i):
                    dot_subgraph.append(f"\t{line}")  # Add indent for each subgraph
                dot_subgraph.append(f'\t\tlabel="{func_name}"')
                dot_subgraph.append("\t}")
                dot_lines += dot_subgraph

        dot_lines.append('}\n')

        return '\n'.join(dot_lines)


class FuncCallOp(Operand):
    def __init__(self, ident: str, *operands: Operand, ir: SSA):
        if ident not in ir.func_roots:
            raise SSACompileError(f"function {ident} is not declared.")
        if ir.func_roots[ident].func.num_operands != len(operands):
            raise SSACompileError(f"function {ident} expects {ir.func_roots[ident].func.num_operands}"
                                  f" operands, but got {len(operands)}.")

        self.ident = ident
        self.operands: Tuple[Operand] = operands

    def __str__(self):
        if self.ident == Operation.READ:
            return "read"
        elif self.ident == Operation.WRITE:
            return f"write {str(self.operands[0])}"
        elif self.ident == Operation.WRITE_NL:
            return "writeNewLine"
        else:
            return f"{self.ident}({', '.join([str(op) for op in self.operands])})"
