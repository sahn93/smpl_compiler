import ssa

from typing import Set, List, Optional, Dict
from enum import Enum
from heapq import heapify, heappush, heappop


class Register(Enum):
    ZERO = 0
    RET_ADDRESS = 31
    GLOBAL_VAR_POINTER = 30
    STACK_POINTER = 29
    FRAME_POINTER = 28
    MEM_SWAP_REGISTER = 27


class Node:
    def __init__(self):
        self.adj_nodes: Set[Node] = set()
        self.color: int = -1
        self.cost: int = 0

    def __lt__(self, other):
        if not isinstance(other, Node):
            raise TypeError
        return self.cost < other.cost


class InstructionNode(Node):
    def __init__(self, instr: ssa.Instruction):
        super().__init__()
        self.instr: ssa.Instruction = instr

    def __eq__(self, other):
        return isinstance(other, InstructionNode) and self.instr.i == other.instr.i

    def __hash__(self):
        return self.instr.i

    def __str__(self):
        return f"instr_{self.instr.i}"


class ClusterNode(Node):
    def __init__(self, phi_node: Node):
        super().__init__()
        if isinstance(phi_node, ClusterNode):
            self.instr_nodes: Set[InstructionNode] = phi_node.instr_nodes
        elif isinstance(phi_node, InstructionNode):
            self.instr_nodes: Set[InstructionNode] = {phi_node}
        else:
            raise TypeError
        # Override phi_node's edges
        self.merge(phi_node)

    def __hash__(self):
        return -min([hash(instr_node) for instr_node in self.instr_nodes])

    def __eq__(self, other):
        return isinstance(other, ClusterNode) and self.instr_nodes == other.instr_nodes

    def merge(self, node: Node):
        self.cost += node.cost
        if isinstance(node, ClusterNode):
            self.instr_nodes |= node.instr_nodes
        elif isinstance(node, InstructionNode):
            self.instr_nodes.add(node)
        else:
            raise TypeError
        for adj_node in node.adj_nodes:
            adj_node.adj_nodes.discard(node)
            adj_node.adj_nodes.add(self)
            self.adj_nodes.add(adj_node)
        return self

    def __str__(self):
        return f"Cluster: {', '.join([str(instr_node) for instr_node in self.instr_nodes])}"


class RegisterAllocator:

    def __init__(self, ir: ssa.SSA):
        self.ir = ir
        self.phis: Dict[ssa.Instruction, ssa.BasicBlock] = {}
        self.id_to_node: Dict[int, Node] = {}
        self.nodes_min_heap: List[Node] = []
        heapify(self.nodes_min_heap)
        self.curr_color = 1

    def allocate_registers(self) -> None:
        # Allocate registers for each function. Dump all the registers before a function call.
        for func_name, root_block in self.ir.func_roots.items():
            # Skip for built-in functions
            if func_name in self.ir.builtin_funcs:
                continue

            # Build an interference graph from the last block without resolving phis.
            self.build_interference_graph(root_block.func.last_block)

            # # Print graph for debugging.
            # for i, node in self.id_to_node.items():
            #     print(i, node, [str(adj_node) for adj_node in node.adj_nodes])

            # Resolve all phi functions
            self.resolve_phis()

            # Build a min_heap of nodes.
            for node in self.id_to_node.values():
                if node not in self.nodes_min_heap:
                    heappush(self.nodes_min_heap, node)

            # # Print graph for debugging.
            # print("After phi-resolving")
            # for node in self.nodes_min_heap:
            #     print(node, ", adj: ", [str(adj_node) for adj_node in node.adj_nodes])

            # Allocate register for each node.
            self.color()

            # Remove Phis.
            self.remove_phis()

            # Init for the next function
            self.phis = {}
            self.id_to_node = {}
            self.nodes_min_heap = []
            heapify(self.nodes_min_heap)
            self.curr_color = 1

    def build_interference_graph(self, basic_block: Optional[ssa.BasicBlock]):
        """
        Get live set of given basic block.
        :param basic_block: A basic block to get a live set.
        """
        if basic_block.branch_block and not basic_block.branch_block.backward_pass_visited:
            if basic_block.type == ssa.BasicBlockType.WHILE_JOIN:
                phi_lhs, phi_rhs = self.basic_block_backward_pass(basic_block)
                branch_last_block = basic_block.preds[1]
                branch_last_block.live_set = basic_block.live_set - phi_lhs
                self.build_interference_graph(branch_last_block)
            else:
                # If-join block and branch block is not passed yet. Stop and restart from the other branch.
                pass

        elif basic_block.type == ssa.BasicBlockType.IF_JOIN:
            phi_lhs, phi_rhs = self.basic_block_backward_pass(basic_block)
            basic_block.backward_pass_visited = True
            basic_block.preds[0].live_set -= phi_rhs
            self.build_interference_graph(basic_block.preds[0])  # will stop before the original block.
            basic_block.preds[1].live_set -= phi_lhs
            self.build_interference_graph(basic_block.preds[1])  # will continue upwards

        elif basic_block.type == ssa.BasicBlockType.WHILE_JOIN:
            phi_lhs, phi_rhs = self.basic_block_backward_pass(basic_block)
            basic_block.backward_pass_visited = True
            pred_block = basic_block.preds[0]
            pred_block.live_set = basic_block.live_set - phi_rhs
            self.build_interference_graph(pred_block)

        else:
            # If if-statements' original block, merge successors' live set.
            if basic_block.branch_block and basic_block.fall_through_block:
                basic_block.live_set |= basic_block.fall_through_block.live_set
                basic_block.live_set |= basic_block.branch_block.live_set

            # Pass current block
            self.basic_block_backward_pass(basic_block)
            basic_block.backward_pass_visited = True

            # Iterate through the predecessor if exists.
            if basic_block.preds:
                pred_block = basic_block.preds[0]
                pred_block.live_set = basic_block.live_set
                self.build_interference_graph(pred_block)

    def basic_block_backward_pass(self, basic_block: ssa.BasicBlock):
        # Backward pass a basic block and add live values to the live set.
        phi_lhs: Set[ssa.Instruction] = set()
        phi_rhs: Set[ssa.Instruction] = set()

        # Move immediate values in phi functions to the registers in the parent block.
        for instr in basic_block.instrs:
            if instr.operation != ssa.Operation.PHI:
                break
            for i, op in enumerate(instr.operands):
                if isinstance(op, ssa.ImmediateOp):
                    instr.operands[i] = basic_block.preds[i].emit(ssa.Operation.ADD, ssa.ImmediateOp(0), op)

        for instr in reversed(basic_block.instrs):
            # If there is no occurrence until it is removed,
            if instr.operation == ssa.Operation.PHI:
                self.phis[instr] = basic_block
                lhs, rhs = instr.operands
                if isinstance(lhs, ssa.InstructionOp):
                    phi_lhs.add(lhs.instr)
                if isinstance(rhs, ssa.InstructionOp):
                    phi_rhs.add(rhs.instr)

            if instr.i in self.id_to_node:
                # 1. Discard this instruction from the live set and the graph.
                basic_block.live_set.discard(instr)

                # 2. Add interference edges between the instruction and other live values.
                instr_node = self.id_to_node[instr.i]
                for live_instr in basic_block.live_set:
                    live_node = self.id_to_node[live_instr.i]
                    instr_node.adj_nodes.add(live_node)
                    live_node.adj_nodes.add(instr_node)
            else:
                # Perform Dead Code Elimination if available.
                if instr.operation not in ssa.void_operations + [ssa.Operation.CALL, ssa.Operation.PHI]:
                    instr.is_dead = True

            # 3. Add operand values into the live set.
            if instr.operation == ssa.Operation.CALL:
                # if there is a func call instruction, backward pass the called function
                func_call_op = instr.operands[0]
                if isinstance(func_call_op, ssa.FuncCallOp):
                    operands = func_call_op.args
                else:
                    raise ValueError(f"Expected {ssa.FuncCallOp}, but caught {func_call_op}")
            else:
                operands = instr.operands

            for op in operands:
                # Check whether the op is a value.
                if isinstance(op, ssa.InstructionOp) and not op.instr.is_void:
                    # If there is no node for this instruction, make one.
                    if op.instr.i not in self.id_to_node:
                        self.id_to_node[op.instr.i] = InstructionNode(op.instr)

                    # Add instruction to the live set.
                    basic_block.live_set.add(op.instr)

                    # Add cost to the node.
                    self.id_to_node[op.instr.i].cost += basic_block.factor

        return phi_lhs, phi_rhs

    def color(self):
        if not self.nodes_min_heap:
            return
        # Remove node from the graph
        min_node = heappop(self.nodes_min_heap)
        for adj_node in min_node.adj_nodes:  # Remove edges from the graph
            adj_node.adj_nodes.discard(min_node)

        # Color recursively, the higher cost it is, the first it is colored.
        self.color()

        # Restore the node
        heappush(self.nodes_min_heap, min_node)
        for adj_node in min_node.adj_nodes:  # Restore the edges
            adj_node.adj_nodes.add(min_node)

        if self.curr_color < Register.MEM_SWAP_REGISTER.value:
            # print(f"Assign R{self.curr_color} to the {min_node}")
            min_node.color = self.curr_color  # Assign a register if color is less than 27 (Memory swap register)
        else:
            min_node.color = self.curr_color + 5  # colors start from 32 are for virtual registers.

        # Update instruction labels
        if isinstance(min_node, InstructionNode):
            min_node.instr.register = min_node.color
        elif isinstance(min_node, ClusterNode):
            for instr_node in min_node.instr_nodes:
                instr_node.instr.register = min_node.color

        self.curr_color += 1

    def resolve_phis(self):
        for phi, basic_block in self.phis.items():
            # Remove phi from the graph and replace it to a cluster node.
            if phi.i not in self.id_to_node:
                self.id_to_node[phi.i] = InstructionNode(phi)
            cluster_node = ClusterNode(self.id_to_node[phi.i])
            self.id_to_node[phi.i] = cluster_node

            for i, op in enumerate(phi.operands):
                if isinstance(op, ssa.InstructionOp) and not op.instr.is_void:
                    op_node = self.id_to_node[op.instr.i]
                    if op_node not in cluster_node.adj_nodes:
                        # Merge the node to the cluster.
                        cluster_node.merge(op_node)
                        self.id_to_node[op.instr.i] = cluster_node

    def remove_phis(self):
        for phi, basic_block in self.phis.items():
            for i, op in enumerate(phi.operands):
                if isinstance(op, ssa.InstructionOp) and op.instr.register != phi.register:
                    parent_block = basic_block.preds[i]
                    instr_op = parent_block.emit(ssa.Operation.ADD, ssa.ImmediateOp(0), op)
                    instr_op.instr.register = phi.register
            phi.is_dead = True
