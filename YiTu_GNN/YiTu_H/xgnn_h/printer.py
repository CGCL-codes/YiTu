from xgnn_h import ir


class Printer:
    def dump(self, dataflow):
        count = [-1]
        node2nid = dict()

        #
        def visit_dfs(node: ir.Op) -> str:
            params = []
            for k in node.prevs:
                nid = visit_dfs(
                    node.prevs[k]
                )
                params.append(
                    '{}: %{}'.format(k, nid)
                )
            if isinstance(node, (ir.OpVertFunc,
                                 ir.OpEdgeFunc)):
                params.insert(
                    1, 'fn: {}'.format(node.func_name)
                )
            
            # printed
            if node in node2nid:
                return node2nid[node]

            # new line
            nid = None
            if node.name:
                nid = node.name
                nstr = '%' + nid
                node2nid[node] = nid
            else:
                count[0] += 1
                nid = str(count[0])
                node2nid[node] = nid
                nstr = '%{}'.format(nid)
            if node.size:
                nstr += ': {}'.format(
                    node.size
                )
            nstr += ' = {}'.format(
                type(node).__name__
            )
            if len(params) > 0:
                nstr += '('
                nstr += ', '.join(params)
                nstr += ')'
            print(nstr)

            if node.name:
                return node.name
            return count[0]

        #
        if isinstance(dataflow, dict):
            for k in dataflow:
                visit_dfs(dataflow[k])
        elif isinstance(dataflow, ir.Op):
            visit_dfs(dataflow)
        else:
            raise NotImplementedError
