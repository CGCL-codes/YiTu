import torch
from xgnn_h import mp, ir, block
from tqdm import tqdm


class Stitcher:
    def _packing_ffd(self, weights, cap):
        bins = []
        n_bins = 0
        weights = sorted(
            weights,
            key=lambda x: x[-1],
            reverse=True
        )
        bin_remains = [0] * len(weights)
        for item in weights:
            w = item[-1]
            for i in range(n_bins):
                if w <= bin_remains[i]:
                    bin_remains[i] -= w
                    bins[i].append(item)
                    break
            else:
                bin_remains[n_bins] = cap - w
                bins.append([item])
                n_bins += 1
        assert n_bins == len(bins)
        while len(bins) > 2:
            bin_sizes = [
                sum([x[-1] for x in group])
                for group in bins
            ]
            extra = bins.pop(-1)
            if bin_sizes[0] < bin_sizes[1]:
                bins[0].extend(extra)
            else:
                bins[1].extend(extra)
        return bins

    def _build_hgraph(self, hgraph: mp.HeteroGraph, stitch_rules: dict):
        stitch_map = {}
        new_hgraph = mp.HeteroGraph()
        new_hgraph.device = hgraph.device
        new_hgraph.nty2num = hgraph.nty2num
        for idx, stitches in stitch_rules.items():
            # stitch
            n_edges = 0
            block_list = []
            for sty, ety, dty, num in stitches:
                n_edges += num
                old_blk = hgraph.hetero_graph[
                    sty, ety, dty
                ].blk
                block_list.append(old_blk)
            # concat
            n_rows = hgraph.nty2num[dty]
            new_blk = block.stitch_csr(
                block_list
            ).to('cuda')
            assert new_blk.size[0] == n_rows
            assert len(new_blk.adj_sparse[1]) == n_edges
            # new name
            new_sty = '-->'.join(
                item[0] for item in stitches
            )
            new_ety = '-->'.join(
                item[1] for item in stitches
            )
            new_hgraph.idx2rel[
                len(new_hgraph.idx2rel)
            ] = [new_sty, new_ety, dty]
            new_hgraph.rel2idx[
                new_sty, new_ety, dty
            ] = len(new_hgraph.rel2idx)
            new_hgraph.hetero_graph[
                new_sty, new_ety, dty
            ] = mp.Graph(new_blk)
            stitch_map[idx] = (
                new_sty, new_ety, dty
            )
        return new_hgraph, stitch_map

    def _replace_spmm(self,
                      spmm_nodes: set,
                      hgraph: mp.HeteroGraph,
                      stitch_rules: dict,
                      stitch_map: dict,
                      new_hgraph: mp.HeteroGraph):
        #
        def match_rule(stgt, etgt, dtgt):
            for idx, stitches in stitch_rules.items():
                for sty, ety, dty, _ in stitches:
                    if sty == stgt and \
                            ety == etgt and \
                            dty == dtgt:
                        return idx
            return -1

        stitch_idxes = set()
        for spmm_node in spmm_nodes:
            graph_node = spmm_node.prevs['g']
            splited = graph_node.name.split('.')
            matched_idx = match_rule(
                *hgraph.idx2rel[int(splited[1])]
            )
            if matched_idx == -1:
                continue
            stitch_idxes.add(matched_idx)

        #
        def match_spmms(stitches):
            node_res = []
            for stgt, etgt, dtgt, _ in stitches:
                for spmm_node in spmm_nodes:
                    graph_node = spmm_node.prevs['g']
                    splited = graph_node.name.split('.')
                    sty, ety, dty = hgraph.idx2rel[int(splited[1])]
                    if sty == stgt and \
                            ety == etgt and \
                            dty == dtgt:
                        node_res.append(spmm_node)
            assert len(stitches) == len(node_res)
            return node_res

        #
        stitched_nodes = []
        for idx in stitch_idxes:
            stitches = stitch_rules[idx]
            stitch_rel = stitch_map[idx]
            stitch_spmms = match_spmms(stitches)
            new_graph = ir.OpGraph(
                new_hgraph.hetero_graph[stitch_rel].blk,
                name='stitch.{}'.format(idx)
            )
            # replace nodes
            sddmm_scheme = None
            stitch_queries = []
            stitch_keys, stitch_values = [], []
            for spmm_node in stitch_spmms:
                graph_node = spmm_node.prevs['g']
                value_node = spmm_node.prevs['x']
                stitch_values.append(value_node)
                if 'e' not in spmm_node.prevs:
                    continue
                sddmm_node = spmm_node.prevs['e']
                if not isinstance(sddmm_node, ir.OpFusedSDDMM):
                    raise NotImplementedError
                sddmm_scheme = sddmm_node.fusion_scheme
                query_node = sddmm_node.prevs['q']
                key_node = sddmm_node.prevs['k']
                #
                assert graph_node.size[0] == query_node.size[0]
                assert graph_node.size[1] == value_node.size[0]
                assert graph_node.size[1] == key_node.size[0]
                stitch_queries.append(query_node)
                stitch_keys.append(key_node)
            # pack it
            new_value = ir.OpConcat(xs=stitch_values, dim=0)
            assert new_value.size[0] == new_graph.size[1]
            new_key, new_query = None, None
            if stitch_keys and stitch_queries:
                new_key = ir.OpConcat(xs=stitch_keys, dim=0)
                new_query = ir.OpStack(xs=stitch_queries, dim=0)
                assert new_query.size[0] == len(stitch_queries)
                assert new_query.size[1] == new_graph.size[0]
                assert new_key.size[0] == new_graph.size[1]
            # new node
            new_sddmm = None
            if stitch_keys and stitch_queries:
                n_edges = sum([
                    item[-1]
                    for item in stitches
                ])
                new_sddmm = ir.OpFusedSDDMM(
                    size=[n_edges, new_key.size[-1]],
                    graph=new_graph, query=new_query,
                    key=new_key, fusion_scheme=sddmm_scheme
                )
            new_spmm = ir.OpFusedSPMM(
                graph=new_graph,
                edge=new_sddmm,
                x=new_value
            )
            stitched_nodes.append(new_spmm)

        return stitched_nodes

    def _stitch_hetg(self, dataflow: ir.Op, kwargs: dict):
        # group by dty
        many2one = {}
        hgraph: mp.HeteroGraph = kwargs['hgraph']
        for sty, ety, dty in hgraph.hetero_graph:
            if dty not in many2one:
                many2one[dty] = []
            blk: block.Block = hgraph.hetero_graph[
                sty, ety, dty
            ].blk
            many2one[dty].append(
                [sty, ety, dty, blk.num_edges()]
            )

        # bin packing
        stitch_rules = {}
        for het_weights in many2one.values():
            if len(het_weights) <= 2:
                continue
            bin_cap = sum(
                [x[-1] for x in het_weights]
            )
            for candidate in self._packing_ffd(
                    het_weights, cap=bin_cap // 2):
                if len(candidate) == 1:
                    continue
                if len(candidate) > 16:
                    continue
                idx = len(stitch_rules)
                stitch_rules[idx] = candidate

        # build graph
        new_hgraph, stitch_map = self._build_hgraph(
            hgraph, stitch_rules=stitch_rules
        )
        assert len(stitch_rules) == len(stitch_map)
        assert len(new_hgraph.hetero_graph) == len(stitch_map)

        #
        def visit_scale(root_node: ir.Op,
                        scale_nodes=None):
            if scale_nodes is None:
                scale_nodes = set()
            for name in root_node.prevs:
                visit_scale(root_node.prevs[name],
                            scale_nodes)
            #
            if isinstance(root_node, ir.OpScale):
                scale_nodes.add(root_node)
            return scale_nodes
        scale_nodes = visit_scale(dataflow)

        #
        def visit_spmm(root_nodes: list,
                       spmm_nodes=None,
                       depth=0):
            if depth >= 5:
                return
            if spmm_nodes is None:
                spmm_nodes = set()
            for node in root_nodes:
                if isinstance(node, ir.OpFusedSPMM):
                    spmm_nodes.add(node)
                    continue
                for child in node.prevs.values():
                    if isinstance(child, ir.OpAdd):
                        continue
                    visit_spmm([child],
                               spmm_nodes,
                               depth + 1)
            return spmm_nodes

        def visit_accumulate(root_node: ir.Op,
                             accum_nodes=None):
            if accum_nodes is None:
                accum_nodes = set()
            for child in root_node.prevs.values():
                if not isinstance(child, ir.OpAdd):
                    continue
                visit_accumulate(child, accum_nodes)
                accum_nodes.add(child)
            return accum_nodes

        # begin stitching
        for scale_node in scale_nodes:
            accum_nodes = visit_accumulate(scale_node)
            if not accum_nodes:
                continue
            spmm_nodes = visit_spmm(accum_nodes)
            assert len(spmm_nodes) == len(accum_nodes) + 1
            stitched_node = self._replace_spmm(
                spmm_nodes=spmm_nodes, hgraph=hgraph,
                stitch_rules=stitch_rules, stitch_map=stitch_map,
                new_hgraph=new_hgraph
            )
            new_accum = None
            if len(stitched_node) == 0:
                continue
            elif len(stitched_node) == 1:
                new_accum = stitched_node[0]
            elif len(stitched_node) == 2:
                new_accum = ir.OpAdd(
                    stitched_node[0],
                    stitched_node[1]
                )
            else:
                raise RuntimeError
            if len(new_accum.size) == 3:
                new_accum = ir.OpMean(
                    new_accum, dim=1
                )
            scale_node.prevs['x'] = new_accum

        kwargs['stitch'] = new_hgraph
        return dataflow

    def transform(self, dataflow, kwargs: dict):
        if isinstance(dataflow, ir.Op):
            dataflow = self._stitch_hetg(
                dataflow, kwargs=kwargs
            )
            return dataflow
        else:
            raise NotImplementedError
