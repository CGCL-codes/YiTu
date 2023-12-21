//
// A demo program of reordering using Rabbit Order.
//
// Author: ARAI Junya <arai.junya@lab.ntt.co.jp> <araijn@gmail.com>
//

#include <boost/range/adaptor/transformed.hpp>
#include <boost/range/algorithm/count.hpp>

#include "rabbit_order.hpp"
#include "edge_list.hpp"

using rabbit_order::vint;
typedef std::vector<std::vector<std::pair<vint, float> > > adjacency_list;

vint count_unused_id(const vint n, const std::vector<edge_list::edge>& edges) {
  std::vector<char> appears(n);
  for (size_t i = 0; i < edges.size(); ++i) {
    appears[std::get<0>(edges[i])] = true;
    appears[std::get<1>(edges[i])] = true;
  }
  return static_cast<vint>(boost::count(appears, false));
}

template<typename RandomAccessRange>
adjacency_list make_adj_list(const vint n, const RandomAccessRange& es) {
  using std::get;

  // Symmetrize the edge list and remove self-loops simultaneously
  std::vector<edge_list::edge> ss(boost::size(es) * 2);
  #pragma omp parallel for
  for (size_t i = 0; i < boost::size(es); ++i) {
    auto& e = es[i];
    if (get<0>(e) != get<1>(e)) {
      ss[i * 2    ] = std::make_tuple(get<0>(e), get<1>(e), get<2>(e));
      ss[i * 2 + 1] = std::make_tuple(get<1>(e), get<0>(e), get<2>(e));
    } else {
      // Insert zero-weight edges instead of loops; they are ignored in making
      // an adjacency list
      ss[i * 2    ] = std::make_tuple(0, 0, 0.0f);
      ss[i * 2 + 1] = std::make_tuple(0, 0, 0.0f);
    }
  }

  // Sort the edges
  __gnu_parallel::sort(ss.begin(), ss.end());

  // Convert to an adjacency list
  adjacency_list adj(n);
  #pragma omp parallel
  {
    // Advance iterators to a boundary of a source vertex
    const auto adv = [](auto it, const auto first, const auto last) {
      while (first != it && it != last && get<0>(*(it - 1)) == get<0>(*it))
        ++it;
      return it;
    };

    // Compute an iterator range assigned to this thread
    const int    p      = omp_get_max_threads();
    const size_t t      = static_cast<size_t>(omp_get_thread_num());
    const size_t ifirst = ss.size() / p * (t)   + std::min(t,   ss.size() % p);
    const size_t ilast  = ss.size() / p * (t+1) + std::min(t+1, ss.size() % p);
    auto         it     = adv(ss.begin() + ifirst, ss.begin(), ss.end());
    const auto   last   = adv(ss.begin() + ilast,  ss.begin(), ss.end());

    // Reduce edges and store them in std::vector
    while (it != last) {
      const vint s = get<0>(*it);

      // Obtain an upper bound of degree and reserve memory
      const auto maxdeg = 
          std::find_if(it, last, [s](auto& x) {return get<0>(x) != s;}) - it;
      adj[s].reserve(maxdeg);

      while (it != last && get<0>(*it) == s) {
        const vint t = get<1>(*it);
        float      w = 0.0;
        while (it != last && get<0>(*it) == s && get<1>(*it) == t)
          w += get<2>(*it++);
        if (w > 0.0)
          adj[s].push_back({t, w});
      }

      // The actual degree can be smaller than the upper bound
      adj[s].shrink_to_fit();
    }
  }

  return adj;
}

adjacency_list read_graph(const std::string& graphpath) {
  const auto edges = edge_list::read(graphpath);

  // The number of vertices = max vertex ID + 1 (assuming IDs start from zero)
  const auto n =
      boost::accumulate(edges, static_cast<vint>(0), [](vint s, auto& e) {
          return std::max(s, std::max(std::get<0>(e), std::get<1>(e)) + 1);});

  if (const size_t c = count_unused_id(n, edges)) {
    std::cerr << "WARNING: " << c << "/" << n << " vertex IDs are unused"
              << " (zero-degree vertices or noncontiguous IDs?)\n";
  }

  return make_adj_list(n, edges);
}

template<typename InputIt>
typename std::iterator_traits<InputIt>::difference_type
count_uniq(const InputIt f, const InputIt l) {
  std::vector<typename std::iterator_traits<InputIt>::value_type> ys(f, l);
  return boost::size(boost::unique(boost::sort(ys)));
}

double compute_modularity(const adjacency_list& adj, const vint* const coms) {
  const vint  n    = static_cast<vint>(adj.size());
  const auto  ncom = count_uniq(coms, coms + n);
  double      m2   = 0.0;  // total weight of the (bidirectional) edges

  std::unordered_map<vint, double[2]> degs(ncom);  // ID -> {all, loop}
  degs.reserve(ncom);

  #pragma omp parallel reduction(+:m2)
  {
    std::unordered_map<vint, double[2]> mydegs(ncom);
    mydegs.reserve(ncom);

    #pragma omp for
    for (vint v = 0; v < n; ++v) {
      const vint  c = coms[v];
      auto* const d = &mydegs[c];
      for (const auto e : adj[v]) {
        m2      += e.second;
        (*d)[0] += e.second;
        if (coms[e.first] == c) (*d)[1] += e.second;
      }
    }

    #pragma omp critical
    {
      for (auto& kv : mydegs) {
        auto* const d = &degs[kv.first];
        (*d)[0] += kv.second[0];
        (*d)[1] += kv.second[1];
      }
    }
  }
  assert(static_cast<intmax_t>(degs.size()) == ncom);

  double q = 0.0;
  for (auto& kv : degs) {
    const double all  = kv.second[0];
    const double loop = kv.second[1];
    q += loop / m2 - (all / m2) * (all / m2);
  }

  return q;
}

void detect_community(adjacency_list adj) {
  auto _adj = adj;  // copy `adj` because it is used for computing modularity

  std::cerr << "Detecting communities...\n";
  const double tstart = rabbit_order::now_sec();
  //--------------------------------------------
  auto       g = rabbit_order::aggregate(std::move(_adj));
  const auto c = std::make_unique<vint[]>(g.n());
  #pragma omp parallel for
  for (vint v = 0; v < g.n(); ++v)
    c[v] = rabbit_order::trace_com(v, &g);
  //--------------------------------------------
  std::cerr << "Runtime for community detection [sec]: "
            << rabbit_order::now_sec() - tstart << std::endl;

  // Print the result
  std::copy(&c[0], &c[g.n()], std::ostream_iterator<vint>(std::cout, "\n"));

  std::cerr << "Computing modularity of the result...\n";
  const double q = compute_modularity(adj, c.get());
  std::cerr << "Modularity: " << q << std::endl;
}

void reorder(adjacency_list adj) {
  std::cerr << "Generating a permutation...\n";
  const double tstart = rabbit_order::now_sec();
  //--------------------------------------------
  const auto g = rabbit_order::aggregate(std::move(adj));
  const auto p = rabbit_order::compute_perm(g);
  //--------------------------------------------
  std::cerr << "Runtime for permutation generation [sec]: "
            << rabbit_order::now_sec() - tstart << std::endl;

  // Print the result
  std::copy(&p[0], &p[g.n()], std::ostream_iterator<vint>(std::cout, "\n"));
}





int main(int argc, char* argv[]) {
  using boost::adaptors::transformed;

  // Parse command-line arguments
  if (argc != 2 && (argc != 3 || std::string("-c") != argv[1])) {
    std::cerr << "Usage: reorder [-c] GRAPH_FILE\n"
              << "  -c    Print community IDs instead of a new ordering\n";
    exit(EXIT_FAILURE);
  }
  const std::string graphpath = argc == 3 ? argv[2] : argv[1];
  const bool        commode   = argc == 3;

  std::cerr << "Number of threads: " << omp_get_max_threads() << std::endl;

  std::cerr << "Reading an edge-list file: " << graphpath << std::endl;


  auto       adj = read_graph(graphpath);
  const auto m   =
      boost::accumulate(adj | transformed([](auto& es) {return es.size();}),
                        static_cast<size_t>(0));
  std::cerr << "Number of vertices: " << adj.size() << std::endl;
  std::cerr << "Number of edges: "    << m          << std::endl;

  if (commode)
    detect_community(std::move(adj));
  else
    reorder(std::move(adj));

  return EXIT_SUCCESS;
}

