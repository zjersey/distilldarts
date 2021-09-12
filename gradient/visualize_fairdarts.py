
import genotypes
import json
from graphviz import Digraph
import argparse
import os

parser = argparse.ArgumentParser("visualization")
parser.add_argument('--exp_dir', type=str, required=True)
parser.add_argument('--model_number', type=int, default=49)
parser.add_argument('--normal_name', type=str, default='normal')
parser.add_argument('--reduction_name', type=str, default='reduction')

args = parser.parse_args()


def plot(genotype, filename):
  g = Digraph(
      format='pdf',
      edge_attr=dict(fontsize='20', fontname="times"),
      node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5', penwidth='2', fontname="times"),
      engine='dot')
  g.body.extend(['rankdir=LR'])

  g.node("c_{k-2}", fillcolor='darkseagreen2')
  g.node("c_{k-1}", fillcolor='darkseagreen2')

  node_set = []
  for [op, node1, node2] in genotype:
    node_min = min(node1,node2)
    node_max = max(node1,node2)
    if node_min == 0:
      u = "c_{k-2}"
    elif node_min == 1:
      u = "c_{k-1}"
    else:
      if node_min not in node_set:
        g.node(str(node_min-2), fillcolor='lightblue')
        node_set.append(node_min)
      u = str(node_min-2)

    if node_max not in node_set:
      g.node(str(node_max-2), fillcolor='lightblue')
      node_set.append(node_max)
    v = str(node_max-2)
    g.edge(u, v, label=op, fillcolor="gray")


  g.node("c_{k}", fillcolor='palegoldenrod')
  for i in node_set:
    g.edge(str(i-2), "c_{k}", fillcolor="gray")

  g.render(filename, view=False)


if __name__ == '__main__':

  genotype_file = os.path.join(args.exp_dir, 'results_of_7q', 'genotype', '%d.txt'%args.model_number)
  tmp_dict = json.load(open(genotype_file, 'r'))
  genotype = genotypes.Genotype(**tmp_dict)
  #plot(genotype.normal, args.normal_name)
  plot(genotype.reduce, args.reduction_name)

