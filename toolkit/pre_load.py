# -*- coding: utf-8 -*-
import sys
import os
from toolkit.neo_models import Neo4j
from toolkit.tree_API import TREE
# 将项目路径添加到 Python 解释器的搜索路径中
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# 预加载neo4j
neo_con = Neo4j()
neo_con.connectDB()
print('neo4j connected!')

# 读取层次树
tree = TREE()
tree.read_edge('toolkit/bird_tree.txt')
tree.read_leaf('toolkit/bird_leaf.txt')

print('level tree load over~~~')
