import unittest
import numpy as np
from sim.a4.data import decode_maze,encode_maze_mask
from sim.a4.maze import bfs_oracle,wave_parent,parent_path,maze_gate

class MazeGate(unittest.TestCase):
    def fixture(self):
        rgb=np.zeros((3,3,5),dtype=int)
        rgb[:,1,:]=1
        rgb[:,1,0]=[0,1,0]; rgb[:,1,4]=[1,0,0]
        target=np.zeros((3,5),dtype=int); target[1,:]=1
        return rgb,target

    def test_roundtrip_and_unique_path(self):
        rgb,target=self.fixture(); raw=encode_maze_mask(rgb); labels=encode_maze_mask(target)
        decoded=decode_maze(raw,labels)
        np.testing.assert_equal(decoded['rgb'],rgb)
        np.testing.assert_equal(decoded['target'],target)
        np.testing.assert_equal(bfs_oracle(decoded['open'],decoded['source'],decoded['goal']),target)
        np.testing.assert_equal(wave_parent(decoded['open'],decoded['source'],decoded['goal']),target)
        self.assertEqual(maze_gate([raw],[labels])['count'],1)

    def test_cycle_and_wrong_parent_rejected(self):
        open_cells=np.ones((2,2),bool); source=np.zeros((2,2),bool); source[0,0]=1
        goal=np.zeros((2,2),bool); goal[1,1]=1
        with self.assertRaises(ValueError): bfs_oracle(open_cells,source,goal)
        with self.assertRaises(ValueError): parent_path({(1,1):(0,0)},source,goal,open_cells)

    def test_bend_branch_and_disconnected(self):
        for cells in ([(0,0),(1,0),(1,1)],[(0,1),(1,0),(1,1),(1,2),(2,1)]):
            opened=np.zeros((3,3),bool)
            for p in cells: opened[p]=1
            source=np.zeros_like(opened); source[cells[0]]=1
            goal=np.zeros_like(opened); goal[cells[-1]]=1
            np.testing.assert_equal(wave_parent(opened,source,goal),bfs_oracle(opened,source,goal))
        opened[2,2]=1; opened[2,1]=0
        with self.assertRaises(ValueError): bfs_oracle(opened,source,goal)

    def test_broken_block_and_endpoint_rejected(self):
        rgb,target=self.fixture(); raw=encode_maze_mask(rgb); labels=encode_maze_mask(target)
        raw[0,3,3]^=1
        with self.assertRaises(ValueError): decode_maze(raw,labels)
        rgb[:,1,4]=1
        with self.assertRaises(ValueError): decode_maze(encode_maze_mask(rgb),labels)
