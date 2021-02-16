import re
import unittest
import astar
import subprocess
import timeout_decorator
import time
import sys

#
#  You will need to have the timeout_decorator available for your python install
#  for these tests to work as written...
#
#  pip install timeout_decorator
#
#
class AStarTestCase(unittest.TestCase):


    def test_AStar_onDefaultGridProblem(self):
        grid = [[1,  10, 10, 10, 10],
                [1,  1,  10, 10, 10],
                [10, 1,  1,  1,  1],
                [10, 10, 1,  10, 1],
                [10, 10, 1,  10, 1]]
        gp = astar.GridProblem(grid)
        a = astar.AStar(gp)
        pth = a.search((0,0))
        self.assertEqual(8, pth.g, "Expected a path cost of 8 on the default grid")
        states = a.list_of_states(pth)
        self.assertEqual(9, len(states), "Expected path to contain 9 elements")
        self.assertTrue((0,0) in a.reached, "Expected (0,0) on reached dict")
        self.assertTrue((0,1) in a.reached, "Expected (0,1) on reached dict")
        self.assertFalse((0,2) in a.reached, "Didn't expect (row 0, col 2) in reached")
        # so, this isn't guaranteed to happen unless h(x) = 0
        # self.assertTrue((4,2) in a.reached, "Expected (4,2) on reached dict")
        print("Your reached dict has: ", a.reached)

    def test_AStar_onDefaultGridProblemNoHeuristic(self):
        grid = [[1,  10, 10, 10, 10],
                [1,  1,  10, 10, 10],
                [10, 1,  1,  1,  1],
                [10, 10, 1,  10, 1],
                [10, 10, 1,  10, 1]]
        gp = astar.GridProblem(grid, hfn=lambda x: 0)
        a = astar.AStar(gp)
        pth = a.search((0,0))
        self.assertEqual(8, pth.g, "Expected a path cost of 8 on the default grid")
        states = a.list_of_states(pth)
        self.assertEqual(9, len(states), "Expected path to contain 9 elements")
        self.assertTrue((0,0) in a.reached, "Expected (0,0) on reached dict")
        self.assertTrue((4,2) in a.reached, "Expected (4,2) on reached dict")
        print("Your reached dict has: ", a.reached)


    def test_AStar_Greedyness(self):
        grid = [[1,  10, 10, 1,  1,  1],
                [1,  1,  10, 1, 10,  1],
                [10, 1,  1,  1, 30,  1],
                [10, 10, 10, 1, 10,  1],
                [10, 10, 10, 2, 1,  1]]
        gp = astar.GridProblem(grid, hfn=lambda x: 0, goaltest=lambda x: x == (4,5))
        a = astar.AStar(gp)
        pth = a.search((0,0))
        print("PATH IS", a.list_of_states(pth), file=sys.stderr)
        self.assertEqual(10, pth.g, "Expected a path cost of 10 on this grid")
        states = a.list_of_states(pth)
        self.assertEqual(10, len(states), "Expected path to contain 10 elements")
        self.assertTrue((0,0) in a.reached, "Expected (0,0) on reached dict")
        self.assertTrue((4,3) in a.reached, "Expected (4,3) on reached dict")
        print("Your reached dict has: ", a.reached)


    def test_AStar_on8x12png(self):
        ip = astar.ImageProblem('12x8.png', lambda x: x == (11,7), lambda s: abs(s[0]-11)+abs(s[1]-7))
        a = astar.AStar(ip)
        pth = a.search((0,0))
        self.assertEqual(15, pth.g, "Expected a path cost of 15 from UL-corner to LR the '12x8.png'")
        states = a.list_of_states(pth)
        self.assertFalse((1,0) in states, "Didn't expect (1,0) on the path... path should have taken the diagonal")

    def test_AStar_on8x12trickypng(self):
        #ip = astar.ImageProblem('12x8tricky.png', lambda x: x == (11,7), lambda s: abs(s[0]-11)+abs(s[1]-7))

        def h(s):
            wleft  = abs(s[0] - 11)
            hleft = abs(s[1] - 7)
            diag = min(wleft, hleft)
            remainder = max(wleft, hleft) - diag
            return diag + remainder

        ip = astar.ImageProblem('12x8tricky.png', lambda x: x == (11,7), lambda x: 0)
        a = astar.AStar(ip)
        pth = a.search((0,0))
        self.assertEqual("18.41", "%.2f"%pth.g, "Expected a path cost of 18.41 from UL-corner to LR the '12x8tricky.png'")
        states = a.list_of_states(pth)
        self.assertFalse((1,0) in states, "Didn't expect (1,0) on the path... path should have taken the diagonal")

    def test_AStar_on8x12trickypngWithNonAdmissibleHeuristic(self):
        #ip = astar.ImageProblem('12x8tricky.png', lambda x: x == (11,7), lambda s: abs(s[0]-11)+abs(s[1]-7))

        def h(s):
            wleft  = abs(s[0] - 11)
            hleft = abs(s[1] - 7)
            return wleft+hleft

        ip = astar.ImageProblem('12x8tricky.png', lambda x: x == (11,7), h)
        a = astar.AStar(ip)
        pth = a.search((0,0))
        self.assertEqual("19.41", "%.2f"%pth.g, "Expected to find a suboptimal path with my weird heuristic '12x8tricky.png'")
        states = a.list_of_states(pth)
        self.assertFalse((1,0) in states, "Didn't expect (1,0) on the path... path should have taken the diagonal")


    def test_AStar_asScriptOnGrid(self):
        out = subprocess.getoutput("python astar.py")
        print(out)
        lines = out.splitlines()

        # this can be a bit tricky to test since minor variations may have
        # different use of the reached dict without affecting optimality and completeness.
        #
        # expected = "10 states in the closed list/reached dict"
        # self.assertEqual(expected, lines[-3], f"Expected 3rd to last line to read '{expected}'")

        expected = "9 states on path to goal"
        self.assertEqual(expected, lines[-2], f"Expected 2nd to last line to read '{expected}'")
        expected = "cost is: 8.00"
        self.assertEqual(expected, lines[-1], f"Expected last line to read '{expected}'")
        stateslist = ['(4, 4)']
        self.assertTrue(len(lines) >= 13, "hmm. your ouput doesnt have enough lines for all the states...did you remove output?")
        path = "\n".join(lines[-14:-4])
        self.assertEqual("(0, 0)\n(1, 0)\n(1, 1)\n(2, 1)\n(2, 2)\n(2, 3)\n(2, 4)\n(3, 4)\n(4, 4)", path)

    def test_AStar_asScriptOn12x8png(self):
        out = subprocess.getoutput("python3 astar.py 12x8.png")
        lines = out.splitlines()
        expected = "16 states on path to goal"
        self.assertEqual(expected, lines[-2], f"Expected 2nd to last line to read '{expected}'")
        expected = "cost is: 15.00"
        self.assertEqual(expected, lines[-1], f"Expected last line to read '{expected}'")


    def testAStarLongWithLongTimer(self):
        def h(s):
            wleft  = abs(s[0] - 79)
            hleft = abs(s[1] - 79)
            diag = min(wleft, hleft)
            remainder = max(wleft, hleft) - diag
            return diag + remainder

        timeout = 30
        msg = "Hint: AStar Exceeded %d seconds, this you'll need to optimize a bit for this one"%timeout
        start = time.time()
        @timeout_decorator.timeout(timeout, exception_message=msg)
        def innertest():
            ip = astar.ImageProblem('80x80-0.png', lambda x: x == (79,79), h)
            a = astar.AStar(ip)
            pth = a.search((35,35))
            self.assertTrue(pth)
            self.assertTrue(pth.f > 204.02 and pth.f < 204.024) 

        innertest()
        stop = time.time()
        print("Ran the 'long' test in %.1f seconds"%(stop-start))

    def testAStarLongWithOptimizedTimer(self):
        def h(s):
            wleft  = abs(s[0] - 79)
            hleft = abs(s[1] - 79)
            diag = min(wleft, hleft)
            remainder = max(wleft, hleft) - diag
            return diag + remainder

        timeout = 10
        msg = "Hint: AStar Exceeded %d seconds, you'll need to optimize a bit for this one"%timeout
        @timeout_decorator.timeout(timeout, exception_message=msg)
        def innertest():
            ip = astar.ImageProblem('80x80-0.png', lambda x: x == (79,79), h)
            a = astar.AStar(ip)
            pth = a.search((35,35))
            self.assertTrue(pth)
            self.assertTrue(pth.f > 204.02 and pth.f < 204.024) 

        innertest()

    def testAStarLongWithNoHeuristicOptimizedTimer(self):
        def h(s):
            return 0

        timeout = 10
        msg = "Hint: AStar Exceeded %d seconds, you'll need to optimize a bit for this one"%timeout
        @timeout_decorator.timeout(timeout, exception_message=msg)
        def innertest():
            ip = astar.ImageProblem('80x80-0.png', lambda x: x == (79,79), h)
            a = astar.AStar(ip)
            pth = a.search((35,35))
            self.assertTrue(pth)

        innertest()
