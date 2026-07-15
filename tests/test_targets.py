from grid_world.env.belief import initial_belief,update_belief
from grid_world.env.grid import GridSpec
def test_belief():
    spec=GridSpec("test",1,3,(0,0),(2,2),frozenset({(1,0)}))
    belief=update_belief(initial_belief(spec),
        {"position":[0,0],"free":[[0,1]],"blocked":[[1,0]],"wall":[]})
    assert belief[(1,0)]=="O" and belief[(0,1)]=="F" and belief[(1,1)]=="U"
