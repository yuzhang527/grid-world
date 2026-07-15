from grid_world.env.grid import GridSpec,GridWorld
from grid_world.env.planning import shortest_path_length
def test_grid():
    spec=GridSpec("test",1,3,(0,0),(2,2),frozenset({(1,0)}))
    env=GridWorld(spec); feedback=env.feedback()
    assert [1,0] in feedback["blocked"] and [0,1] in feedback["free"]
    assert env.available_actions()==["UP"]
    assert env.step("UP")==((0,1),True)
    assert shortest_path_length(spec)==4
