from grid_world.prompting.parser import parse_response
def test_parse():
    text='x {"belief_grid":[["U","F"],["F","U"]],"action":"right"} y'
    result=parse_response(text,2)
    assert result.error is None and result.data["action"]=="RIGHT"
def test_bad():
    assert parse_response('{"belief_grid":[["X"]],"action":"UP"}',1).data is None
