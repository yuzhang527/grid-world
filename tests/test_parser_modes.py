from grid_world.prompting.parser import parse_response


def test_action_only_parser_discards_extra_map_fields():
    response = (
        '{"belief_grid":[["F","U"],["U","F"]],'
        '"thought":"hidden","action":"left"}'
    )
    result = parse_response(
        response,
        2,
        require_belief_grid=False,
    )
    assert result.error is None
    assert result.data == {"action": "LEFT"}


def test_explicit_parser_still_requires_grid():
    result = parse_response('{"action":"UP"}', 2)
    assert result.data is None
    assert "belief_grid" in result.error
