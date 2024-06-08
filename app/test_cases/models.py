class TestCase:
    def __init__(self, scenario, expected_output=None, id=None):
        self.scenario = scenario
        self.expected_output = expected_output
        self.id = id