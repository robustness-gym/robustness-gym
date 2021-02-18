"""Unittests for Identifiers."""
from unittest import TestCase

from robustnessgym.core.identifier import Identifier


class TestIdentifier(TestCase):
    def setUp(self):
        self.min_identifier = Identifier(_name="MyIdentifier")
        self.identifier = Identifier(
            _name="MyIdentifier",
            _index=1,
            param="a",
            param_2="b",
        )

    def test_init(self):
        # Create a simple identifier with a name
        identifier = Identifier(_name="MyIdentifier")
        self.assertEqual(str(identifier), "MyIdentifier")

        # Create an identifier with a string index
        identifier = Identifier(_name="MyIdentifier", _index="abc")
        self.assertEqual(str(identifier), "MyIdentifier-abc")

        # Create an identifier with an integer index
        identifier = Identifier(_name="MyIdentifier", _index=1)
        self.assertEqual(str(identifier), "MyIdentifier-1")

        # Create an identifier with an integer index and two parameters
        identifier = Identifier(_name="MyIdentifier", _index=1, param="a", param_2="b")
        self.assertEqual(str(identifier), "MyIdentifier-1(param=a, param_2=b)")

    def test_name(self):
        # Check the name of the identifier
        self.assertEqual(self.identifier.name, "MyIdentifier")
        self.assertEqual(self.min_identifier.name, "MyIdentifier")

    def test_index(self):
        # Check the index of the identifier
        self.assertEqual(self.identifier.index, "1")
        self.assertEqual(self.min_identifier.index, None)

    def test_parameters(self):
        # Check the parameters of the identifier
        self.assertEqual(self.identifier.parameters, {"param": "a", "param_2": "b"})
        self.assertEqual(self.min_identifier.parameters, {})

    def test_range(self):
        # Use the range function to create multiple identifiers
        identifiers = Identifier.range(3, _name="MyIdentifier", param="a", param_2="b")
        for i, identifier in enumerate(identifiers):
            self.assertEqual(identifier, f"MyIdentifier-{i + 1}(param=a, param_2=b)")

    def test_eq(self):
        # Two identifiers created with the same arguments should be equal
        identifier = Identifier(_name="MyIdentifier", _index=1, param="a", param_2="b")
        self.assertEqual(self.identifier, identifier)
        self.assertNotEqual(self.min_identifier, identifier)

        # But not two identifiers created with different arguments
        identifier = Identifier(_name="MyIdentifier", _index=2, param="a", param_2="b")
        self.assertNotEqual(self.identifier, identifier)
        self.assertNotEqual(self.min_identifier, identifier)

    def test_dumps(self):
        # Dump the identifier to a json
        self.assertEqual(
            self.identifier.dumps(),
            '{"_name": "MyIdentifier", "_index": "1", "_parameters": {"param": "a", '
            '"param_2": "b"}}',
        )

    def test_loads(self):
        # Dump the identifier to a json string and load it back
        s = self.identifier.dumps()
        identifier = Identifier.loads(s)
        self.assertEqual(identifier, self.identifier)

    def test_add_parameter(self):
        self.identifier.add_parameter("extra", "value")
        self.assertEqual(
            self.identifier.parameters, {"param": "a", "param_2": "b", "extra": "value"}
        )

    def test_without(self):
        identifier = self.identifier.without("param")
        self.assertTrue("param" not in identifier.parameters)

    def test_parse(self):
        identifier = Identifier.parse(
            "Spacy(lang=en_core_web_sm, neuralcoref=False, columns=['text'])"
        )
        self.assertEqual(
            str(identifier),
            "Spacy(lang=en_core_web_sm, neuralcoref=False, columns=['text'])",
        )
        self.assertEqual(
            set(identifier.parameters.keys()), {"lang", "neuralcoref", "columns"}
        )
