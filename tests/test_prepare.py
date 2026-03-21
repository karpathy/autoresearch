import types
import sys
import unittest

sys.modules.setdefault("rustbpe", types.ModuleType("rustbpe"))

from prepare import _pack_row


class PackRowTests(unittest.TestCase):
    def test_split_rows_preserve_long_document_suffixes(self):
        long_doc = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        short_doc = [90, 91]
        doc_buffer = [(long_doc, 0), (short_doc, 0)]

        row1 = _pack_row(doc_buffer, row_capacity=6, refill_buffer=lambda: None, buffer_size=0)
        row2 = _pack_row(doc_buffer, row_capacity=6, refill_buffer=lambda: None, buffer_size=0)

        self.assertEqual(row1, [90, 91, 10, 11, 12, 13])
        self.assertEqual(row2, [14, 15, 16, 17, 18, 19])
        self.assertEqual(doc_buffer, [])

    def test_split_updates_buffer_offset_instead_of_dropping_remainder(self):
        doc_buffer = [([1, 2, 3, 4, 5], 0)]

        row = _pack_row(doc_buffer, row_capacity=3, refill_buffer=lambda: None, buffer_size=0)

        self.assertEqual(row, [1, 2, 3])
        self.assertEqual(doc_buffer, [([1, 2, 3, 4, 5], 3)])


if __name__ == "__main__":
    unittest.main()
