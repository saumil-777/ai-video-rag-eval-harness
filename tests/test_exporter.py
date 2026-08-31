import unittest
from utils.exporter import generate_txt_report, generate_pdf_report


class TestExporter(unittest.TestCase):

    def test_01_generate_txt_report(self):
        """Verify formatted text report contains expected sections."""
        txt = generate_txt_report(
            title="Sprint Planning",
            summary="Agreed on 5 tickets.",
            action_items="1. Dev A to finish PR.",
            key_decisions="1. Target release Friday.",
            open_questions="None",
            transcript="Full transcript text.",
        )

        self.assertIn("Sprint Planning", txt)
        self.assertIn("Agreed on 5 tickets.", txt)
        self.assertIn("1. Target release Friday.", txt)

    def test_02_generate_pdf_report(self):
        """Verify PDF generation returns a valid PDF byte buffer."""
        pdf_bytes = generate_pdf_report(
            title="Sprint Planning",
            summary="Agreed on 5 tickets.",
            action_items="1. Dev A to finish PR.",
            key_decisions="1. Target release Friday.",
            open_questions="None",
            transcript="Full transcript text.",
        )

        self.assertIsInstance(pdf_bytes, bytes)
        self.assertTrue(pdf_bytes.startswith(b"%PDF-"))


if __name__ == "__main__":
    unittest.main()
