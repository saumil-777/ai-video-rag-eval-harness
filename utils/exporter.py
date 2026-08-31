import io
from fpdf import FPDF


def generate_txt_report(
    title: str,
    summary: str,
    action_items: str,
    key_decisions: str,
    open_questions: str,
    transcript: str,
) -> str:
    """Generate a clean, formatted plain text report of the meeting analysis."""
    divider = "=" * 80 + "\n"
    sub_divider = "-" * 80 + "\n"

    report = [
        divider,
        "🎬 AI VIDEO ASSISTANT — MEETING REPORT\n",
        divider,
        f"\n📌 TITLE: {title or 'Untitled Meeting'}\n\n",
        sub_divider,
        "📋 SUMMARY:\n",
        sub_divider,
        f"{summary or 'No summary available.'}\n\n",
        sub_divider,
        "✅ ACTION ITEMS:\n",
        sub_divider,
        f"{action_items or 'None'}\n\n",
        sub_divider,
        "🔑 KEY DECISIONS:\n",
        sub_divider,
        f"{key_decisions or 'None'}\n\n",
        sub_divider,
        "❓ OPEN QUESTIONS:\n",
        sub_divider,
        f"{open_questions or 'None'}\n\n",
        divider,
        "📝 FULL TRANSCRIPT:\n",
        divider,
        f"{transcript or 'No transcript available.'}\n",
    ]
    return "".join(report)


class PDFReport(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(120, 120, 150)
        self.cell(0, 8, "AI Video Assistant -- Meeting Intelligence Report", border=False, align="R")
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")


def clean_text_for_pdf(text: str) -> str:
    """Sanitize text characters for standard PDF Helvetica font encoding."""
    if not text:
        return ""
    replacements = {
        "\u2013": "-",
        "\u2014": "--",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2022": "*",
        "\u2026": "...",
    }
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def generate_pdf_report(
    title: str,
    summary: str,
    action_items: str,
    key_decisions: str,
    open_questions: str,
    transcript: str,
) -> bytes:
    """Generate a clean, professional PDF document as a byte buffer using fpdf2."""
    pdf = PDFReport()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Document Header
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(124, 58, 237)  # Accent Purple
    pdf.cell(0, 12, "AI Video Meeting Report", new_x="LMARGIN", new_y="NEXT", align="L")
    pdf.ln(2)

    # Session Title
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(30, 30, 40)
    safe_title = clean_text_for_pdf(title or "Untitled Meeting")
    pdf.multi_cell(0, 8, f"Title: {safe_title}")
    pdf.ln(5)

    def add_section(section_title: str, content: str, r=124, g=58, b=237):
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(r, g, b)
        pdf.cell(0, 8, section_title, new_x="LMARGIN", new_y="NEXT")
        pdf.set_draw_color(200, 200, 220)
        pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x() + 180, pdf.get_y())
        pdf.ln(3)

        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(50, 50, 60)
        safe_content = clean_text_for_pdf(content or "None")
        pdf.multi_cell(0, 6, safe_content)
        pdf.ln(6)

    # Add Report Sections
    add_section("Meeting Summary", summary)
    add_section("Action Items", action_items, r=16, g=185, b=129)
    add_section("Key Decisions", key_decisions, r=6, g=182, b=212)
    add_section("Open Questions", open_questions, r=245, g=158, b=11)
    add_section("Full Meeting Transcript", transcript, r=100, g=100, b=120)

    return bytes(pdf.output())
