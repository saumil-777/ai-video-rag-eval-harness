import unittest
import uuid
from dotenv import load_dotenv

load_dotenv()

from core.vector_store import (
    sanitize_collection_name,
    build_vector_store,
    get_retriever,
)


class TestVectorStore(unittest.TestCase):

    def test_01_sanitize_collection_name(self):
        """Verify collection name formatting rules."""
        s1 = sanitize_collection_name("Meeting #1 Video")
        self.assertTrue(s1.isalnum() or '_' in s1 or '-' in s1)
        self.assertTrue(len(s1) >= 3 and len(s1) <= 63)

    def test_02_build_and_retrieve(self):
        """Verify building a vector store and retrieving relevant documents."""
        session_id = f"session_vt_{uuid.uuid4().hex[:8]}"
        transcript = "The marketing team launched campaign X in August."

        vs = build_vector_store(transcript, collection_name=session_id)
        retriever = get_retriever(vs, k=2)

        docs = retriever.invoke("When was campaign X launched?")
        self.assertTrue(len(docs) > 0)
        self.assertIn("August", docs[0].page_content)


if __name__ == "__main__":
    unittest.main()
