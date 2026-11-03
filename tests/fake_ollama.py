"""In-memory stand-in for the local Ollama editor and reviewer."""


class FakeClient:
    def __init__(
        self,
        invalid_edits=0,
        wording=None,
        speaker="A",
        split=False,
        review_id_offset=0,
        paragraph_fields=None,
        review_fields=None,
    ):
        self.invalid_edits = invalid_edits
        self.wording = wording
        self.speaker = speaker
        self.split = split
        self.review_id_offset = review_id_offset
        self.paragraph_fields = paragraph_fields or {}
        self.review_fields = review_fields or {}
        self.unloaded = False
        self.prepared = False

    def ready(self):
        return {"model": "fake-local-test"}

    def prepare(self, status=print):
        self.prepared = True
        return self.ready()

    def unload(self):
        self.unloaded = True

    def chat(self, instruction, data, schema):
        if "editable_words" in data:
            return self._edit(data["editable_words"])
        return {
            "reviews": [
                {
                    "paragraph_id": p["paragraph_id"] + self.review_id_offset,
                    "uncertain": False,
                    "reasons": [],
                    **self.review_fields,
                }
                for p in data["paragraphs"]
            ]
        }

    def _edit(self, words):
        invalid = self.invalid_edits > 0
        self.invalid_edits -= 1
        middle = len(words) // 2
        groups = [words[:middle], words[middle:]] if self.split else [words]
        paragraphs = [self._paragraph(group) for group in groups]
        if invalid:
            paragraphs[0]["first_word"] += 1
        return {"paragraphs": paragraphs}

    def _paragraph(self, words):
        return {
            "first_word": words[0]["id"],
            "last_word": words[-1]["id"],
            "speaker": self.speaker,
            "text": self.wording or " ".join(w["t"] for w in words),
            "uncertain": False,
            "reason": "Grouped into a readable paragraph.",
            **self.paragraph_fields,
        }
