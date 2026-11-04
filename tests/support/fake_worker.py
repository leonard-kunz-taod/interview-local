"""Stand-in for the speech worker processes, with canned stage results."""

import json

WORDS = [
    {"start": 0, "end": 1, "text": "Hallo", "timing": "aligned", "probability": None}
]
TURNS = [{"start": 0, "end": 1, "speaker": "s0"}]
DIARIZATION = {"exclusive": TURNS, "regular": TURNS}
OUTPUTS = {"asr": {"segments": []}, "align": WORDS, "diarize": DIARIZATION}
SPOKEN_SEGMENT = {
    "start": 0,
    "end": 1,
    "text": " Grüße",
    "words": [
        {"start": 0, "end": 1, "text": "Grüße", "timing": "whisper", "probability": 0.9}
    ],
}
SPOKEN_OUTPUTS = dict(OUTPUTS, asr={"segments": [SPOKEN_SEGMENT]})
STAGES_READING_THEIR_SOURCE = {"decode", "asr", "align", "diarize"}


class FakeWorker:
    """Writes canned stage results instead of running speech models.

    Like the real worker, a stage fails when its source is missing, alignment
    reads the transcript saved as asr.json next to its audio, and completion is
    reported through status.
    """

    def __init__(self, outputs):
        self.outputs = outputs
        self.stages = []
        self.workspaces = []
        self.tokens = set()
        self.offline = set()

    def __call__(self, stage, source, target, status, token, offline):
        self.stages.append(stage)
        self.workspaces.append(target.parent)
        self.tokens.add(token)
        self.offline.add(offline)
        if stage in STAGES_READING_THEIR_SOURCE:
            source.read_bytes()
        if stage == "align":
            source.with_name("asr.json").read_text(encoding="utf-8")
        if stage == "decode":
            target.touch()
        elif stage in self.outputs:
            result = json.dumps(self.outputs[stage], ensure_ascii=False)
            target.write_text(result, encoding="utf-8")
        status(f"Stage {stage} completed in 0.0 s")
