"""Orpheus token protocol, independent of the inference and UI libraries."""

AUDIO_START = 128257
AUDIO_END = 128258
END_OF_TURN = 128262
CODE_OFFSET = 128266
CODEBOOK_SIZE = 4096
FRAME_SIZE = 7


def build_prompt(model, text, voice):
    # Use the GGUF's own tokenizer; an unrelated HF tokenizer can change IDs.
    text_ids = model.tokenize(f"{voice}: {text}".encode("utf-8"), add_bos=True, special=False)
    return [128259, *text_ids, 128009, 128260, 128261, AUDIO_START]


def parse_output(generated_ids):
    """Validate complete SNAC frames, dropping only an unfinished final frame."""
    tokens = []
    for token in generated_ids:
        if token in (AUDIO_END, END_OF_TURN):
            break
        if token == AUDIO_START and not tokens:
            continue
        position = len(tokens) % FRAME_SIZE
        code = token - CODE_OFFSET - position * CODEBOOK_SIZE
        if not 0 <= code < CODEBOOK_SIZE:
            raise ValueError(f"Invalid audio token at position {len(tokens)}: {token}")
        tokens.append(code)
    tokens = tokens[:len(tokens) // FRAME_SIZE * FRAME_SIZE]
    if not tokens:
        raise ValueError("The model did not generate a complete audio frame. Try again or increase Max New Tokens.")
    return tokens


def split_codes(codes):
    """Map each seven-code frame onto SNAC's three temporal resolutions."""
    if not codes or len(codes) % FRAME_SIZE:
        raise ValueError("Audio codes must contain complete seven-code frames.")
    if any(not 0 <= code < CODEBOOK_SIZE for code in codes):
        raise ValueError("Audio code is outside the SNAC codebook.")
    return [
        codes[::7],
        [codes[i + j] for i in range(0, len(codes), 7) for j in (1, 4)],
        [codes[i + j] for i in range(0, len(codes), 7) for j in (2, 3, 5, 6)],
    ]
