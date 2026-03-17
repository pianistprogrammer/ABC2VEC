"""ABC notation transposition utility."""

import re
from typing import Dict, Tuple


class ABCTransposer:
    """
    Transpose ABC notation tunes by semitones.

    Critical for the Transposition Invariance pre-training objective.
    The same tune in different keys should produce similar embeddings.

    Handles:
        - Notes (A-G in upper/lower case)
        - Accidentals (^ for sharp, _ for flat, = for natural)
        - Key signatures

    Example:
        >>> transposer = ABCTransposer()
        >>> body = "D2 EF | G2 AB |"
        >>> transposed = transposer.transpose_abc_body(body, semitones=5)
        >>> print(transposed)  # Transposed to G major
        G2 A^A | C2 DE |
    """

    # Chromatic scale (using sharps)
    CHROMATIC_SHARP = [
        "C", "C#", "D", "D#", "E", "F",
        "F#", "G", "G#", "A", "A#", "B"
    ]

    CHROMATIC_FLAT = [
        "C", "Db", "D", "Eb", "E", "F",
        "Gb", "G", "Ab", "A", "Bb", "B"
    ]

    # ABC note names to chromatic index (C=0, C#=1, ..., B=11)
    NOTE_TO_SEMITONE: Dict[str, int] = {
        "C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11
    }

    # Key roots to semitone offset
    KEY_TO_SEMITONE: Dict[str, int] = {
        "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
        "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
        "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
    }

    @classmethod
    def transpose_note(
        cls,
        note_char: str,
        accidental: str,
        semitones: int
    ) -> Tuple[str, str]:
        """
        Transpose a single note by semitones.

        Args:
            note_char: Single character note (A-G, case-sensitive)
            accidental: Accidental symbol (^, _, or empty string)
            semitones: Number of semitones to transpose (positive or negative)

        Returns:
            Tuple of (new_note_char, new_accidental)
        """
        base = note_char.upper()
        if base not in cls.NOTE_TO_SEMITONE:
            return note_char, accidental

        # Compute current semitone including accidental
        current = cls.NOTE_TO_SEMITONE[base]
        if accidental == "^":  # sharp
            current += 1
        elif accidental == "_":  # flat
            current -= 1

        # Transpose and wrap around octave
        new_semitone = (current + semitones) % 12

        # Map back to note + accidental
        sharp_name = cls.CHROMATIC_SHARP[new_semitone]

        if len(sharp_name) == 1:
            new_note = sharp_name
            new_acc = ""
        elif "#" in sharp_name:
            new_note = sharp_name[0]
            new_acc = "^"
        else:
            new_note = sharp_name[0]
            new_acc = "_"

        # Preserve case (lowercase = upper octave in ABC)
        if note_char.islower():
            new_note = new_note.lower()

        return new_note, new_acc

    @classmethod
    def transpose_key(cls, key_str: str, semitones: int) -> str:
        """
        Transpose a key signature string.

        Examples:
            - 'D' + 5 → 'G'
            - 'Ador' + 5 → 'Ddor'
            - 'Em' + 7 → 'Bm'

        Args:
            key_str: Key signature string (e.g., 'D', 'Gmix', 'Ador')
            semitones: Number of semitones to transpose

        Returns:
            Transposed key signature string
        """
        # Parse root note and mode suffix
        match = re.match(r"^([A-G][#b]?)(.*)$", key_str.strip())
        if not match:
            return key_str

        root = match.group(1)
        mode = match.group(2)

        # Transpose root note
        if root in cls.KEY_TO_SEMITONE:
            new_semitone = (cls.KEY_TO_SEMITONE[root] + semitones) % 12
            new_root = cls.CHROMATIC_SHARP[new_semitone]
            # Convert sharp notation to ABC notation (# instead of ♯)
            if "#" in new_root:
                new_root = new_root.replace("#", "#")
            return new_root + mode

        return key_str

    @classmethod
    def transpose_abc_body(cls, abc_body: str, semitones: int) -> str:
        """
        Transpose all notes in an ABC body by semitones.

        Handles ABC-specific notation:
            - ^C (sharp)
            - _B (flat)
            - =F (natural/explicit)

        Args:
            abc_body: ABC notation body string
            semitones: Number of semitones to transpose (can be negative)

        Returns:
            Transposed ABC body string
        """
        if semitones == 0:
            return abc_body

        result = []
        i = 0

        while i < len(abc_body):
            ch = abc_body[i]

            # Check for accidental prefix (^, _, =)
            if (
                ch in "^_="
                and i + 1 < len(abc_body)
                and abc_body[i + 1].upper() in "ABCDEFG"
            ):
                accidental = ch if ch != "=" else ""
                note = abc_body[i + 1]
                new_note, new_acc = cls.transpose_note(
                    note, accidental, semitones
                )
                if new_acc:
                    result.append(new_acc)
                result.append(new_note)
                i += 2

            elif ch.upper() in "ABCDEFG":
                # Regular note without accidental
                new_note, new_acc = cls.transpose_note(ch, "", semitones)
                if new_acc:
                    result.append(new_acc)
                result.append(new_note)
                i += 1

            else:
                # Non-note character (rhythm, bar lines, etc.)
                result.append(ch)
                i += 1

        return "".join(result)

    @classmethod
    def transpose_abc_full(
        cls,
        abc_text: str,
        semitones: int
    ) -> str:
        """
        Transpose a full ABC tune including headers.

        Transposes both the K: (key) header field and all notes in the body.

        Args:
            abc_text: Complete ABC notation text with headers
            semitones: Number of semitones to transpose

        Returns:
            Fully transposed ABC text
        """
        if semitones == 0:
            return abc_text

        lines = abc_text.split("\n")
        result_lines = []

        for line in lines:
            stripped = line.strip()

            # Transpose key header
            if stripped.startswith("K:"):
                key_value = stripped[2:].strip()
                new_key = cls.transpose_key(key_value, semitones)
                result_lines.append(f"K:{new_key}")

            # Transpose body lines (lines with notes and bar lines)
            elif "|" in stripped or any(
                ch.upper() in "ABCDEFG" for ch in stripped
            ):
                transposed_line = cls.transpose_abc_body(stripped, semitones)
                result_lines.append(transposed_line)

            else:
                # Header or empty lines pass through
                result_lines.append(stripped)

        return "\n".join(result_lines)
