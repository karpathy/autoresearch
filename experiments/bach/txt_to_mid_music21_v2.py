import os
import datetime
from music21 import stream, note, key, meter, midi
import re

# Base name for MIDI files (derived from input filename at runtime)

# Mappings
note_map = {
            # Enharmonic equivalents for C
            'B#': 'C', 'C': 'C', 'Dbb': 'C',
            # Enharmonic equivalents for 🅲 (C#/Db/Bx)
            'Bx': '🅲', 'C#': '🅲', 'D-': '🅲',
            # Enharmonic equivalents for D
            'Cx': 'D', 'D': 'D', 'Ebb': 'D',
            # Enharmonic equivalents for 🅳 (D#/Eb/Fbb)
            'D#': '🅳', 'E-': '🅳', 'Fbb': '🅳',
            # Enharmonic equivalents for E
            'Dx': 'E', 'E': 'E', 'Fb': 'E',
            # Enharmonic equivalents for F
            'E#': 'F', 'F': 'F', 'Gbb': 'F',
            # Enharmonic equivalents for 🅵 (F#/Gb/Ex)
            'Ex': '🅵', 'F#': '🅵', 'G-': '🅵',
            # Enharmonic equivalents for G
            'Fx': 'G', 'G': 'G', 'Abb': 'G',
            # Enharmonic equivalents for 🅶 (G#/Ab)
            'G#': '🅶', 'A-': '🅶',
            # Enharmonic equivalents for A
            'Gx': 'A', 'A': 'A', 'Bbb': 'A',
            # Enharmonic equivalents for 🅰 (A#/Bb/Cbb)
            'A#': '🅰', 'B-': '🅰', 'Cbb': '🅰',
            # Enharmonic equivalents for B
            'Ax': 'B', 'B': 'B', 'Cb': 'B'
        }

duration_map = {
    12.0: '⒏', 8.0: '8', 6.0: '⒋', 4.0: '4',
    3.0: '⒉', 2.25: '⧈',  # = 9/4
    2.0: '2', 1.75: '⊞',  # = 7/4 = 14/8
    1.5: '⒈', 1.25: '⨁',  # = 5/4 = 10/8
    1.125: '⊕',  # = 9/8
    1.0: '1',
    0.875: '⅞', 0.75: '¾', 0.66666667: '⅔',
    0.625: '⅝', 0.5625: '⅗',  # = 9/16 (approx ⅗)
    0.5: '½', 0.375: '⅜', 0.33333333: '⅓',
    0.29166667: '³',  # = 7/24 (approx 3/10)
    0.25: '¼', 0.1875: '⅕',  # = 3/16 (approx ⅕)
    0.16666667: '⅙', 0.14583333: '⅐',  # = 7/48 (approx ⅐)
    0.125: '⅛', 0.10416667: '⅑',  # = 5/48 (approx ⅑)
    0.09375: '⅒',  # = 3/32 (approx ⅒)
    0.08333333: '⑫', 0.0625: '⑯', 0.04166667: '㉔',
    0.03125: '㉜', 0.02083333: '㊽',
    0.015625: 'ⓢ', 0.0078125: 'ⓗ', 0.00390625: 'ⓣ'
}

# Reverse map: custom symbol -> canonical music21 pitch name
symbol_to_pitch = {
    'C': 'C', '🅲': 'C#', 'D': 'D', '🅳': 'E-',
    'E': 'E', 'F': 'F', '🅵': 'F#', 'G': 'G',
    '🅶': 'G#', 'A': 'A', '🅰': 'B-', 'B': 'B'
}

# Subscript digit mapping for octave numbers
subscript_digits = {
    '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
    '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9'
}

# Functions
def is_header_line(line):
    # Simple check for a header line based on your sample data
    return line.strip().endswith(':')

def is_valid_header(header):
    key_pattern = r"([A-G][#♯b♭]?)(major|minor)"
    meter_pattern = r"(\d+)/(\d+)"
    return re.search(key_pattern, header) and re.search(meter_pattern, header)

def parse_header(header):
    key_pattern = r"([A-G][#♯b♭]?)(major|minor)"
    meter_pattern = r"(\d+)/(\d+)"
    key_match = re.search(key_pattern, header)
    meter_match = re.search(meter_pattern, header)

    if key_match and meter_match:
        key_str, mode = key_match.groups()
        num, denom = meter_match.groups()
        # FIX #1: Pass key and mode as separate arguments (as in v1)
        return key.Key(key_str, mode), meter.TimeSignature(f"{num}/{denom}")
    else:
        return key.Key('C'), meter.TimeSignature("4/4")  # Default values

def extract_octave(text):
    """Extract subscript octave digits from the start of text, return (octave_str, remainder)."""
    octave = ''
    i = 0
    while i < len(text) and text[i] in subscript_digits:
        octave += subscript_digits[text[i]]
        i += 1
    return octave, text[i:]

def parse_note_duration(pair):
    # FIX #3: Handle rests (🤫) directly
    if pair.startswith('🤫'):
        rest_remainder = pair[len('🤫'):]
        # Strip any octave subscripts (shouldn't be present for rests, but just in case)
        _, duration_part = extract_octave(rest_remainder)
        duration = [k for k, v in duration_map.items() if v == duration_part]
        if duration:
            return note.Rest(quarterLength=duration[0])
        return None

    # Find the note and duration in the pair
    for note_symbol in note_map.values():
        if pair.startswith(note_symbol):
            # Extract the note part
            note_part = note_symbol
            remainder = pair[len(note_part):]

            # FIX #2: Extract subscript octave number before the duration
            octave_str, duration_part = extract_octave(remainder)

            # Find the corresponding pitch and duration
            pitch_name = symbol_to_pitch.get(note_part)
            duration = [k for k, v in duration_map.items() if v == duration_part]

            if pitch_name and duration:
                dur_value = duration[0]
                # Build full pitch string with octave if present
                if octave_str:
                    full_pitch = pitch_name + octave_str
                else:
                    full_pitch = pitch_name
                return note.Note(full_pitch, quarterLength=dur_value)
    return None

def save_midi(s, filename):
    # Append a timestamp to the filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    filename_with_timestamp = f"{filename}_{timestamp}.mid"

    # Save the stream as a MIDI file
    mf = midi.translate.streamToMidiFile(s)
    mf.open(filename_with_timestamp, 'wb')
    mf.write()
    mf.close()

def header_to_label(header):
    """Turn a header like 'bwv783 v1 Amajor 12/8:' into a filename-safe label."""
    label = header.strip().rstrip(':')
    label = label.replace('/', '-').replace(' ', '_')
    return label

# Main process
input_file = 'input.txt'
output_dir = './txt_to_mid_transcriptions'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

input_base = os.path.splitext(os.path.basename(input_file))[0]

current_stream = stream.Stream()
current_header_label = None
midi_file_count = 0

with open(input_file, 'r') as file:
    for line in file:
        if is_valid_header(line):
            if len(current_stream.elements) > 0:
                name = f"{input_base}_{current_header_label}" if current_header_label else f"{input_base}_{midi_file_count}"
                save_midi(current_stream, os.path.join(output_dir, name))
                current_stream = stream.Stream()
                midi_file_count += 1

            current_header_label = header_to_label(line)
            current_key, current_meter = parse_header(line)
            current_stream.append(current_key)
            current_stream.append(current_meter)
        else:
            for pair in line.split():
                parsed_note = parse_note_duration(pair)
                if parsed_note:
                    current_stream.append(parsed_note)

# Save the last stream if not empty
if len(current_stream.elements) > 0:
    name = f"{input_base}_{current_header_label}" if current_header_label else f"{input_base}_{midi_file_count}"
    save_midi(current_stream, os.path.join(output_dir, name))
