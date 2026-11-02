#!/usr/bin/env python3
"""Experimental editor for Vanilla Angband (VNLA) save files.

Supports five-stat current saves and six-stat legacy saves, plus the
block-based save format and the version-1 `player` and `misc` blocks described
by the upstream Angband source. Unknown formats are not guessed.

Safety rules:
- The original is never overwritten unless --in-place is explicitly used.
- Existing output files are never overwritten unless --overwrite is explicit.
- Any pre-existing block-checksum mismatch prevents saving.
- Output is written to a temporary file, reparsed, and byte-verified before
  it is atomically installed. A non-colliding backup is made before replacing
  an existing file.

This is still an experimental editor. Close Angband and test on a copy.
"""

from __future__ import annotations

import argparse
import os
import shutil
import stat
import struct
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

MAGIC = b"SaveVNLA"
BLOCK_HEADER_SIZE = 28

FORMATS = {
    "u8": "<B",
    "i8": "<b",
    "u16": "<H",
    "i16": "<h",
    "u32": "<I",
    "i32": "<i",
    "f32": "<f",
    "f64": "<d",
}


class SaveFormatError(ValueError):
    """Raised when the save file is not a supported/parseable format."""


@dataclass
class Block:
    name: str
    version: int
    size: int
    checksum: int
    header_offset: int
    data_offset: int
    data: bytearray
    padding: bytes

    @property
    def checksum_ok(self) -> bool:
        return (sum(self.data) & 0xFFFFFFFF) == self.checksum


@dataclass
class Field:
    name: str
    block: str
    offset: int
    kind: str
    length: int = 0  # used for fixed-length, null-terminated strings
    description: str = ""

    @property
    def byte_size(self) -> int:
        if self.kind == "string":
            return self.length
        return struct.calcsize(FORMATS[self.kind])


class Cursor:
    """Small bounds-checked reader used to locate serialized player fields."""

    def __init__(self, data: bytearray, block_name: str):
        self.data = data
        self.pos = 0
        self.block_name = block_name

    def require(self, amount: int) -> None:
        if amount < 0 or self.pos + amount > len(self.data):
            raise SaveFormatError(
                f"'{self.block_name}' block ended unexpectedly at offset {self.pos}"
            )

    def take(self, amount: int) -> int:
        self.require(amount)
        start = self.pos
        self.pos += amount
        return start

    def number(self, kind: str) -> int:
        if kind not in FORMATS:
            raise ValueError(f"Unsupported field type: {kind}")
        fmt = FORMATS[kind]
        self.require(struct.calcsize(fmt))
        value = struct.unpack_from(fmt, self.data, self.pos)[0]
        self.pos += struct.calcsize(fmt)
        return value

    def cstring(self) -> tuple[int, int]:
        start = self.pos
        try:
            end = self.data.index(0, start)
        except ValueError as exc:
            raise SaveFormatError(
                f"Unterminated string in '{self.block_name}' block at {start}"
            ) from exc
        self.pos = end + 1
        return start, self.pos - start


class AngbandSaveEditor:
    """Read, inspect, and modify supported Angband save-file fields."""

    def __init__(self, filename: os.PathLike | str):
        self.path = Path(filename).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.original = self.path.read_bytes()
        self.blocks: list[Block] = []
        self.fields: dict[str, Field] = {}
        self.warnings: list[str] = []
        self._parse_file()
        # Freeze input integrity status before any in-memory field edits.
        # Block.checksum_ok reflects the current mutable payload after edits;
        # this list records whether the file was already damaged on disk.
        self.input_checksum_mismatches = [
            b.name for b in self.blocks if not b.checksum_ok
        ]
        # Keep block inspection usable even when a future/variant player block
        # has fields this editor does not yet understand.
        try:
            self._discover_known_fields()
        except (SaveFormatError, KeyError, IndexError, struct.error) as exc:
            # Never leave partially discovered offsets exposed.  A partially
            # decoded layout can point valid-looking names at the wrong bytes.
            self.fields.clear()
            self.warnings.append(
                f"Known-field discovery stopped; named edits disabled: {exc}"
            )

    # ---------- Save-file/block parsing ----------

    def _parse_file(self) -> None:
        raw = self.original
        if len(raw) < 8 or raw[:8] != MAGIC:
            variant = raw[4:8] if len(raw) >= 8 else b""
            raise SaveFormatError(
                "This program currently supports Vanilla Angband save files "
                f"with header SaveVNLA; found {raw[:8]!r} (variant {variant!r})."
            )

        pos = 8
        while pos < len(raw):
            if len(raw) - pos < BLOCK_HEADER_SIZE:
                raise SaveFormatError(f"Truncated block header at file offset {pos}")

            header = raw[pos : pos + BLOCK_HEADER_SIZE]
            name_raw = header[:16]
            if name_raw[15] != 0:
                raise SaveFormatError(
                    f"Block name at file offset {pos} is not null-terminated"
                )
            name_bytes = name_raw.split(b"\0", 1)[0]
            try:
                name = name_bytes.decode("ascii")
            except UnicodeDecodeError as exc:
                raise SaveFormatError(
                    f"Non-ASCII block name at offset {pos}"
                ) from exc

            version, size, checksum = struct.unpack_from("<III", header, 16)
            data_start = pos + BLOCK_HEADER_SIZE
            data_end = data_start + size
            padded_size = (size + 3) & ~3
            padded_end = data_start + padded_size
            if data_end > len(raw) or padded_end > len(raw):
                raise SaveFormatError(
                    f"Block '{name}' at offset {pos} claims {size} data bytes, "
                    "which extends beyond the file"
                )

            data = bytearray(raw[data_start:data_end])
            padding = raw[data_end:padded_end]
            block = Block(
                name=name,
                version=version,
                size=size,
                checksum=checksum,
                header_offset=pos,
                data_offset=data_start,
                data=data,
                padding=padding,
            )
            self.blocks.append(block)
            if not block.checksum_ok:
                self.warnings.append(
                    f"Checksum differs for block '{name}' "
                    f"(stored 0x{checksum:08x}, "
                    f"calculated 0x{sum(data) & 0xFFFFFFFF:08x}); "
                    "Angband 4.2.6 does not validate this word when loading."
                )
            pos = padded_end

        if not self.blocks:
            raise SaveFormatError("No data blocks found in this save file")

    def _block(self, name: str) -> Block:
        matches = [
            b for b in self.blocks if b.name.casefold() == name.casefold()
        ]
        if not matches:
            raise KeyError(f"No block named {name!r}; use list_blocks()")
        if len(matches) > 1:
            raise SaveFormatError(
                f"Multiple blocks named {name!r}; refusing ambiguity"
            )
        return matches[0]

    # ---------- Field registration/discovery ----------

    def _add_field(
        self,
        name: str,
        block: str,
        offset: int,
        kind: str,
        *,
        length: int = 0,
        description: str = "",
        aliases: tuple[str, ...] = (),
    ) -> None:
        b = self._block(block)
        field = Field(
            name=name,
            block=block,
            offset=offset,
            kind=kind,
            length=length,
            description=description,
        )
        if offset < 0 or offset + field.byte_size > len(b.data):
            raise SaveFormatError(
                f"Field {name!r} points outside '{block}' block "
                f"(offset={offset}, size={field.byte_size}, block={len(b.data)})"
            )
        keys = (name,) + aliases
        for key in keys:
            self.fields[key.casefold()] = field

    def _discover_known_fields(self) -> None:
        """Decode fields whose layout is known from the current upstream source."""
        player = self._block("player")
        if player.version != 1:
            self.warnings.append(
                f"Player block version is {player.version}; known field discovery "
                "currently supports version 1 only."
            )
            return

        c = Cursor(player.data, "player")

        # Six leading C strings.
        string_specs = [
            ("player.name", "Character name", ("name", "character_name")),
            ("player.death_cause", "Death cause", ("death_cause",)),
            ("player.history", "Character history", ("history",)),
            ("player.race", "Race name", ("race",)),
            ("player.shape", "Shape name", ("shape",)),
            ("player.class", "Class name", ("class",)),
        ]
        for field_name, description, aliases in string_specs:
            off, length = c.cstring()
            self._add_field(
                field_name,
                "player",
                off,
                "string",
                length=length,
                description=description,
                aliases=aliases,
            )

        for field_name, desc, aliases in [
            ("player.name_suffix", "Name suffix option", ()),
            ("player.hitdie", "Character hit die", ("hitdie",)),
            ("player.expfact", "Experience factor", ("exp_factor",)),
        ]:
            off = c.take(1)
            self._add_field(
                field_name, "player", off, "u8",
                description=desc, aliases=aliases
            )

        for field_name, desc in [
            ("player.age", "Character age"),
            ("player.height", "Character height"),
            ("player.weight", "Character weight"),
        ]:
            off = c.take(2)
            self._add_field(
                field_name, "player", off, "i16", description=desc
            )

        # STAT_MAX is stored in the save itself. Current Vanilla Angband uses
        # five stats (STR, INT, WIS, DEX, CON); older saves may contain six,
        # including CHR. Read the count instead of assuming one layout.
        stat_count = c.number("u8")
        stat_names = [
            "strength",
            "intelligence",
            "wisdom",
            "dexterity",
            "constitution",
            "charisma",
        ]
        short_names = ["str", "int", "wis", "dex", "con", "chr"]
        if stat_count not in (5, 6):
            raise SaveFormatError(
                f"Unsupported stat count {stat_count} in player block; "
                "supported layouts have 5 (current) or 6 (legacy) stats"
            )
        stat_names = stat_names[:stat_count]
        short_names = short_names[:stat_count]
        for group, group_alias in [
            ("max", "maximum"),
            ("current", "cur"),
            ("map", "mapping"),
            ("birth", "birth"),
        ]:
            for stat, short in zip(stat_names, short_names):
                off = c.take(2)
                canonical = f"stat_{group}.{short}"
                aliases = (
                    f"stat_{group}.{stat}",
                    f"stat_{group_alias}.{short}",
                    f"stat_{group_alias}.{stat}",
                    f"{group_alias}_{stat}",
                )
                self._add_field(
                    canonical,
                    "player",
                    off,
                    "i16",
                    description=f"{group.title()} {stat}",
                    aliases=aliases,
                )

        # Birth height/weight, reserved u16, birth gold.
        c.take(2)
        c.take(2)
        c.take(2)
        off = c.take(4)
        # The writer serializes four bytes with wr_u32b(), while rd_player()
        # reads those bytes with rd_s32b() into int32_t. Model the loader's
        # signed interpretation so the editor cannot create negative game data.
        self._add_field(
            "player.birth_gold",
            "player",
            off,
            "i32",
            description="Gold at character creation",
        )

        # Body name and variable-length body-slot table.
        c.cstring()
        body_count = c.number("u16")
        if body_count > 128:
            raise SaveFormatError(
                f"Implausible player body slot count: {body_count}"
            )
        for _ in range(body_count):
            c.take(2)  # slot type
            c.cstring()  # slot name

        c.take(4)  # reserved padding

        # Core player values.
        numeric_fields = [
            # The save writer uses wr_u32b() for these four-byte values, but
            # rd_player() reads them with rd_s32b() into signed int32_t fields.
            # Use i32 to match what the game itself will load.
            ("player.gold", "i32", "Current gold", ("gold", "au")),
            ("player.max_exp", "i32", "Maximum experience", ("max_exp",)),
            ("player.exp", "i32", "Current experience", ("experience", "exp")),
            ("player.exp_fraction", "u16", "Experience fraction", ()),
            ("player.level", "i16", "Character level", ("level", "lev")),
            ("player.max_hp", "i16", "Maximum hit points", ("max_hp", "mhp")),
            ("player.hp", "i16", "Current hit points", ("hp", "chp")),
            ("player.hp_fraction", "u16", "Hit point fraction", ()),
            ("player.max_mana", "i16", "Maximum mana", ("max_mana", "msp")),
            ("player.mana", "i16", "Current mana", ("mana", "csp")),
            ("player.mana_fraction", "u16", "Mana fraction", ()),
            ("player.max_level", "i16", "Highest character level", ("max_level",)),
            ("player.max_depth", "i16", "Deepest dungeon depth reached", ("max_depth",)),
            ("player.recall_depth", "i16", "Recall depth", ("recall_depth",)),
        ]
        for name, kind, desc, aliases in numeric_fields:
            off = c.take(struct.calcsize(FORMATS[kind]))
            self._add_field(
                name, "player", off, kind,
                description=desc, aliases=aliases
            )

        # Position/flags and player movement state. 4.2.6 writes the old
        # grid coordinates followed by two reserved int16 values ("oops").
        c.take(2)  # old grid y/x
        c.take(4)  # two reserved int16 values
        c.take(1)  # skip command coercion
        c.take(1)  # unignoring
        for name, desc in [
            ("player.deep_descent", "Deep descent counter"),
            ("player.energy", "Player energy"),
            ("player.word_recall", "Word of Recall countdown"),
        ]:
            off = c.take(2)
            self._add_field(
                name, "player", off, "i16", description=desc
            )

        timed_count = c.number("u8")
        if timed_count > 128:
            raise SaveFormatError(
                f"Implausible timed-effect count: {timed_count}"
            )
        off_count = c.pos - 1
        self._add_field(
            "player.timed_effect_count",
            "player",
            off_count,
            "u8",
            description="Number of timed-effect entries",
        )
        for index in range(timed_count):
            off = c.take(2)
            self._add_field(
                f"player.timed_effect_{index}",
                "player",
                off,
                "i16",
                description=(
                    f"Timed-effect raw slot {index} "
                    "(meaning is version-specific)"
                ),
            )

        for name, desc, aliases in [
            ("player.total_energy", "Total energy used", ("total_energy",)),
            ("player.resting_turns", "Turns spent resting", ("resting_turns",)),
        ]:
            off = c.take(4)
            self._add_field(
                name,
                "player",
                off,
                "u32",
                description=desc,
                aliases=aliases,
            )

        # Eight reserved u32s follow.  Validate the exact expected tail before
        # exposing any of the guessed offsets.  This catches same-version save
        # layouts from builds whose player serialization differs.
        remaining = len(player.data) - c.pos
        if remaining != 32:
            raise SaveFormatError(
                f"Unexpected player block tail: found {remaining} bytes after "
                "the known fields; expected 32 reserved bytes. This save's "
                "layout is not confirmed, so named editing is disabled."
            )

        # The misc block has a simple, stable version-1 prefix.
        misc = self._block("misc")
        if misc.version == 1:
            mc = Cursor(misc.data, "misc")
            for name, desc, kind, aliases in [
                ("misc.randart_seed", "Random-artifact seed", "u32", ()),
                ("misc.flavor_seed", "Object-flavor seed", "u32", ()),
                ("misc.total_winner", "Winner flag", "u16", ()),
                ("misc.noscore", "No-score flags", "u16", ()),
                ("misc.is_dead", "Dead-character flag", "u8", ("is_dead",)),
                ("misc.current_turn", "Current game turn", "i32", ("current_turn", "turn")),
            ]:
                off = mc.take(struct.calcsize(FORMATS[kind]))
                self._add_field(
                    name, "misc", off, kind,
                    description=desc, aliases=aliases
                )
        else:
            self.warnings.append(
                f"Misc block version is {misc.version}; skipping known misc fields."
            )

    # ---------- Public API ----------

    def list_blocks(self) -> list[dict]:
        """Return block names, versions, sizes, and checksum status."""
        return [
            {
                "name": b.name,
                "version": b.version,
                "size": b.size,
                "checksum_ok": b.checksum_ok,
            }
            for b in self.blocks
        ]

    def list_fields(self) -> list[dict]:
        """Return each unique named field and its current value."""
        unique = {id(f): f for f in self.fields.values()}
        rows = []
        for f in sorted(unique.values(), key=lambda item: item.name):
            try:
                value = self.get_value(f.name)
            except Exception as exc:  # report field even if malformed
                value = f"<unreadable: {exc}>"
            rows.append({
                "name": f.name,
                "value": value,
                "type": f.kind,
                "description": f.description,
            })
        return rows

    def _resolve(self, name: str) -> Field:
        try:
            return self.fields[name.casefold()]
        except KeyError as exc:
            suggestions = ", ".join(
                f.name for f in self._unique_fields()[:20]
            )
            raise KeyError(
                f"Unknown field {name!r}. Known fields include: {suggestions}"
            ) from exc

    def _unique_fields(self) -> list[Field]:
        return list({id(f): f for f in self.fields.values()}.values())

    def get_value(self, name: str):
        """Read a named field, e.g. get_value('gold') or get_value('player.hp')."""
        f = self._resolve(name)
        block = self._block(f.block)
        if f.kind == "string":
            raw = bytes(block.data[f.offset : f.offset + f.length])
            return raw.split(b"\0", 1)[0].decode("utf-8", errors="replace")
        return struct.unpack_from(FORMATS[f.kind], block.data, f.offset)[0]

    def modify(self, name: str, value) -> None:
        """Modify a discovered field by name and value."""
        f = self._resolve(name)
        block = self._block(f.block)

        # Match hard constraints enforced by Angband 4.2.6's rd_player().
        # These are rejected before a save is ever written.
        if f.name == "player.level":
            try:
                level_value = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "player.level must be an integer from 1 to 50"
                ) from exc
            if level_value < 1 or level_value > 50:
                raise ValueError(
                    "Angband 4.2.6 accepts player.level only from 1 to 50"
                )

        if f.name in {
            "player.gold",
            "player.birth_gold",
            "player.exp",
            "player.max_exp",
        }:
            try:
                signed_value = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{f.name} must be a non-negative integer"
                ) from exc
            if signed_value < 0:
                raise ValueError(f"{f.name} cannot be negative")

        if f.kind == "string":
            if not isinstance(value, str):
                raise TypeError(f"{name} expects text")
            encoded = value.encode("utf-8") + b"\0"
            if len(encoded) != f.length:
                raise ValueError(
                    f"{name} must remain exactly {f.length - 1} UTF-8 bytes "
                    "long; changing string length requires rebuilding the block"
                )
            block.data[f.offset : f.offset + f.length] = encoded
            return

        fmt = FORMATS[f.kind]
        try:
            packed = struct.pack(fmt, value)
        except (struct.error, TypeError) as exc:
            limits = {
                "u8": "0..255",
                "i8": "-128..127",
                "u16": "0..65535",
                "i16": "-32768..32767",
                "u32": "0..4294967295",
                "i32": "-2147483648..2147483647",
            }
            hint = f" (allowed range: {limits[f.kind]})" if f.kind in limits else ""
            raise ValueError(
                f"Invalid value {value!r} for {name} ({f.kind}){hint}"
            ) from exc
        block.data[f.offset : f.offset + len(packed)] = packed

    # Friendly alias for users who prefer explicit setter naming.
    set_value = modify

    def register_offset_field(
        self,
        name: str,
        block_name: str,
        offset: int,
        kind: str,
        description: str = "Custom user-defined field",
    ) -> None:
        """Register an experimental field using a block-relative byte offset.

        Example: register_offset_field('my_flag', 'misc', 20, 'u8')
        Supported kinds: u8, i8, u16, i16, u32, i32, f32, f64.
        """
        if kind not in FORMATS:
            raise ValueError(f"kind must be one of {', '.join(FORMATS)}")
        self._add_field(name, block_name, offset, kind, description=description)

    def find_bytes(
        self, pattern: bytes, block_name: Optional[str] = None
    ) -> list[dict]:
        """Find a byte pattern and return matching block-relative offsets."""
        if not pattern:
            raise ValueError("pattern cannot be empty")
        blocks = [self._block(block_name)] if block_name else self.blocks
        hits = []
        for block in blocks:
            start = 0
            while True:
                index = block.data.find(pattern, start)
                if index < 0:
                    break
                hits.append({
                    "block": block.name,
                    "offset": index,
                    "file_offset": block.data_offset + index,
                })
                start = index + 1
        return hits

    def replace_bytes(
        self,
        old: bytes,
        new: bytes,
        *,
        block_name: Optional[str] = None,
        occurrence: Optional[int] = None,
    ) -> int:
        """Replace a byte pattern without changing length; updates block checksum on save.

        By default this refuses ambiguous matches. Set occurrence to a zero-based
        match index to select one match, or use block_name to narrow the search.
        """
        if not old:
            raise ValueError("old pattern cannot be empty")
        if len(old) != len(new):
            raise ValueError("old and new byte sequences must have equal length")
        hits = self.find_bytes(old, block_name)
        if not hits:
            raise ValueError("pattern not found")
        if occurrence is None and len(hits) != 1:
            raise ValueError(
                f"pattern matched {len(hits)} locations; narrow by block_name "
                "or provide occurrence=<zero-based index>"
            )
        if occurrence is None:
            occurrence = 0
        if occurrence < 0 or occurrence >= len(hits):
            raise IndexError("occurrence index is out of range")
        hit = hits[occurrence]
        block = self._block(hit["block"])
        offset = hit["offset"]
        block.data[offset : offset + len(old)] = new
        return 1

    def hexdump(
        self, block_name: str, start: int = 0, length: int = 256
    ) -> str:
        """Return a compact hex/ASCII dump of a block."""
        block = self._block(block_name)
        if start < 0 or length < 0 or start > len(block.data):
            raise ValueError("invalid hexdump range")
        end = min(len(block.data), start + length)
        lines = []
        for pos in range(start, end, 16):
            chunk = bytes(block.data[pos : min(pos + 16, end)])
            hex_part = " ".join(f"{byte:02x}" for byte in chunk)
            ascii_part = "".join(
                chr(b) if 32 <= b < 127 else "." for b in chunk
            )
            lines.append(f"{pos:08x}  {hex_part:<47}  {ascii_part}")
        return "\n".join(lines)

    def save(
        self,
        output: Optional[os.PathLike | str] = None,
        *,
        overwrite: bool = False,
        in_place: bool = False,
    ) -> Path:
        """Write a minimal, verified patch for Vanilla Angband 4.2.6.

        Only edited payload bytes and the corresponding block checksum word
        are changed.  Block order, sizes, padding, and all unrelated blocks
        remain byte-for-byte identical.

        The default destination is a new *_edited* file. Existing destinations
        are refused unless overwrite=True. The source itself is refused unless
        in_place=True; in-place replacement always gets a unique backup.
        The original on-disk checksums must all be valid before any save.
        """
        if in_place and output is not None:
            raise ValueError("Do not combine in_place=True with an output path")
        if in_place:
            output_path = self.path
        elif output is None:
            output_path = self.path.with_name(
                f"{self.path.stem}_edited{self.path.suffix}"
            )
        else:
            output_path = Path(output).expanduser().resolve()

        source_is_destination = output_path == self.path
        if source_is_destination and not in_place:
            raise SaveFormatError(
                "Refusing to overwrite the original save. Use in_place=True "
                "only after making sure you have a separate copy."
            )
        if not source_is_destination and output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Output already exists: {output_path}. Choose another path or "
                "explicitly enable overwrite."
            )
        if self.input_checksum_mismatches:
            names = ", ".join(self.input_checksum_mismatches)
            raise SaveFormatError(
                "Refusing to edit a source save that already has invalid block "
                f"checksum(s): {names}. Keep the original and inspect/recover it "
                "before attempting an edit."
            )

        out = bytearray(self.original)
        changed_blocks: list[str] = []
        allowed_differences: set[int] = set()

        for block in self.blocks:
            original_data = self.original[
                block.data_offset:block.data_offset + block.size
            ]
            edited_data = bytes(block.data)

            if len(edited_data) != block.size:
                raise SaveFormatError(
                    f"Refusing to resize block '{block.name}': "
                    f"original={block.size}, edited={len(edited_data)}"
                )
            if edited_data == original_data:
                continue

            # Patch only contiguous runs of changed payload bytes.
            i = 0
            while i < block.size:
                if edited_data[i] == original_data[i]:
                    i += 1
                    continue
                run_start = i
                while i < block.size and edited_data[i] != original_data[i]:
                    i += 1
                run_end = i
                file_start = block.data_offset + run_start
                out[file_start:file_start + (run_end - run_start)] = (
                    edited_data[run_start:run_end]
                )
                allowed_differences.update(
                    range(file_start, file_start + run_end - run_start)
                )

            # Angband 4.2.6's loader does not compare this checksum while
            # loading, but it is part of the save format and should remain
            # correct for tools and future saves. The checksum is the
            # unsigned 32-bit sum of the payload bytes.
            checksum_offset = block.header_offset + 24
            old_checksum_bytes = bytes(out[checksum_offset:checksum_offset + 4])
            new_checksum_bytes = struct.pack(
                "<I", sum(edited_data) & 0xFFFFFFFF
            )
            out[checksum_offset:checksum_offset + 4] = new_checksum_bytes
            for j, (old_byte, new_byte) in enumerate(
                zip(old_checksum_bytes, new_checksum_bytes)
            ):
                if old_byte != new_byte:
                    allowed_differences.add(checksum_offset + j)

            changed_blocks.append(block.name)

        if not changed_blocks:
            raise SaveFormatError("No changes to save")

        if len(out) != len(self.original):
            raise SaveFormatError("Internal safety check failed: output size changed")

        # The exact set of changed bytes must be limited to edited payloads
        # and their matching four-byte checksum fields.
        actual_differences = {
            i for i, (old, new_byte) in enumerate(zip(self.original, out))
            if old != new_byte
        }
        if actual_differences != allowed_differences:
            unexpected = sorted(actual_differences ^ allowed_differences)[:12]
            raise SaveFormatError(
                "Internal safety check failed: unexpected byte differences at "
                + ", ".join(hex(x) for x in unexpected)
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        source_stat = self.path.stat()

        # Never clobber an earlier backup. Use .bak, then .bak.1, .bak.2, ...
        backup_path = output_path.with_suffix(output_path.suffix + ".bak")
        if source_is_destination or (output_path.exists() and overwrite):
            suffix = 1
            while backup_path.exists():
                backup_path = output_path.with_suffix(
                    output_path.suffix + f".bak.{suffix}"
                )
                suffix += 1

        fd, temp_name = tempfile.mkstemp(
            prefix=output_path.name + ".", suffix=".tmp", dir=output_path.parent
        )
        try:
            # Preserve source ownership when running as root. When running as
            # the source owner, preserve its group if it is one of the
            # caller's groups. The mode is copied from the source save.
            try:
                if hasattr(os, "geteuid") and os.geteuid() == 0:
                    os.fchown(fd, source_stat.st_uid, source_stat.st_gid)
                elif source_stat.st_uid == os.geteuid():
                    allowed_groups = set(os.getgroups()) | {os.getegid()}
                    if source_stat.st_gid in allowed_groups:
                        try:
                            os.fchown(fd, -1, source_stat.st_gid)
                        except PermissionError:
                            pass
                os.fchmod(fd, stat.S_IMODE(source_stat.st_mode))
            except OSError as exc:
                raise SaveFormatError(
                    "Could not preserve the save file's ownership/permissions: "
                    f"{exc}"
                ) from exc

            with os.fdopen(fd, "wb") as stream:
                stream.write(out)
                stream.flush()
                os.fsync(stream.fileno())

            # Reparse the temporary file and verify the complete block layout.
            verified = AngbandSaveEditor(temp_name)
            expected_blocks = [
                (
                    b.name, b.version, b.size, b.header_offset,
                    b.data_offset, b.padding,
                )
                for b in self.blocks
            ]
            actual_blocks = [
                (
                    b.name, b.version, b.size, b.header_offset,
                    b.data_offset, b.padding,
                )
                for b in verified.blocks
            ]
            if actual_blocks != expected_blocks:
                raise SaveFormatError(
                    "Verification failed: block layout or padding changed"
                )

            # Every block, changed or untouched, must now have a valid checksum.
            bad_after_write = [
                b.name for b in verified.blocks if not b.checksum_ok
            ]
            if bad_after_write:
                raise SaveFormatError(
                    "Verification failed: invalid checksum(s) in output: "
                    + ", ".join(bad_after_write)
                )

            # Verify the discovered fields round-trip to their edited values.
            for field in self._unique_fields():
                expected_block = self._block(field.block)
                actual_block = verified._block(field.block)
                if field.kind == "string":
                    expected_value = bytes(
                        expected_block.data[
                            field.offset:field.offset + field.length
                        ]
                    ).split(b"\0", 1)[0].decode("utf-8", errors="replace")
                    actual_value = bytes(
                        actual_block.data[
                            field.offset:field.offset + field.length
                        ]
                    ).split(b"\0", 1)[0].decode("utf-8", errors="replace")
                else:
                    expected_value = struct.unpack_from(
                        FORMATS[field.kind], expected_block.data, field.offset
                    )[0]
                    actual_value = struct.unpack_from(
                        FORMATS[field.kind], actual_block.data, field.offset
                    )[0]
                if actual_value != expected_value:
                    raise SaveFormatError(
                        f"Verification failed: {field.name} did not round-trip"
                    )

            # Independently compare the bytes read from disk. No byte outside
            # an edited payload run or its block checksum may have changed.
            with open(temp_name, "rb") as stream:
                disk_bytes = stream.read()
            if len(disk_bytes) != len(self.original):
                raise SaveFormatError("Verification failed: file size changed")
            if any(
                disk_bytes[i] != self.original[i]
                for i in range(len(disk_bytes))
                if i not in allowed_differences
            ):
                raise SaveFormatError(
                    "Verification failed: a byte outside the requested field "
                    "or its checksum changed"
                )

            # Do not commit an edit based on a stale snapshot. This also catches
            # Angband saving the character after this editor opened it.
            try:
                current_source = self.path.read_bytes()
            except OSError as exc:
                raise SaveFormatError(
                    f"Could not re-read the source save before commit: {exc}"
                ) from exc
            if current_source != self.original:
                raise SaveFormatError(
                    "The source save changed on disk after the editor opened it. "
                    "Nothing was replaced; reopen the newest save and retry."
                )

            # Re-check the destination immediately before committing.
            if not source_is_destination and output_path.exists() and not overwrite:
                raise FileExistsError(
                    f"Output appeared while saving: {output_path}. Nothing was replaced."
                )

            # Keep a unique backup before replacing an existing file.
            if source_is_destination or (output_path.exists() and overwrite):
                suffix = 1
                while backup_path.exists():
                    backup_path = output_path.with_suffix(
                        output_path.suffix + f".bak.{suffix}"
                    )
                    suffix += 1
                if source_is_destination:
                    shutil.copy2(self.path, backup_path)
                else:
                    shutil.copy2(output_path, backup_path)

            os.replace(temp_name, output_path)

        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            try:
                os.unlink(temp_name)
            except OSError:
                pass
            raise

        print(
            "Patched block(s): " + ", ".join(changed_blocks),
            file=sys.stderr,
        )
        print(
            "Verified all block checksums; unchanged file regions were "
            "preserved byte-for-byte.",
            file=sys.stderr,
        )
        if source_is_destination:
            print(f"Backup: {backup_path}", file=sys.stderr)
        elif backup_path.exists():
            print(f"Replaced-file backup: {backup_path}", file=sys.stderr)
        return output_path

    def summary(self) -> str:
        lines = [
            f"File: {self.path}",
            f"Size: {len(self.original)} bytes",
            "Blocks:",
        ]
        for b in self.blocks:
            status = "OK" if b.checksum_ok else "MISMATCH"
            lines.append(
                f"  {b.name:<18} version={b.version:<3} "
                f"size={b.size:<8} checksum={status}"
            )
        if self.warnings:
            lines.append("Warnings:")
            lines.extend(f"  - {w}" for w in self.warnings)
        lines.append(
            "Discovered editable fields: "
            f"{len({id(f) for f in self.fields.values()})}"
        )
        return "\n".join(lines)


def parse_value(text: str):
    """Parse CLI values as int (supports 0x hex), float, or plain text."""
    try:
        return int(text, 0)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


def interactive(editor: AngbandSaveEditor) -> None:
    print(editor.summary())
    print(
        "\nCommands: list | get NAME | set NAME VALUE | blocks | "
        "dump BLOCK [OFFSET] | save [PATH] | quit"
    )
    while True:
        try:
            line = input("angband-editor> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        parts = line.split()
        command = parts[0].casefold()
        try:
            if command in {"quit", "exit"}:
                return
            elif command == "list":
                for row in editor.list_fields():
                    print(
                        f"{row['name']:<34} {row['value']!r:<16} "
                        f"{row['description']}"
                    )
            elif command == "blocks":
                for block in editor.list_blocks():
                    print(block)
            elif command == "get" and len(parts) == 2:
                print(editor.get_value(parts[1]))
            elif command == "set" and len(parts) >= 3:
                name = parts[1]
                raw_value = " ".join(parts[2:])
                value = parse_value(raw_value)
                editor.modify(name, value)
                print(f"Set {name} = {value!r}")
            elif command == "dump" and len(parts) in {2, 3}:
                offset = int(parts[2], 0) if len(parts) == 3 else 0
                print(editor.hexdump(parts[1], offset))
            elif command == "save" and len(parts) in {1, 2}:
                dest = editor.save(parts[1] if len(parts) == 2 else None)
                print(f"Saved: {dest}")
            else:
                print(
                    "Invalid command. Try: list, get NAME, set NAME VALUE, "
                    "blocks, dump BLOCK [OFFSET], save [PATH], quit"
                )
        except Exception as exc:
            print(f"Error: {exc}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Experimental Vanilla Angband save-file editor"
    )
    parser.add_argument("savefile", help="path to a Vanilla Angband save file")
    parser.add_argument("--list", action="store_true", help="list discovered fields")
    parser.add_argument("--blocks", action="store_true", help="list save-file blocks")
    parser.add_argument("--get", metavar="FIELD", help="print a field value")
    parser.add_argument(
        "--set",
        nargs=2,
        metavar=("FIELD", "VALUE"),
        help="modify a named field",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="output path (default: *_edited*)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacing an existing output (a unique .bak is kept)",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="explicitly replace the input save (a unique .bak is kept)",
    )
    args = parser.parse_args(argv)

    if args.in_place and args.output:
        parser.error("--in-place cannot be combined with --output")
    if args.overwrite and args.in_place:
        parser.error("--overwrite is unnecessary with --in-place")

    try:
        editor = AngbandSaveEditor(args.savefile)
        if args.blocks:
            for item in editor.list_blocks():
                print(item)
        if args.list:
            for item in editor.list_fields():
                print(
                    f"{item['name']:<34} {item['value']!r:<16} "
                    f"{item['description']}"
                )
        if args.get:
            print(editor.get_value(args.get))
        if args.set:
            field_name, raw_value = args.set
            value = parse_value(raw_value)
            editor.modify(field_name, value)
            saved = editor.save(
                args.output,
                overwrite=args.overwrite,
                in_place=args.in_place,
            )
            print(
                f"Set {field_name} = {value!r}\n"
                f"Saved edited file to: {saved}"
            )
        if not (args.blocks or args.list or args.get or args.set):
            interactive(editor)
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
