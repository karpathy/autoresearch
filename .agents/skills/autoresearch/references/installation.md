# Installation and updates

The installer fingerprints the source files it copies and writes a schema 2 `skill.manifest.json` into the target; source checkouts need no manifest. It stages and verifies the new copy before activating it, keeps a backup during replacement, and restores the previous target if post-install verification fails.

```text
python SKILL_DIR/install.py --target agents
python SKILL_DIR/install.py --target codex
```

An existing target is updated only when its manifest is valid and every installed file is still owned and unchanged. Any modified managed file, untracked file or directory, symlink, or unsupported old manifest causes the installer to stop before writing. There is no force option that adopts or deletes unknown content. A schema 1 installation cannot verify references or the installer itself; back it up and install into a clean destination. Preserve old lab data through the explicit read-only ZIP export, then prepare a new lab separately.

The installed manifest detects file-set or content differences; it is not signed and does not establish who produced the package. Run the installer from a source package obtained through a trusted channel. Source checkouts carry no manifest, and the installer writes one only into the target it creates or updates; runtime checks never create a missing manifest or silently update its baseline.
