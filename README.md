# Guess the Card

Windows helper for the Guess the Card event in Anime Card Clash. It reads the
four choices from the Roblox window and compares the blurred card with its
image catalog. It uses normal foreground mouse input and does not read or write Roblox process memory or inject anything.

## Download

- [Guess-The-Card-Setup.zip](https://github.com/senkoisbroken/Guess-The-Card/releases/latest/download/Guess-The-Card-Setup.zip)
- [Latest release, checksum, and scan notes](https://github.com/senkoisbroken/Guess-The-Card/releases/latest)

The [VirusTotal report for an earlier setup ZIP](https://www.virustotal.com/gui/file/fb6d5cc0efcf015fa4e76aed13fdbcd10869d8d74813fca6f970d084280d57d5?nocache=1)
covers a different SHA-256. See the latest release notes for the Defender
scan of the current download.

Extract the ZIP and run **Start.bat**. Open the event in Roblox and press **F8**
to start or pause. The first launch sets up 64-bit Python 3.13 and dependencies
if needed.

To update, close the macro and run **Update.bat**. It downloads the latest
GitHub release, verifies SHA-256, and preserves learned cards, logs, and the
Python environment. No GitHub login is needed. If your copy still asks for a
ZIP, select the latest setup ZIP once to switch to online updates.

The runnable source and catalog are in [Guess-The-Card/](Guess-The-Card/).
Use the release ZIP for the packaged Windows launchers.
