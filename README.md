# Guess the Card visual macro

This is a separate screen-only Windows macro. It captures the Roblox window and uses normal mouse input; it does not read Roblox process memory or use Potassium.

1. Start `start.bat`, open the Guess the Card event, and bring Roblox to the foreground.
2. Press F7 to start or pause. Automatic confident answers and the Give up / Play again sequence are enabled by default; uncheck automatic clicking before starting to preview suggestions without clicking. Give up is only clicked after its text and the Continue heading are recognized; the paid Continue button is never targeted.
3. By default, the macro uses the closest supported candidate among the visible choices when unsure. This can be wrong and cost a life. If it has no usable image and choice evidence, it waits rather than picking randomly. Uncheck the guess setting to leave all uncertain questions unanswered. It checks the question again before clicking so it doesn't send a stale answer to a new round.

The matcher reads the four visible choices and compares the card with catalog or learned images whose names correspond to those choices. It checks the image and text confidence before clicking; if a card image is missing or the choices cannot be read clearly, it can still skip or misidentify a question.

When it skips an unknown card or clicks an incorrect answer, it watches for the game's revealed green answer and can save the blurred card as an example in `assets/learned`. It compares the displayed answer choices before saving, and ignores reveals it cannot confidently pair with the question. Different images can be saved under the same answer name. Learned examples are loaded on restart; they help with repeated cards, but do not cover unseen cards immediately. Keep Roblox foreground while the answer is revealed.

Layout recognition has been checked offline against captures from 1920×1009 and 800×599 client windows. Other sizes and the updated small-window clicking and replay need live testing. The event timer, changing graphics, OCR errors, similar card art, and missing wiki entries can still cause missed or incorrect answers. Some answers in the observed run were provided by the player.

`wiki_catalog.csv` contains 1,520 entries, including the wiki's Card Index and Skin Index, additional individual card pages, and named card artwork found in uploads. Each entry retains its source URL and revision or upload timestamp. Some uploaded cards belong to other modes, and their presence in Guess the Card has not been verified. `wiki_manifest.json` is the machine-readable version. The wiki may be incomplete or out of date; this is not a verified list of every in-game card or skin. `tools/fetch_wiki.py` refreshes the two indexes using the MediaWiki API and preserves the separately collected page and upload images.

Sources: [Card Index](https://anime-card-clash-official.fandom.com/wiki/Card_Index), [Skin Index](https://anime-card-clash-official.fandom.com/wiki/Miscellaneous_Page). Image attribution and source URLs are retained in the catalog. Respect the source wiki's license when redistributing its content.

The setup ZIP installs its own dependencies with 64-bit Python 3.13. F8 and F7 both start or pause. Run Update.bat to install the latest GitHub release. For an offline update, run Update.bat --local. Both preserve local learning and logs. Review diagnostic log lets you inspect the recent log before optionally sending it; screenshots and learned cards are never uploaded.
