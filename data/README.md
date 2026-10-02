# Sample Data

## stackoverflow_sample.json

Source: [mikex86/stackoverflow-posts](https://huggingface.co/datasets/mikex86/stackoverflow-posts)
on the Hugging Face Hub, a mirror of the
[Stack Exchange Data Dump](https://archive.org/details/stackexchange).

**License:** Stack Exchange user contributions are licensed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). This sample is a
derivative of that data and is redistributed under the same terms. Each record
keeps its original `content_license` field from the source dump for per-post
provenance.

**Attribution:** Content originally contributed by Stack Overflow users via the
Stack Exchange Data Dump.

**Fields:**
- `title` — question title (semantic/free-text clustering field)
- `tags` — list of tags (lexical/list clustering field)
- `user_id` — the original StackOverflow `OwnerUserId` (join field example)
- `content_license` — license of the original post, carried through from the source dump

**Regenerate or resample:**
```
pip install -r scripts/requirements.txt
python scripts/fetch_stackoverflow_sample.py --start-date YYYY-MM-DD --end-date YYYY-MM-DD --limit N
```
See `scripts/fetch_stackoverflow_sample.py` for all options.
