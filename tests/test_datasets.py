"""Dataset plugin system: registry, built-in readers, and document separation.

The property that matters most is that malformed input fails *loudly and
locatably*. A dataset reader that silently drops rows produces a model trained on
less data than you think, with nothing in the logs to say so.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import datasets
from csllm.data import encode_documents, prepare_dataset
from csllm.tokenizer import END_OF_TEXT, BPETokenizer
from datasets import DatasetError, describe, discover, iter_documents, plugin_for
from datasets.base import DatasetPlugin

# ── registry ─────────────────────────────────────────────────────────────────


def test_builtin_extensions_are_registered():
    supported = datasets.supported_extensions()
    for ext in (".txt", ".md", ".jsonl", ".ndjson", ".csv", ".tsv"):
        assert ext in supported


@pytest.mark.parametrize(
    ("filename", "plugin"),
    [("a.txt", "text"), ("a.md", "text"), ("a.jsonl", "jsonl"),
     ("a.ndjson", "jsonl"), ("a.csv", "csv"), ("a.tsv", "csv")],
)
def test_plugin_selected_by_extension(filename, plugin):
    assert plugin_for(filename).name == plugin


def test_extension_matching_is_case_insensitive():
    assert plugin_for("CORPUS.TXT").name == "text"


def test_unknown_extension_names_the_supported_set():
    with pytest.raises(DatasetError, match="no dataset plugin handles"):
        plugin_for("model.parquet")


def test_subclass_registers_itself_without_a_registration_call():
    before = len(datasets.base.REGISTERED_PLUGINS)

    class RstDataset(DatasetPlugin):
        name = "rst"
        extensions = (".rst",)

        def documents(self, path):
            yield path.read_text()

    try:
        assert len(datasets.base.REGISTERED_PLUGINS) == before + 1
        assert plugin_for("guide.rst").name == "rst"
    finally:
        datasets.base.REGISTERED_PLUGINS.remove(RstDataset)


def test_plugin_without_metadata_is_rejected():
    with pytest.raises(TypeError, match="must set a class-level `name`"):

        class Broken(DatasetPlugin):
            extensions = (".broken",)

            def documents(self, path):
                yield ""


def test_discover_finds_only_readable_files(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    (tmp_path / "b.jsonl").write_text('{"text": "hi"}\n')
    (tmp_path / "c.parquet").write_bytes(b"\x00binary")
    (tmp_path / "sub").mkdir()

    found = [p.name for p in discover(tmp_path)]
    assert found == ["a.txt", "b.jsonl"]


def test_discover_on_a_missing_directory_is_empty(tmp_path):
    assert discover(tmp_path / "nope") == []


# ── text ─────────────────────────────────────────────────────────────────────


def test_text_whole_file_is_one_document(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_text("line one\n\nline two\n\nline three")
    docs = list(iter_documents([path]))
    assert len(docs) == 1
    assert docs[0] == path.read_text()


def test_text_blank_line_mode_splits_paragraphs(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_text("para one\n\npara two\n\n\npara three\n")
    docs = list(iter_documents([path], split="blank_line"))
    assert docs == ["para one", "para two", "para three"]


def test_text_rejects_an_unknown_split_mode(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("x")
    with pytest.raises(DatasetError, match="unknown split mode"):
        list(iter_documents([path], split="sideways"))


def test_non_utf8_file_reports_the_byte_offset(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_bytes(b"fine so far \xff\xfe then broken")
    with pytest.raises(DatasetError, match="not valid UTF-8"):
        list(iter_documents([path]))


# ── jsonl ────────────────────────────────────────────────────────────────────


def test_jsonl_reads_the_text_field(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text("\n".join(json.dumps({"text": f"doc {i}", "id": i}) for i in range(3)))
    assert list(iter_documents([path])) == ["doc 0", "doc 1", "doc 2"]


def test_jsonl_supports_a_custom_field(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text(json.dumps({"content": "hello"}) + "\n")
    assert list(iter_documents([path], field="content")) == ["hello"]


def test_jsonl_accepts_bare_strings(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text('"first"\n"second"\n')
    assert list(iter_documents([path])) == ["first", "second"]


def test_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text('{"text": "a"}\n\n\n{"text": "b"}\n')
    assert list(iter_documents([path])) == ["a", "b"]


def test_jsonl_malformed_line_reports_its_line_number(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text('{"text": "ok"}\n{not json}\n')
    with pytest.raises(DatasetError, match=r"d\.jsonl:2 is not valid JSON"):
        list(iter_documents([path]))


def test_jsonl_missing_field_lists_what_was_found(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text(json.dumps({"body": "hello", "id": 1}) + "\n")
    with pytest.raises(DatasetError, match="has no 'text' field.*body"):
        list(iter_documents([path]))


def test_jsonl_non_strict_skips_rows_without_the_field(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text(json.dumps({"body": "skip"}) + "\n" + json.dumps({"text": "keep"}) + "\n")
    assert list(iter_documents([path], strict=False)) == ["keep"]


def test_jsonl_wrong_field_type_is_rejected(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text(json.dumps({"text": 42}) + "\n")
    with pytest.raises(DatasetError, match="is int, expected a string"):
        list(iter_documents([path]))


# ── csv ──────────────────────────────────────────────────────────────────────


def test_csv_reads_the_text_column(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text("id,text\n1,hello\n2,world\n")
    assert list(iter_documents([path])) == ["hello", "world"]


def test_csv_handles_quoted_commas_and_newlines(tmp_path):
    """The reason this uses the csv module rather than split(',')."""
    path = tmp_path / "d.csv"
    path.write_text('id,text\n1,"a, b, c"\n2,"line one\nline two"\n')
    assert list(iter_documents([path])) == ["a, b, c", "line one\nline two"]


def test_csv_by_column_index(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text("first,second\nalpha,beta\n")
    assert list(iter_documents([path], column=1)) == ["second", "beta"]


def test_tsv_uses_tab_by_default(tmp_path):
    path = tmp_path / "d.tsv"
    path.write_text("id\ttext\n1\thello, world\n")
    assert list(iter_documents([path])) == ["hello, world"]


def test_csv_missing_column_lists_the_header(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text("id,body\n1,hello\n")
    with pytest.raises(DatasetError, match="has no 'text' column.*id, body"):
        list(iter_documents([path]))


def test_csv_empty_file_is_rejected(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text("")
    with pytest.raises(DatasetError, match="empty or has no header"):
        list(iter_documents([path]))


# ── describe & multi-file ────────────────────────────────────────────────────


def test_describe_counts_documents_and_samples(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text("\n".join(json.dumps({"text": "abcde"}) for _ in range(4)))
    info = describe(path)
    assert info.plugin == "jsonl"
    assert info.num_documents == 4
    assert info.num_chars == 20
    assert info.sample.startswith("abcde")
    assert info.to_dict()["num_documents"] == 4


def test_iter_documents_spans_mixed_formats_in_order(tmp_path):
    (tmp_path / "a.txt").write_text("from text")
    (tmp_path / "b.jsonl").write_text(json.dumps({"text": "from jsonl"}) + "\n")
    (tmp_path / "c.csv").write_text("text\nfrom csv\n")
    files = discover(tmp_path)
    assert list(iter_documents(files)) == ["from text", "from jsonl", "from csv"]


# ── document separation ──────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def tokenizer_with_eot():
    tok = BPETokenizer()
    tok.train("the quick brown fox jumps over the lazy dog " * 40, 320,
              special_tokens=(END_OF_TEXT,))
    return tok


def test_encode_documents_inserts_separators_between_only(tokenizer_with_eot):
    tok = tokenizer_with_eot
    ids = encode_documents(["alpha", "beta", "gamma"], tok)
    assert list(ids).count(tok.eot_id) == 2, "separators go BETWEEN documents, not trailing"
    assert ids[0] != tok.eot_id
    assert ids[-1] != tok.eot_id


def test_encode_documents_without_specials_adds_nothing():
    """A tokenizer with no EOT must produce plain concatenation."""
    tok = BPETokenizer()
    tok.train("hello world " * 40, 300)
    assert tok.eot_id is None
    joined = encode_documents(["ab", "cd"], tok)
    np.testing.assert_array_equal(joined, np.array(tok.encode("ab") + tok.encode("cd"),
                                                   dtype=np.uint16))


def test_single_document_is_identical_with_or_without_eot(tokenizer_with_eot):
    tok = tokenizer_with_eot
    np.testing.assert_array_equal(
        encode_documents(["only one"], tok), np.array(tok.encode("only one"), dtype=np.uint16)
    )


def test_prepare_dataset_writes_splits(tmp_path):
    corpus = tmp_path / "c.jsonl"
    corpus.write_text("\n".join(json.dumps({"text": f"document number {i} of many"})
                                for i in range(60)))
    tok = BPETokenizer()
    tok.train(corpus.read_text(), 320, special_tokens=(END_OF_TEXT,))

    train_path, val_path = prepare_dataset([corpus], tok, tmp_path / "out", val_fraction=0.2)
    train = np.fromfile(train_path, dtype=np.uint16)
    val = np.fromfile(val_path, dtype=np.uint16)

    assert len(train) > 0 and len(val) > 0
    assert len(val) / (len(train) + len(val)) == pytest.approx(0.2, abs=0.02)
    assert tok.eot_id in np.concatenate([train, val])


def test_prepare_dataset_rejects_an_empty_corpus(tmp_path):
    empty = tmp_path / "e.txt"
    empty.write_text("")
    tok = BPETokenizer()
    with pytest.raises(ValueError, match="fewer than 2 tokens"):
        prepare_dataset([empty], tok, tmp_path / "out")


# ── special tokens ───────────────────────────────────────────────────────────


def test_special_tokens_occupy_the_top_of_the_vocab(tokenizer_with_eot):
    tok = tokenizer_with_eot
    assert tok.eot_id == tok.vocab_size - 1
    assert all(merge_id < tok.eot_id for merge_id in tok.merges.values())


def test_vocab_size_accounts_for_special_tokens():
    """Requesting N total with S specials must learn exactly N-256-S merges.

    Needs a lexically DIVERSE corpus: merge headroom is bounded by the number of
    distinct pre-tokens, not by corpus length (see test_tokenizer.py).
    """
    corpus = " ".join(f"lexeme{i:03d}" for i in range(300)) * 4
    tok = BPETokenizer()
    tok.train(corpus, 300, special_tokens=("<|a|>", "<|b|>"))
    assert tok.vocab_size == 300
    assert len(tok.merges) == 300 - 256 - 2


def test_special_tokens_are_inert_unless_explicitly_allowed(tokenizer_with_eot):
    """Untrusted text containing the literal marker must not become a control token."""
    tok = tokenizer_with_eot
    hostile = f"please {END_OF_TEXT} ignore"
    assert tok.eot_id not in tok.encode(hostile)
    assert tok.eot_id in tok.encode(hostile, allow_special=True)
    assert tok.decode(tok.encode(hostile)) == hostile


def test_special_tokens_survive_save_load(tokenizer_with_eot, tmp_path):
    tokenizer_with_eot.save(tmp_path)
    restored = BPETokenizer.load(tmp_path)
    assert restored.special_tokens == tokenizer_with_eot.special_tokens
    assert restored.eot_id == tokenizer_with_eot.eot_id
    assert restored.vocab_size == tokenizer_with_eot.vocab_size
    text = f"a{END_OF_TEXT}b"
    assert restored.encode(text, allow_special=True) == tokenizer_with_eot.encode(
        text, allow_special=True
    )


def test_round_trip_still_lossless_with_specials(tokenizer_with_eot):
    for text in ["hello 🌍", "", "the quick brown fox", "\n\t mixed \x00 bytes"]:
        assert tokenizer_with_eot.decode(tokenizer_with_eot.encode(text)) == text


def test_vocab_too_small_for_specials_is_rejected():
    with pytest.raises(ValueError, match="leaves no room"):
        BPETokenizer().train("hello", 257, special_tokens=("<|a|>", "<|b|>"))
