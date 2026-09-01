"""Tests for get_fasta: the pysam fast path and its parity with bedtools."""

import shutil

import pytest

from oligominer.specificity.alignment.get_fasta import (
    _get_fasta_bedtools,
    _iter_bed_records,
    _pysam_available,
    _reverse_complement,
    get_fasta,
)

# A reference exercising the cases bedtools handles: soft-masked (lowercase)
# bases, N, and IUPAC ambiguity codes.
_REF_SEQ = "ACGTacgtNNRYacgtACGT"

# +/- strand records, including one spanning the N/IUPAC stretch.
_BED = (
    "c1\t0\t8\tnameA\t0\t+\n"
    "c1\t0\t8\tnameB\t0\t-\n"
    "c1\t8\t14\tnameC\t0\t-\n"
    "c1\t12\t20\tnameD\t0\t+\n"
)


@pytest.fixture
def ref_fasta(tmp_path):
    """A small indexed reference FASTA (pysam needs the .fai)."""
    pysam = pytest.importorskip("pysam")
    path = tmp_path / "ref.fa"
    path.write_text(">c1\n%s\n" % _REF_SEQ)
    pysam.faidx(str(path))
    return str(path)


# ---------------------------------------------------------------------------
# pure helpers (no external tools)
# ---------------------------------------------------------------------------

class TestReverseComplement:
    """The reverse complement must match bedtools' behavior exactly."""

    def test_preserves_case(self):
        # bedtools getfasta -s preserves soft-masking through the complement.
        assert _reverse_complement("ACGTacgt") == "acgtACGT"

    def test_complements_iupac_and_n(self):
        # R<->Y, K<->M, B<->V, D<->H; S/W/N self-complement.
        assert _reverse_complement("RYKMBVDHSWN") == "NWSDHBVKMRY"

    def test_is_an_involution(self):
        seq = "ACGTacgtNNRYacgtACGT"
        assert _reverse_complement(_reverse_complement(seq)) == seq


class TestIterBedRecords:
    """BED parsing: coordinates, strand default, and skipped lines."""

    def test_parses_coordinates_and_strand(self):
        records = list(_iter_bed_records(bed_data=_BED))
        assert records[0] == ("c1", 0, 8, "+")
        assert records[1] == ("c1", 0, 8, "-")
        assert len(records) == 4

    def test_defaults_to_plus_without_strand_column(self):
        assert list(_iter_bed_records(bed_data="c1\t0\t8\n")) == [("c1", 0, 8, "+")]

    def test_skips_blank_and_header_lines(self):
        data = "track name=x\n#comment\n\nc1\t0\t8\tn\t0\t+\n"
        assert list(_iter_bed_records(bed_data=data)) == [("c1", 0, 8, "+")]


# ---------------------------------------------------------------------------
# lookup behavior
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _pysam_available(), reason="pysam not installed")
class TestGetFastaPysam:
    """The pysam path returns bedtools-shaped output."""

    def test_strip_col_returns_bare_sequences(self, ref_fasta):
        result = get_fasta(bed_data=_BED, fasta_path=ref_fasta, strip_col=True)
        assert result == "ACGTacgt\nacgtACGT\ngtRYNN\nacgtACGT\n"

    def test_without_strip_col_includes_coordinate_column(self, ref_fasta):
        result = get_fasta(bed_data=_BED, fasta_path=ref_fasta, strip_col=False)
        assert result.startswith("c1:0-8(+)\tACGTacgt\n")

    def test_bed_path_and_bed_data_agree(self, ref_fasta, tmp_path):
        bed_file = tmp_path / "in.bed"
        bed_file.write_text(_BED)
        from_data = get_fasta(bed_data=_BED, fasta_path=ref_fasta)
        from_path = get_fasta(bed_path=str(bed_file), fasta_path=ref_fasta)
        assert from_data == from_path

    def test_output_file_is_written_and_returns_none(self, ref_fasta, tmp_path):
        out = tmp_path / "out.txt"
        result = get_fasta(bed_data=_BED, fasta_path=ref_fasta, output_file=str(out))
        assert result is None
        assert out.read_text() == get_fasta(bed_data=_BED, fasta_path=ref_fasta)


@pytest.mark.skipif(
    not _pysam_available() or shutil.which("bedtools") is None,
    reason="needs both pysam and bedtools to compare",
)
@pytest.mark.parametrize("strip_col", [True, False])
def test_pysam_output_is_identical_to_bedtools(ref_fasta, strip_col):
    """The fast path must be byte-identical to `bedtools getfasta -tab -s`."""
    fast = get_fasta(bed_data=_BED, fasta_path=ref_fasta, strip_col=strip_col)
    reference = _get_fasta_bedtools(ref_fasta, None, _BED, None, False, strip_col)
    assert fast == reference
