"""# Get FASTA

Extracts sequences from a reference genome at positions specified by BED
coordinates.

Uses an indexed `pysam` fetch when pysam is importable, and falls back to
`bedtools getfasta` otherwise. Both paths produce byte-identical output.

Why the pysam path exists: for probe-specificity workloads this lookup performs
one indexed read per alignment (10^4-10^5 reads per probe). Profiling the
alignment step on a repetitive probe against CHM13 showed `process_alignments`
-- essentially this function -- accounting for ~96% of the step, far more than
bowtie2 itself, and it does not improve with more threads. The bedtools path
pays, on top of the reads themselves, a subprocess fork/exec, BED
serialization, an awk pipe and a text re-parse; the pysam path issues the same
indexed reads in-process through a single persistent FASTA handle. Measured
~35x faster on a warm cache with identical sequences returned.
"""

from oligominer.utils import (
    check_input_exists,
    check_output_exists,
    ensure_executable,
    require_one_of,
)
from oligominer.utils.shell_pipeline import ShellPipeline, run_cmd

# Case-preserving IUPAC complement, matching `bedtools getfasta -s`: bedtools
# complements ambiguity codes (R<->Y, K<->M, B<->V, D<->H; S/W/N self-complement)
# and preserves soft-masked (lowercase) bases.
_COMPLEMENT = str.maketrans(
    "ACGTURYSWKMBVDHNacgturyswkmbvdhn",
    "TGCAAYRSWMKVBHDNtgcaayrswmkvbhdn",
)


def _reverse_complement(seq):
    """Reverse complement a sequence, preserving case and IUPAC codes."""
    return seq.translate(_COMPLEMENT)[::-1]


def _iter_bed_records(bed_path=None, bed_data=None):
    """Yield (chrom, start, stop, strand) for each record of a BED file or string.

    Track/browser/comment lines and blank lines are skipped. Records without a
    strand column default to '+', matching bedtools.
    """
    if bed_path is not None:
        with open(bed_path) as handle:
            text = handle.read()
    else:
        text = bed_data

    for line in text.split("\n"):
        if not line or line.startswith(("#", "track", "browser")):
            continue
        fields = line.split("\t")
        strand = fields[5] if len(fields) > 5 else "+"
        yield fields[0], int(fields[1]), int(fields[2]), strand


def _get_fasta_pysam(fasta_path, bed_path, bed_data, strip_col):
    """Indexed pysam implementation of `bedtools getfasta -tab -s`.

    Coordinates are BED (0-based, half-open), '-' strand records are reverse
    complemented, and case is preserved -- identical to bedtools. Requires a
    FASTA index (.fai), which pysam creates if absent and writable.
    """
    import pysam

    handle = pysam.FastaFile(fasta_path)
    try:
        rows = []
        for chrom, start, stop, strand in _iter_bed_records(bed_path, bed_data):
            seq = handle.fetch(chrom, start, stop)
            if strand == "-":
                seq = _reverse_complement(seq)
            if strip_col:
                rows.append(seq)
            else:
                rows.append(f"{chrom}:{start}-{stop}({strand})\t{seq}")
    finally:
        handle.close()

    return "".join(row + "\n" for row in rows)


def _get_fasta_bedtools(fasta_path, bed_path, bed_data, output_file, verbose, strip_col):
    """Run `bedtools getfasta` -- the fallback used when pysam is absent."""
    ensure_executable("bedtools")

    # build the bedtools getfasta command
    bedtools_cmd = ["bedtools", "getfasta", "-tab", "-s", "-fi", fasta_path, "-bed"]
    if bed_path is not None:
        bedtools_cmd.append(bed_path)
        input_data = None
    else:
        bedtools_cmd.append("-")
        input_data = bed_data

    # single command when no stripping needed, pipeline when piping through awk
    if strip_col:
        pipeline = ShellPipeline()
        pipeline.add(bedtools_cmd)
        pipeline.add(["awk", "{ print $2}"])
        return pipeline.run(input_data=input_data, output_file=output_file, verbose=verbose)

    return run_cmd(bedtools_cmd, input_data=input_data, output_file=output_file, verbose=verbose)


def _pysam_available():
    """True when pysam can be imported (enables the fast lookup path)."""
    try:
        import pysam  # noqa: F401
    except ImportError:
        return False
    return True


def get_fasta(
    bed_path=None, bed_data=None, fasta_path=None, output_file=None, verbose=False, strip_col=True
):
    """Extract sequences from a reference FASTA at BED-specified coordinates.

    Looks up sequences with an indexed pysam fetch when pysam is available,
    otherwise shells out to bedtools getfasta. Both paths return the same
    sequences: BED coordinates, '-' strand records reverse complemented, and
    original case preserved. Accepts BED input either as a file path or as
    piped string data. When strip_col is True, only the sequences are returned
    (equivalent to piping bedtools output through awk).

    Args:
        bed_path (str, optional): path to a BED file.
        bed_data (str, optional): BED data as a string.
        fasta_path (str): path to the reference FASTA file.
        output_file (str, optional): path to write output. If None, result
            is returned as a string.
        verbose (bool): if True, print stdout and stderr of each command
            to the terminal. Only applies to the bedtools fallback.
        strip_col (bool): if True, strip the coordinate column from the
            output, returning only sequences.

    Returns:
        result (str or None): extracted sequences, or None if output_file
            is provided.

    Raises:
        ValueError: if neither or both of bed_path and bed_data are provided.
    """
    # ensure the target fasta file exists
    check_input_exists(fasta_path)

    require_one_of(bed_path, bed_data, "bed_path", "bed_data")

    if _pysam_available():
        result = _get_fasta_pysam(fasta_path, bed_path, bed_data, strip_col)
        if output_file is not None:
            with open(output_file, "w") as handle:
                handle.write(result)
            result = None
    else:
        result = _get_fasta_bedtools(
            fasta_path, bed_path, bed_data, output_file, verbose, strip_col
        )

    # check if the output file was created
    if output_file is not None:
        check_output_exists(output_file)

    # success
    return result
