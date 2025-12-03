# Nanopublications

This directory contains nanopublication-related files including:
- Nanopub wave files (.trig format)
- Hypothesis mapping files
- Scripts for processing and canonicalizing nanopublications

## Directory Structure

- **`waves/`**: Complete set of nanopublication wave files (np_levin_nanopubs_wave*.trig, 19 files including waves A-R)
- **`npi-waves/`**: Subset of wave files with different structure (10 files, appears to be alternative/processed versions)
- **`waves_fixed/`**: Fixed/canonicalized versions of wave files (empty)
- **`*.trig`**: Hypothesis mapping files and other nanopub files

**Note:** The batch publisher defaults to publishing `waves/` only (the complete set). Use `--include npi-waves/` if you want to publish those files as well, but they appear to be duplicates/alternatives of files already in `waves/`.

## Scripts

### Processing Scripts
- **`np_canonicalize_slash_uris.py`**: Canonicalize nanopubs and normalize graph IRIs
- **`nanopub_restructure.py`**: Restructure nanopublication files
- **`planform_to_nanopubs.py`**: Convert PlanformDB data to nanopublications

### Publishing Scripts
- **`publish_all_nanopubs.py`**: Batch script to sign and publish all nanopublication files
- **`publish_batch.sh`**: Convenient wrapper script for batch publishing

### Utility Scripts
- **`np`** or **`np_utils.py`**: Command-line utilities for nanopub operations
  - `check <file.trig>` - Check if a nanopub is valid
  - `mktrusty <file.trig> [-o output]` - Create trusty URI version
  - `sign <file.trig> [-o output]` - Sign a nanopub
  - `publish <file.trig>` - Publish a nanopub

## Files

- **`np_hypothesis_mappings_A_to_Q.trig`**: Hypothesis mappings
- **`hypothesis_mappings_A_to_Q.trig`**: Alternative hypothesis mappings format

## Publishing Nanopublications

### Setup

Before publishing, you need to install the nanopub library and set up your profile:

```bash
pip install nanopub rdflib
python3 -m nanopub setup --newkeys  # Interactive setup - generates RSA keys and asks for ORCID
```

**Important:** Use the `--newkeys` flag to automatically generate RSA keys. The setup will:
1. Ask for your ORCID iD (e.g., `https://orcid.org/0000-0000-0000-0000`)
2. Ask for your name
3. Generate RSA keys automatically
4. Store configuration in `~/.nanopub/`

**Note:** Make sure your virtual environment is activated before running the setup command.

See [`SETUP_GUIDE.md`](./SETUP_GUIDE.md) for detailed setup instructions and troubleshooting.

### Batch Publishing

To sign and publish all nanopublication files:

```bash
# Preview what would be published (dry-run)
./publish_batch.sh --dry-run

# Publish to TEST server first (recommended)
./publish_batch.sh --test

# Publish to PRODUCTION (be careful!)
./publish_batch.sh --prod

# Publish only npi-waves files
./publish_batch.sh --test --include npi-waves/

# Test with first 5 files
./publish_batch.sh --test --limit 5
```

Or use the Python script directly:

```bash
python3 publish_all_nanopubs.py --publish test --include npi-waves/ waves/
```

The script will:
- Find all publishable .trig files in npi-waves/ and other directories
- Skip files already listed in a manifest
- Sign and publish each nanopublication
- Track published files in a manifest for future runs

### PlanformDB Nanopubs

To generate and publish nanopubs from PlanformDB:

```bash
# Generate .trig files only
python3 planform_to_nanopubs.py --db planformDB_2.5.0.edb --out ./planform_nanopubs

# Generate and publish to test server
python3 planform_to_nanopubs.py --db planformDB_2.5.0.edb --out ./planform_nanopubs --publish test --limit 10
```

Then use the batch publisher to publish all generated files:

```bash
python3 publish_all_nanopubs.py --publish test --include planform_nanopubs/
```

## Command-Line Utilities

The `np` script provides convenient utilities for working with nanopubs:

```bash
# Check if a nanopub file is valid
./np check file.trig

# Create trusty URI version
./np mktrusty file.trig -o trusty/file.trig

# Sign a nanopub (produces signed.file.trig)
./np sign trusty/file.trig

# Check signed nanopub
./np check signed.file.trig

# Publish a nanopub
./np publish signed.file.trig
```

Or use the Python script directly:
```bash
python3 np_utils.py check file.trig
python3 np_utils.py mktrusty file.trig -o trusty/file.trig
python3 np_utils.py sign trusty/file.trig
python3 np_utils.py publish signed.file.trig
```

## Usage

See individual script files for detailed usage instructions and documentation.

