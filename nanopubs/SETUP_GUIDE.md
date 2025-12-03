# Nanopub Setup Guide

## Initial Setup

Before you can publish nanopublications, you need to set up your profile with RSA keys and your ORCID iD.

### Step 1: Set Up Nanopub Profile

Run the interactive setup command:

```bash
python3 -m nanopub setup --newkeys
```

This will:
1. Ask for your ORCID iD (e.g., `https://orcid.org/0000-0000-0000-0000`)
2. Ask for your name
3. Generate new RSA keys automatically (with `--newkeys` flag)
4. Store the configuration in `~/.nanopub/`

**Important:** Make sure to use the `--newkeys` flag so it generates the RSA keys for you!

### Step 2: Verify Setup

Check that your profile is configured correctly:

```bash
python3 -m nanopub profile
```

This should show your ORCID iD and key paths.

### Alternative: Manual Key Generation

If you prefer to generate keys manually first:

```bash
# Create the directory
mkdir -p ~/.nanopub

# Generate RSA key pair
ssh-keygen -t rsa -b 4096 -f ~/.nanopub/id_rsa -N ""

# Then run setup pointing to these keys
python3 -m nanopub setup \
    --orcid-id "https://orcid.org/YOUR-ORCID-ID" \
    --name "Your Name" \
    --keypair ~/.nanopub/id_rsa.pub ~/.nanopub/id_rsa
```

### Troubleshooting

**Error: FileNotFoundError for id_rsa.pub**

If you see this error, it means the RSA keys haven't been generated yet. Run:

```bash
python3 -m nanopub setup --newkeys
```

**Error: Permission denied**

Make sure the `~/.nanopub/` directory has the correct permissions:

```bash
chmod 700 ~/.nanopub
chmod 600 ~/.nanopub/id_rsa  # Private key should be read-only by owner
chmod 644 ~/.nanopub/id_rsa.pub  # Public key can be readable
```

### After Setup

Once setup is complete, you can use the batch publishing scripts:

```bash
# Preview what would be published
./publish_batch.sh --dry-run

# Publish to test server
./publish_batch.sh --test

# Publish to production
./publish_batch.sh --prod
```

