#!/usr/bin/env python3
"""
Batch script to sign and publish all nanopublication files.

This script processes and publishes:
- All .trig files in waves/ (default)
- All .trig files from planform_to_nanopubs.py output
- Other publishable .trig files in the nanopubs directory

IMPORTANT NOTES:
- Only signed nanopubs are accepted by registries
- Bulk uploading is not officially supported by nanopub registries by design
- This script publishes one nanopub at a time with delays between requests
- For future bulk publishing, consider setting up your own Registry node
- Production servers may have stricter validation than test servers

USAGE
-----
# Dry-run: show what would be published (no actual publishing)
python publish_all_nanopubs.py --dry-run

# Publish to TEST server first (recommended)
python publish_all_nanopubs.py --publish test

# Publish to PRODUCTION (be careful!)
python publish_all_nanopubs.py --publish prod

# Publish to different production servers:
python publish_all_nanopubs.py --publish prod-kp        # KnowledgePixels production
python publish_all_nanopubs.py --publish prod-petapico  # Petapico production (default)
python publish_all_nanopubs.py --publish prod-trusty    # TrustyURI production

# Publish specific directories only
python publish_all_nanopubs.py --publish test --include waves/

# Skip files already in a manifest
python publish_all_nanopubs.py --publish test --skip-manifest published_manifest_test.txt
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

try:
    from nanopub import Nanopub, NanopubClient
    from rdflib import Dataset, Graph
    import requests
except ImportError as e:
    print("ERROR: Required libraries not installed.")
    print("Install with: pip install nanopub rdflib requests")
    print(f"Import error: {e}")
    print(f"Python path: {sys.executable}")
    sys.exit(1)


def load_trig_file(file_path):
    """Load a TriG file into an RDF graph."""
    try:
        from rdflib import Dataset
        graph = Dataset()
        graph.parse(str(file_path), format='trig')
        return graph
    except Exception as e:
        print(f"  ⚠ Warning: Failed to parse {file_path}: {e}")
        return None


def create_nanopub_from_trig(trig_path, graph):
    """Create a Nanopub object from a TriG file."""
    try:
        # The Nanopub class can load directly from a file path
        pub = Nanopub(rdf=Path(trig_path))
        return pub
    except Exception as e:
        print(f"  ⚠ Warning: Failed to create Nanopub from {trig_path}: {e}")
        return None


def find_publishable_files(base_dir, include_patterns=None, exclude_patterns=None):
    """
    Find all publishable .trig files.
    
    Args:
        base_dir: Base directory to search
        include_patterns: List of directory patterns to include (e.g., ['npi-waves/', 'planform_*/'])
        exclude_patterns: List of directory patterns to exclude
    
    Returns:
        List of Path objects to .trig files
    """
    base_path = Path(base_dir)
    trig_files = []
    
    # Default includes: waves (complete set) and planform-generated files
    # Note: npi-waves/ is excluded by default as it's a subset/alternative version
    default_includes = ['waves/', 'planform_*/']
    
    if include_patterns is None:
        include_patterns = default_includes
    
    # Find all .trig files
    for trig_file in base_path.rglob('*.trig'):
        # Skip hypothesis mapping files (these are metadata, not publishable nanopubs)
        if 'hypothesis_mapping' in trig_file.name.lower():
            continue
        
        # Skip files that are already signed (temporary signed files from previous runs)
        if trig_file.name.startswith('signed.'):
            continue
        
        # Check include patterns
        if include_patterns:
            rel_path = str(trig_file.relative_to(base_path))
            matched = False
            for pattern in include_patterns:
                pattern_clean = pattern.rstrip('/')
                # Handle glob patterns (ending with *)
                if pattern_clean.endswith('*'):
                    # Remove the * and trailing slash, match prefix
                    prefix = pattern_clean[:-1]
                    if rel_path.startswith(prefix):
                        matched = True
                        break
                # Handle directory patterns (must start with pattern as directory)
                elif rel_path.startswith(pattern_clean + '/') or rel_path == pattern_clean:
                    matched = True
                    break
            if not matched:
                continue
        
        # Check exclude patterns
        if exclude_patterns:
            rel_path = str(trig_file.relative_to(base_path))
            matched = any(
                pattern.rstrip('/') in rel_path or rel_path.startswith(pattern)
                for pattern in exclude_patterns
            )
            if matched:
                continue
        
        trig_files.append(trig_file)
    
    return sorted(trig_files)


def load_manifest(manifest_path):
    """Load a manifest of already published files."""
    if not manifest_path or not Path(manifest_path).exists():
        return set()
    
    published = set()
    with open(manifest_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Manifest format: filepath|nanopub_uri (or just filepath)
                parts = line.split('|')
                published.add(parts[0])
    
    return published


def save_to_manifest(manifest_path, file_path, nanopub_uri, server_info=None):
    """Save published file info to manifest."""
    manifest_dir = Path(manifest_path).parent
    manifest_dir.mkdir(parents=True, exist_ok=True)
    
    with open(manifest_path, 'a') as f:
        timestamp = datetime.now().isoformat()
        server_str = f"|{server_info}" if server_info else ""
        f.write(f"{file_path}|{nanopub_uri}|{timestamp}{server_str}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Batch publish nanopublication files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--publish',
        choices=['test', 'prod', 'prod-kp', 'prod-petapico', 'prod-trusty'],
        help='Publish destination: test, prod (defaults to petapico), prod-kp (KnowledgePixels), prod-petapico, or prod-trusty'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Dry run: show what would be published without actually publishing'
    )
    parser.add_argument(
        '--base-dir',
        default='.',
        help='Base directory to search for .trig files (default: current directory)'
    )
    parser.add_argument(
        '--include',
        nargs='+',
        help='Include only files in these directories (e.g., npi-waves/ waves/)'
    )
    parser.add_argument(
        '--exclude',
        nargs='+',
        help='Exclude files in these directories'
    )
    parser.add_argument(
        '--skip-manifest',
        help='Path to manifest file listing already published files'
    )
    parser.add_argument(
        '--manifest',
        default=None,
        help='Path to manifest file for tracking published files (default: published_manifest_test.txt or published_manifest_prod.txt based on --publish mode)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of files to process (useful for testing)'
    )
    
    args = parser.parse_args()
    
    # Determine if we're actually publishing
    if not args.publish and not args.dry_run:
        print("ERROR: Must specify either --publish (test/prod) or --dry-run")
        parser.print_help()
        sys.exit(1)
    
    # Set default manifest file based on publish mode (test vs prod)
    if args.manifest is None:
        if args.publish:
            # Normalize prod variants to just 'prod' for manifest naming
            manifest_mode = 'test' if args.publish == 'test' else 'prod'
            args.manifest = f'published_manifest_{manifest_mode}.txt'
        else:
            args.manifest = 'published_manifest_test.txt'  # Default for dry-run
    
    # Determine server name for manifest entry
    server_name_map = {
        'test': 'test_server',
        'prod': 'prod_petapico',
        'prod-petapico': 'prod_petapico',
        'prod-kp': 'prod_knowledgepixels',
        'prod-trusty': 'prod_trustyuri'
    }
    server_name = server_name_map.get(args.publish, 'unknown') if args.publish else 'unknown'
    
    print(f"Using manifest file: {args.manifest}")
    
    # Profile will be loaded automatically by Nanopub.publish() when needed
    # We don't need to load it explicitly here
    
    # Find all publishable files
    base_dir = Path(args.base_dir).resolve()
    print(f"\nSearching for .trig files in: {base_dir}")
    
    trig_files = find_publishable_files(
        base_dir,
        include_patterns=args.include,
        exclude_patterns=args.exclude
    )
    
    if not trig_files:
        print("No publishable .trig files found.")
        return
    
    print(f"Found {len(trig_files)} publishable file(s)")
    
    # Load manifest if provided
    already_published = set()
    if args.skip_manifest:
        already_published = load_manifest(args.skip_manifest)
        print(f"Loaded {len(already_published)} already published files from manifest")
    
    # Filter out already published files
    files_to_publish = []
    for trig_file in trig_files:
        rel_path = str(trig_file.relative_to(base_dir))
        if rel_path not in already_published:
            files_to_publish.append(trig_file)
        else:
            print(f"  ⊘ Skipping (already published): {rel_path}")
    
    # Apply limit if specified
    if args.limit:
        files_to_publish = files_to_publish[:args.limit]
        print(f"Limited to first {args.limit} files")
    
    if not files_to_publish:
        print("No files to publish (all already in manifest or filtered out).")
        return
    
    print(f"\n{'DRY RUN: ' if args.dry_run else ''}Processing {len(files_to_publish)} file(s)...")
    print("=" * 80)
    
    # Process each file
    successful = 0
    failed = 0
    
    for i, trig_file_original in enumerate(files_to_publish, 1):
        rel_path = str(trig_file_original.relative_to(base_dir))
        print(f"\n[{i}/{len(files_to_publish)}] {rel_path}")
        
        # Add a small delay between publications to avoid overwhelming the server
        # Note: Bulk uploading is not officially supported by nanopub registries
        if i > 1 and not args.dry_run:
            import time
            time.sleep(1)  # 1 second delay between publications
        
        trig_file = trig_file_original  # Keep original for reference
        
        if args.dry_run:
            print(f"  ✓ Would publish (dry-run mode)")
            successful += 1
        else:
            try:
                # Two-step process: sign first, then publish
                # This works around profile loading issues
                import subprocess
                import os
                
                private_key_path = os.path.expanduser('~/.nanopub/id_rsa')
                
                if not os.path.exists(private_key_path):
                    raise Exception("Private key not found at ~/.nanopub/id_rsa. "
                                  "Run 'python3 -m nanopub setup --newkeys'")
                
                # Step 1: Sign the nanopub first
                # The sign command creates "signed.{filename}.trig" in the same directory
                # Use absolute paths to avoid path resolution issues
                trig_file_abs = Path(trig_file).resolve()
                private_key_abs = Path(private_key_path).resolve()
                
                sign_cmd = [sys.executable, '-m', 'nanopub', 'sign', 
                           str(trig_file_abs), '--private-key', str(private_key_abs)]
                sign_result = subprocess.run(sign_cmd, capture_output=True, text=True)
                
                if sign_result.returncode != 0:
                    error_msg = sign_result.stderr or sign_result.stdout
                    # Provide more detailed error info
                    cmd_str = ' '.join(sign_cmd)
                    raise Exception(f"Signing failed.\nCommand: {cmd_str}\nError: {error_msg[:500]}")
                
                # The signed file is created as "signed.{filename}.trig" in the same directory
                signed_file = trig_file_abs.parent / f"signed.{trig_file_abs.name}"
                
                if not signed_file.exists():
                    raise Exception(f"Signing succeeded but signed file not found: {signed_file}")
                
                # Step 2: Use signed file for publishing
                trig_file = signed_file
                
                # Step 2: Publish the signed nanopub using direct HTTP upload
                # This bypasses the profile loading issue
                # Determine server URL using NanopubClient to get correct URLs
                from nanopub import NanopubClient
                
                if args.publish == 'test':
                    client = NanopubClient(use_test_server=True)
                    base_url = client.use_server
                elif args.publish == 'prod-kp':
                    # KnowledgePixels production server
                    base_url = 'https://registry.knowledgepixels.com/np/'
                elif args.publish == 'prod-trusty':
                    # TrustyURI production server
                    base_url = 'https://registry.np.trustyuri.net/np/'
                elif args.publish == 'prod' or args.publish == 'prod-petapico':
                    # Default: Petapico production server
                    client = NanopubClient(use_test_server=False)
                    base_url = client.use_server
                else:
                    # Fallback to default production
                    client = NanopubClient(use_test_server=False)
                    base_url = client.use_server
                
                # Upload the signed nanopub file directly
                # Match the format used by nanopub library: load as Dataset, serialize, then POST
                try:
                    # Load the file as a Dataset and serialize it (like the library does)
                    # This ensures proper formatting and URI resolution
                    from rdflib import Dataset
                    
                    g = Dataset()
                    g.parse(str(trig_file), format='trig')
                    trig_content = g.serialize(format='trig')
                    
                    # Post to the base server URL (not /pub endpoint)
                    # The library posts directly to use_server, not use_server/pub
                    headers = {'Content-Type': 'application/trig'}
                    # Ensure URL has trailing slash (required by server)
                    if not base_url.endswith('/'):
                        base_url = base_url + '/'
                    response = requests.post(base_url, headers=headers, data=trig_content.encode('utf-8'), timeout=60)
                    
                    # Check response status before raising exception
                    if response.status_code not in (200, 201):
                        # Try to get error message from response
                        error_msg = response.text.strip() if response.text else ''
                        if not error_msg:
                            # Check response headers for error info
                            error_hint = ''
                            if 'X-Error' in response.headers:
                                error_hint = f" (Header: {response.headers['X-Error']})"
                            elif 'Location' in response.headers:
                                error_hint = f" (Location: {response.headers['Location']})"
                            error_msg = f"Server rejected the nanopub{error_hint}"
                        raise Exception(f"HTTP {response.status_code}: {error_msg}")
                    
                    # Extract nanopub URI from response
                    nanopub_uri = response.text.strip() if response.text else 'published successfully'
                    
                    # Check for error messages in response body (even with 200 status)
                    if response.text and ('error' in response.text.lower() or 'failed' in response.text.lower()):
                        raise Exception(f"Server reported error: {response.text[:500]}")
                    
                    print(f"  ✓ Published: {nanopub_uri}")
                    
                    # Clean up temporary signed file
                    if signed_file.exists() and signed_file.name.startswith('signed.'):
                        try:
                            signed_file.unlink()
                        except:
                            pass
                            
                except requests.exceptions.RequestException as e:
                    # Provide more context if it's an HTTP error
                    if hasattr(e, 'response') and e.response is not None:
                        error_detail = e.response.text[:500] if e.response.text else 'No error details'
                        status_code = e.response.status_code
                        raise Exception(f"HTTP {status_code} error: {error_detail}")
                    raise Exception(f"HTTP upload failed: {str(e)[:300]}")
                
                # Save to manifest (use original path, not signed temp file)
                # Include server info in the manifest entry
                save_to_manifest(args.manifest, rel_path, nanopub_uri, server_name)
                
                successful += 1
            except Exception as e:
                print(f"  ✗ Failed to publish: {e}")
                failed += 1
    
    # Summary
    print("\n" + "=" * 80)
    print(f"Summary:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(files_to_publish)}")
    
    if not args.dry_run and successful > 0:
        print(f"\nPublished files recorded in: {args.manifest}")


if __name__ == "__main__":
    main()

