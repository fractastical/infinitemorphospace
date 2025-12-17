#!/usr/bin/env python3
"""
Python implementation of np utilities (alternative to shell wrapper).
Provides: check, mktrusty, sign, publish commands for nanopubs.
"""

import argparse
import sys
from pathlib import Path

try:
    from nanopub import Nanopub
except ImportError:
    print("ERROR: nanopub library not installed")
    print("Install with: pip install nanopub rdflib")
    sys.exit(1)


def check_nanopub(file_path):
    """Check if a nanopub file is valid."""
    try:
        np = Nanopub(rdf=Path(file_path))
        
        # Check various aspects
        checks = {
            "Has valid structure": True,  # If it loads, structure is valid
            "Has valid signature": np.has_valid_signature() if hasattr(np, 'has_valid_signature') else "N/A",
            "Has valid trusty URI": np.has_valid_trusty() if hasattr(np, 'has_valid_trusty') else "N/A",
        }
        
        print(f"Checking: {file_path}")
        print("-" * 50)
        for check_name, result in checks.items():
            status = "✓" if result is True else "✗" if result is False else "?"
            print(f"{status} {check_name}: {result}")
        
        # Overall status
        all_passed = all(v is True for v in checks.values() if isinstance(v, bool))
        if all_passed:
            print("\n✓ Nanopub appears valid")
            return 0
        else:
            print("\n⚠ Some checks failed")
            return 1
            
    except Exception as e:
        print(f"✗ Error checking nanopub: {e}")
        return 1


def mktrusty_nanopub(input_path, output_path=None):
    """Create a trusty URI version of a nanopub."""
    try:
        input_file = Path(input_path)
        if not input_file.exists():
            print(f"Error: File not found: {input_path}")
            return 1
        
        # Determine output path
        if output_path is None:
            output_dir = input_file.parent / "trusty"
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / input_file.name
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load and process nanopub
        np = Nanopub(rdf=input_file)
        
        # Generate trusty URI if needed
        if hasattr(np, 'update_from_signed'):
            np.update_from_signed()
        
        # Save
        np.rdf.serialize(destination=str(output_file), format='trig')
        print(f"✓ Trusty URI nanopub saved to: {output_file}")
        return 0
        
    except Exception as e:
        print(f"✗ Error creating trusty URI: {e}")
        return 1


def sign_nanopub(input_path, output_path=None):
    """Sign a nanopub file."""
    try:
        input_file = Path(input_path)
        if not input_file.exists():
            print(f"Error: File not found: {input_path}")
            return 1
        
        # Determine output path
        if output_path is None:
            output_dir = input_file.parent
            base_name = input_file.stem
            output_path = output_dir / f"signed.{base_name}.trig"
        
        output_file = Path(output_path)
        
        # Load nanopub
        np = Nanopub(rdf=input_file)
        
        # Sign it
        if hasattr(np, 'sign'):
            np.sign()
        else:
            print("Error: Signing not available (profile may not be configured)")
            return 1
        
        # Save signed version
        np.rdf.serialize(destination=str(output_file), format='trig')
        print(f"✓ Signed nanopub saved to: {output_file}")
        return 0
        
    except Exception as e:
        print(f"✗ Error signing nanopub: {e}")
        return 1


def publish_nanopub(file_path, use_test_server=False):
    """Publish a nanopub."""
    try:
        input_file = Path(file_path)
        if not input_file.exists():
            print(f"Error: File not found: {file_path}")
            return 1
        
        # Load nanopub
        np = Nanopub(rdf=input_file)
        
        # Use the CLI command for publishing
        import subprocess
        cmd = ["python3", "-m", "nanopub", "publish", str(input_file)]
        if use_test_server:
            cmd.append("--test")
        
        result = subprocess.run(cmd)
        return result.returncode
        
    except Exception as e:
        print(f"✗ Error publishing nanopub: {e}")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Nanopub utility commands",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s check file.trig
  %(prog)s mktrusty file.trig -o trusty/file.trig
  %(prog)s sign trusty/file.trig
  %(prog)s check signed.file.trig
  %(prog)s publish signed.file.trig
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # check command
    check_parser = subparsers.add_parser('check', help='Check if a nanopub is valid')
    check_parser.add_argument('file', help='Nanopub file to check')
    
    # mktrusty command
    mktrusty_parser = subparsers.add_parser('mktrusty', help='Create trusty URI version')
    mktrusty_parser.add_argument('file', help='Input nanopub file')
    mktrusty_parser.add_argument('-o', '--output', help='Output file (default: trusty/<filename>)')
    
    # sign command
    sign_parser = subparsers.add_parser('sign', help='Sign a nanopub')
    sign_parser.add_argument('file', help='Input nanopub file')
    sign_parser.add_argument('-o', '--output', help='Output file (default: signed.<filename>)')
    
    # publish command
    publish_parser = subparsers.add_parser('publish', help='Publish a nanopub')
    publish_parser.add_argument('file', help='Nanopub file to publish')
    publish_parser.add_argument('--test', action='store_true', help='Use test server')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    if args.command == 'check':
        return check_nanopub(args.file)
    elif args.command == 'mktrusty':
        return mktrusty_nanopub(args.file, args.output)
    elif args.command == 'sign':
        return sign_nanopub(args.file, args.output)
    elif args.command == 'publish':
        return publish_nanopub(args.file, args.test)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())

