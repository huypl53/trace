"""
Simple test script to demonstrate using the TRACE FastAPI server.

Usage:
    python test_api.py <image_path>

Example:
    python test_api.py sample_image.jpg
"""
import argparse
import sys
import tarfile
from pathlib import Path

import requests


def test_trace_api(image_path: str, api_url: str = "http://localhost:8000/process"):
    """
    Upload an image to the TRACE API and save the results.
    
    Args:
        image_path: Path to the image file to process
        api_url: URL of the TRACE API endpoint
    """
    image_file = Path(image_path)
    
    if not image_file.exists():
        print(f"Error: Image file '{image_path}' not found")
        return False
    
    print(f"Uploading {image_path} to {api_url}...")
    
    try:
        with open(image_file, 'rb') as f:
            files = {'file': (image_file.name, f, 'image/jpeg')}
            response = requests.post(api_url, files=files, timeout=60)
        
        if response.status_code == 200:
            # Save the tar file
            output_file = "trace_results.tar.gz"
            with open(output_file, 'wb') as f:
                f.write(response.content)
            
            print(f"✓ Success! Results saved to {output_file}")
            
            # Extract and show contents
            print("\nExtracting contents:")
            with tarfile.open(output_file, 'r:gz') as tar:
                tar.extractall(path='./trace_output')
                members = tar.getmembers()
                for member in members:
                    print(f"  - {member.name} ({member.size} bytes)")
            
            print("\nFiles extracted to ./trace_output/")
            return True
        else:
            print(f"✗ Error: Server returned status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("✗ Error: Could not connect to the server. Is it running?")
        print("Start the server with: python app.py -c configs/train_trace_wtw.json -m weights/ckpt_2000.pth")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the TRACE API")
    parser.add_argument("image_path", help="Path to the image file to process")
    parser.add_argument("--url", default="http://localhost:8000/process", help="API endpoint URL")
    
    args = parser.parse_args()
    
    success = test_trace_api(args.image_path, args.url)
    sys.exit(0 if success else 1)

