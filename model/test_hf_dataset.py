#!/usr/bin/env python3
"""
Quick test to verify Hugging Face dataset access.
"""

def test_hf_dataset_access():
    """Test that we can access the Hugging Face dataset."""
    print("Testing Hugging Face dataset access...")
    
    try:
        from datasets import load_dataset
        
        # Try to load just the first few samples to test access
        print("Loading dataset info from Hugging Face...")
        dataset = load_dataset("jxie/flickr8k", split="train", streaming=True)
        
        # Get first sample to verify structure
        first_sample = next(iter(dataset))
        
        print("✅ Dataset access successful!")
        print(f"   Sample keys: {list(first_sample.keys())}")
        print(f"   Image type: {type(first_sample.get('image', 'Not found'))}")
        print(f"   Caption type: {type(first_sample.get('caption', 'Not found'))}")
        
        if 'caption' in first_sample:
            caption = first_sample['caption']
            if isinstance(caption, list):
                print(f"   First caption: {caption[0] if caption else 'Empty list'}")
            else:
                print(f"   Caption: {str(caption)[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error accessing dataset: {e}")
        print("This might be due to:")
        print("1. Network connectivity issues")
        print("2. Hugging Face Hub being temporarily unavailable") 
        print("3. Dataset moved or renamed")
        print("\nAlternatives:")
        print("- Try again later")
        print("- Use a different Flickr8K dataset from Hugging Face")
        print("- Download manually and place in data/processed/")
        return False


if __name__ == "__main__":
    success = test_hf_dataset_access()
    if success:
        print("\n🎉 Hugging Face dataset integration is working!")
    else:
        print("\n⚠️ Dataset access test failed (but this doesn't break the model setup)") 