import torch
import sys

def convert_torchscript_to_state_dict(torchscript_path, output_path):
    print(f"Converting {torchscript_path} to {output_path}...")
    
    try:
        # Load TorchScript model
        model = torch.jit.load(torchscript_path)
        print("✓ TorchScript model loaded successfully")
        
        # Extract state dict
        state_dict = model.state_dict()
        print(f"✓ Extracted state dict with {len(state_dict)} parameters")
        
        # Save as state dict
        torch.save(state_dict, output_path)
        print(f"✓ Saved state dict to {output_path}")
        
        # Print some parameter names for verification
        print("\nParameter names (first 10):")
        for i, (name, param) in enumerate(list(state_dict.items())[:10]):
            print(f"  {i+1}. {name}: {param.shape}")
            
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_torchscript.py <input.pt> <output.pt>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    convert_torchscript_to_state_dict(input_file, output_file)