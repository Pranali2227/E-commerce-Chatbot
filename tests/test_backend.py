import io
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
import numpy as np

def run_tests():
    print("=== AuraShop Bot Backend Tests ===")
    
    # 1. Test imports
    try:
        from app.vision import preprocess_image, run_defect_inference
        from app.bot import process_bot_response, get_or_create_session, reset_session
        print("[PASS] Successfully imported all backend modules.")
    except Exception as e:
        print(f"[FAIL] Module import failed: {e}")
        return False

    # 2. Test Computer Vision pipeline
    try:
        # Create a mock 300x300 red image with a black slash (simulating crack)
        img = Image.new("RGB", (300, 300), color=(200, 100, 100))
        # Draw a black line (crack)
        pixels = img.load()
        for i in range(100, 200):
            pixels[i, i] = (0, 0, 0)
            pixels[i+1, i] = (0, 0, 0)
            
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_bytes = img_byte_arr.getvalue()
        
        # Run preprocessing
        img_norm, stats, logs = preprocess_image(img_bytes)
        
        print("\n--- CV Preprocessing Output ---")
        print(f"Tensor shape: {img_norm.shape}")
        print(f"Stats: {stats}")
        print(f"Logs count: {len(logs)}")
        
        assert img_norm.shape == (224, 224, 3), "Tensor shape must be (224, 224, 3)"
        assert "brightness" in stats, "Stats must contain brightness"
        assert "contrast" in stats, "Stats must contain contrast"
        assert "edge_density" in stats, "Stats must contain edge_density"
        print("[PASS] CV Preprocessing tests completed successfully.")
        
        # Run inference
        result = run_defect_inference(img_norm, stats, "defect_camera_shattered.jpg", "Item is cracked")
        print("\n--- CV Model Inference Output ---")
        print(f"Verdict: {result['verdict']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Details: {result['details']}")
        print(f"Verified: {result['verified']}")
        
        assert result['verified'] is True, "Mock image with 'defect' in name and edges should be verified"
        print("[PASS] CV Model Inference tests completed successfully.")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[FAIL] CV pipeline tests failed: {e}")
        return False
        
    # 3. Test Bot Dialog Engine
    try:
        session_id = "test_user_session"
        reset_session(session_id)
        
        print("\n--- Bot Dialogue State Flow ---")
        
        # Test Greeting
        resp, quick, upload = process_bot_response(session_id, "Hello")
        print(f"User: Hello -> Bot: {resp[:60]}... [Upload={upload}]")
        session = get_or_create_session(session_id)
        assert session["state"] == "CHATTING", "Should transition to CHATTING"
        
        # Test General query
        resp, quick, upload = process_bot_response(session_id, "What is your return policy?")
        print(f"User: return policy? -> Bot: {resp[:60]}... [Upload={upload}]")
        
        # Test Return Request Trigger
        resp, quick, upload = process_bot_response(session_id, "I need to return a broken product")
        print(f"User: broken product -> Bot: {resp[:60]}... [Upload={upload}]")
        session = get_or_create_session(session_id)
        assert session["state"] == "WAITING_FOR_UPLOAD", "Should transition to WAITING_FOR_UPLOAD"
        assert upload is True, "Upload prompt should be active"
        
        print("[PASS] Bot dialogue engine tests completed successfully.")
    except Exception as e:
        print(f"[FAIL] Bot dialogue engine tests failed: {e}")
        return False
        
    print("\n======================================")
    print("ALL BACKEND MODULE TESTS COMPLETED: PASS")
    print("======================================")
    return True

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
