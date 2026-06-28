"""
GlowWise AI - Manual Backend API Integration Test
Tests backend FastAPI endpoints:
1. GET /health
2. GET /api/model/status
3. POST /api/predict/satisfaction (positive review)
4. POST /api/predict/satisfaction (negative review)
5. GET /api/insights/summary
6. GET /api/insights/top-terms
7. GET /api/insights/clusters
"""

import requests
import json
import sys

BASE_URL = "http://127.0.0.1:8000"

def test_endpoint(method, path, data=None):
    url = f"{BASE_URL}{path}"
    print(f"\n--- Testing {method} {path} ---")
    try:
        if method == "GET":
            response = requests.get(url, timeout=5)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=5)
        else:
            print(f"Unsupported method: {method}")
            return False
            
        print(f"Status Code: {response.status_code}")
        print("Response Content:")
        print(json.dumps(response.json(), indent=4))
        return response.status_code in (200, 201)
    except requests.exceptions.ConnectionError:
        print(f"[ERROR] Could not connect to the API server at {BASE_URL}.")
        print("Please make sure your FastAPI application is running locally:")
        print("   1. Open a command prompt, navigate to backend/")
        print("   2. Run: uvicorn app.main:app --reload")
        return False
    except Exception as e:
        print(f"[ERROR] Request failed: {e}")
        return False

def main():
    print("=== GlowWise AI Backend API Integration Manual Test ===")
    
    # 1. Health check
    h_ok = test_endpoint("GET", "/health")
    
    # 2. Model Status
    s_ok = test_endpoint("GET", "/api/model/status")
    
    # 3. Predict Satisfaction (Positive Review)
    pos_data = {
        "review_text": "I absolutely love this moisturizer! It makes my skin feel incredibly smooth and glowing. Highly recommend it.",
        "review_title": "Holy grail moisturizer!",
        "product_name": "Ultra Facial Cream",
        "brand_name": "Kiehl's"
    }
    p_pos_ok = test_endpoint("POST", "/api/predict/satisfaction", pos_data)
    
    # 4. Predict Satisfaction (Negative Review)
    neg_data = {
        "review_text": "Highly disappointed with this product. It broke me out in acne all over my face within two days and left a greasy residue.",
        "review_title": "Broke me out",
        "product_name": "Super Hydrator Gel",
        "brand_name": "Murad"
    }
    p_neg_ok = test_endpoint("POST", "/api/predict/satisfaction", neg_data)
    
    # 5. Insights Summary
    sum_ok = test_endpoint("GET", "/api/insights/summary")
    
    # 6. Insights Top Terms
    terms_ok = test_endpoint("GET", "/api/insights/top-terms")
    
    # 7. Insights Clusters
    clust_ok = test_endpoint("GET", "/api/insights/clusters")
    
    print("\n=== Test Executions Summary ===")
    print(f"Health Check:     {'[PASSED]' if h_ok else '[FAILED]'}")
    print(f"Model Status:     {'[PASSED]' if s_ok else '[FAILED]'}")
    print(f"Prediction (Pos): {'[PASSED]' if p_pos_ok else '[FAILED]'}")
    print(f"Prediction (Neg): {'[PASSED]' if p_neg_ok else '[FAILED]'}")
    print(f"Insights Summary: {'[PASSED]' if sum_ok else '[FAILED]'}")
    print(f"Insights Terms:   {'[PASSED]' if terms_ok else '[FAILED]'}")
    print(f"Insights Clusters:{'[PASSED]' if clust_ok else '[FAILED]'}")
    
    if not (h_ok and s_ok and p_pos_ok and p_neg_ok and sum_ok and terms_ok and clust_ok):
        print("\nSome tests failed or were skipped. Check connection or model status.")
        sys.exit(1)
    else:
        print("\nAll tests completed successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()
