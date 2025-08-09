import os
import hmac
import hashlib
import json
import razorpay
from fastapi import HTTPException, status
from typing import Dict, Any
from dotenv import load_dotenv
load_dotenv()  # This loads the .env file
# Initialize Razorpay client
RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET")

if not RAZORPAY_KEY_ID or not RAZORPAY_KEY_SECRET:
    raise ValueError("Razorpay credentials not found in environment variables")

client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))

# Subscription plans configuration (in paise for INR)
SUBSCRIPTION_PLANS = {
    "basic": {
        "name": "Basic",
        "amount": 49900,  # ₹499
        "duration_days": 30,
        "word_limit": 10000,
        "description": "Basic subscription with 10,000 words per month"
    },
    "pro": {
        "name": "Pro",
        "amount": 99900,  # ₹999
        "duration_days": 30,
        "word_limit": 30000,
        "description": "Pro subscription with 30,000 words per month"
    },
    "enterprise": {
        "name": "Enterprise",
        "amount": 199900,  # ₹1,999
        "duration_days": 30,
        "word_limit": 100000,
        "description": "Unlimited words for power users"
    }
}

def create_order(plan_id: str, user_id: int) -> Dict[str, Any]:
    """
    Create a Razorpay order for subscription
    """
    if plan_id not in SUBSCRIPTION_PLANS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid subscription plan"
        )
    
    plan = SUBSCRIPTION_PLANS[plan_id]
    
    try:
        order = client.order.create({
            'amount': plan['amount'],
            'currency': 'INR',
            'receipt': f"order_{user_id}_{plan_id}",
            'payment_capture': '1',
            'notes': {
                'plan_id': plan_id,
                'user_id': user_id
            }
        })
        return {
            "order_id": order['id'],
            "amount": order['amount'],
            "currency": order['currency'],
            "key": RAZORPAY_KEY_ID,
            "plan_id": plan_id,
            "plan_name": plan['name']
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create order: {str(e)}"
        )

def verify_payment_signature(order_id: str, payment_id: str, signature: str) -> bool:
    """
    Verify the payment signature from Razorpay
    """
    try:
        generated_signature = hmac.new(
            RAZORPAY_KEY_SECRET.encode(),
            f"{order_id}|{payment_id}".encode(),
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(generated_signature, signature)
    except Exception:
        return False

def get_plan_details(plan_id: str) -> Dict[str, Any]:
    """
    Get details of a subscription plan
    """
    if plan_id not in SUBSCRIPTION_PLANS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Plan not found"
        )
    return SUBSCRIPTION_PLANS[plan_id]
