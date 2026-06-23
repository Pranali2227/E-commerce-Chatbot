import re
from typing import Dict, Any, List, Tuple

# Simple in-memory session database
# Format: { session_id: { state: str, history: [...], order_id: str, reported_issue: str } }
sessions: Dict[str, Dict[str, Any]] = {}

def get_or_create_session(session_id: str) -> Dict[str, Any]:
    if session_id not in sessions:
        sessions[session_id] = {
            "state": "GREETING",
            "history": [],
            "order_id": None,
            "reported_issue": None,
            "uploaded_image": None,
            "verdict_data": None
        }
    return sessions[session_id]

def reset_session(session_id: str) -> Dict[str, Any]:
    sessions[session_id] = {
        "state": "GREETING",
        "history": [],
        "order_id": None,
        "reported_issue": None,
        "uploaded_image": None,
        "verdict_data": None
    }
    return sessions[session_id]

# Quick replies lists depending on state
QUICK_REPLIES = {
    "GREETING": ["Track Order", "Return Policy", "File Return / Exchange", "Talk to Agent"],
    "CHATTING": ["Track Order", "Return Policy", "File Return / Exchange", "Talk to Agent"],
    "WAITING_FOR_UPLOAD": ["Cancel Return", "Talk to Agent"],
    "DEFECT_VERIFIED": ["Confirm Return Label", "Talk to Agent"],
    "DEFECT_REJECTED": ["Upload New Photo", "Talk to Agent"],
    "SUPPORT_HANDOFF": ["Restart Chat"]
}

def process_bot_response(session_id: str, user_message: str) -> Tuple[str, List[str], bool]:
    """
    Main dialogue engine processing incoming user messages based on current session state.
    Returns:
        bot_response: The text response to send.
        quick_replies: List of buttons to display.
        prompt_upload: Boolean indicating if the UI should open the file upload selector.
    """
    session = get_or_create_session(session_id)
    state = session["state"]
    msg_lower = user_message.lower()
    
    # Store user message in history
    session["history"].append({"sender": "user", "text": user_message})
    
    # Global cancellations / handoffs
    if "talk to agent" in msg_lower or "human" in msg_lower or "representative" in msg_lower:
        session["state"] = "SUPPORT_HANDOFF"
        bot_response = "I am transferring you to a human support agent. They will review our chat history and be with you in 1-2 minutes. Thank you for your patience!"
        return bot_response, QUICK_REPLIES["SUPPORT_HANDOFF"], False
        
    if "restart" in msg_lower or "reset" in msg_lower:
        reset_session(session_id)
        bot_response = "Hi there! I'm your E-commerce Support Bot. How can I help you today?"
        return bot_response, QUICK_REPLIES["GREETING"], False

    if "cancel return" in msg_lower or "cancel" in msg_lower:
        if state in ["WAITING_FOR_UPLOAD", "DEFECT_VERIFIED", "DEFECT_REJECTED"]:
            session["state"] = "CHATTING"
            session["reported_issue"] = None
            bot_response = "Return request cancelled. What else can I help you with?"
            return bot_response, QUICK_REPLIES["CHATTING"], False

    # 1. State: GREETING
    if state == "GREETING":
        session["state"] = "CHATTING"
        # If user starts with something standard, process it, otherwise welcome them
        if any(word in msg_lower for word in ["hi", "hello", "hey", "greetings"]):
            bot_response = "Hello! I am your E-commerce Support Assistant. How can I help you today? You can track an order, ask about returns, or file an exchange for a defective item."
            return bot_response, QUICK_REPLIES["CHATTING"], False
        # Fall through to CHATTING processing for non-greeting starting messages

    # 2. State: CHATTING (General queries)
    if session["state"] == "CHATTING":
        # Check for Policy queries first (avoids false-positive on "return policy")
        if "policy" in msg_lower or "how to return" in msg_lower or "rules" in msg_lower:
            bot_response = (
                "Our standard return policy allows returns within **30 days** of delivery. "
                "Items must be unused and in original packaging. "
                "If the item is **defective**, we provide free return shipping and immediate exchange once the defect is verified via our automated visual check."
            )
            return bot_response, QUICK_REPLIES["CHATTING"], False

        # Check for Order Tracking
        order_match = re.search(r'#(\d{5})', msg_lower)
        if order_match:
            order_id = order_match.group(1)
            session["order_id"] = f"#{order_id}"
            bot_response = f"Checking order #{order_id}... 📦 Status: **In Transit**. It is currently at the FedEx regional facility and is scheduled for delivery on Thursday, June 25."
            return bot_response, QUICK_REPLIES["CHATTING"], False
            
        if "track" in msg_lower or "order" in msg_lower:
            bot_response = "Sure! Please provide your 5-digit Order ID starting with a hashtag (e.g., `#12345`), and I will check its status for you."
            return bot_response, QUICK_REPLIES["CHATTING"], False

        # Check for Return/Exchange Intent
        if any(word in msg_lower for word in ["return", "exchange", "refund", "broken", "defective", "damaged", "scratch", "crack", "tear"]):
            session["state"] = "WAITING_FOR_UPLOAD"
            # Try to extract the issue
            defect_keywords = ["broken", "defective", "damaged", "scratched", "cracked", "torn", "shattered"]
            detected_issue = "Unspecified defect"
            for kw in defect_keywords:
                if kw in msg_lower:
                    detected_issue = f"Item is {kw}"
                    break
            session["reported_issue"] = detected_issue
            
            bot_response = (
                "I see you want to initiate a return or exchange due to a defective or damaged product. "
                "To process this automatically, our system uses **Visual Defect Verification**. "
                "Please upload a clear, well-lit photo of the defective item using the upload box in the chat window."
            )
            return bot_response, QUICK_REPLIES["WAITING_FOR_UPLOAD"], True
            
        # Standard fallback
        bot_response = (
            "I'm not sure I fully understand. I can help you with:\n"
            "• **Tracking an order** (type `#` followed by your 5-digit order ID)\n"
            "• **Return and Exchange policy**\n"
            "• **Filing a return/exchange** due to a product defect.\n"
            "What would you like to do?"
        )
        return bot_response, QUICK_REPLIES["CHATTING"], False

    # 3. State: WAITING_FOR_UPLOAD
    if state == "WAITING_FOR_UPLOAD":
        bot_response = (
            "I'm waiting for you to upload a photo of the defect. "
            "Please click the paperclip/upload icon in the chat bar to upload an image, or click 'Cancel Return' to go back."
        )
        return bot_response, QUICK_REPLIES["WAITING_FOR_UPLOAD"], True

    # 4. State: DEFECT_VERIFIED
    if state == "DEFECT_VERIFIED":
        if "confirm" in msg_lower or "yes" in msg_lower or "label" in msg_lower:
            session["state"] = "GREETING"
            bot_response = (
                "🎉 **Return Authorized!**\n"
                "I have generated your prepaid shipping label (Label ID: **RL-98742-XP**).\n"
                "A PDF has been emailed to you. Please pack the item securely, attach the label, and drop it off at any FedEx location. "
                "We will ship your replacement item immediately. Is there anything else I can help you with?"
            )
            return bot_response, QUICK_REPLIES["GREETING"], False
        else:
            bot_response = "Would you like me to generate your prepaid shipping label now? (Click 'Confirm Return Label' or type 'yes' to proceed)."
            return bot_response, QUICK_REPLIES["DEFECT_VERIFIED"], False

    # 5. State: DEFECT_REJECTED
    if state == "DEFECT_REJECTED":
        if "upload" in msg_lower or "photo" in msg_lower or "try again" in msg_lower:
            session["state"] = "WAITING_FOR_UPLOAD"
            bot_response = "Okay, please upload a clearer, well-lit close-up photo of the defect. Our system will analyze it again."
            return bot_response, QUICK_REPLIES["WAITING_FOR_UPLOAD"], True
        else:
            bot_response = (
                "If you believe this is an error, you can upload a new photo, or "
                "click **Talk to Agent** to transfer this request to our human review team."
            )
            return bot_response, QUICK_REPLIES["DEFECT_REJECTED"], False

    # 6. State: SUPPORT_HANDOFF
    if state == "SUPPORT_HANDOFF":
        bot_response = "A support agent will join the chat shortly. To restart the bot dialogue in the meantime, type 'restart'."
        return bot_response, QUICK_REPLIES["SUPPORT_HANDOFF"], False

    # Catch-all
    bot_response = "How else can I assist you today?"
    return bot_response, QUICK_REPLIES["GREETING"], False
